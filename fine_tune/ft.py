import sys, os
import math
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
import logging
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, dataset_and_transform_generate
from utils.aggregate_block.fix_random import fix_random
from utils.bd_dataset import prepro_cls_DatasetBD

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from load_data import CustomDataset_v2
from utils.save_load_attack import load_attack_result
from utils.choose_index import choose_index_v2
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from test import test


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def compute_fisher_information(model, dataloader, device, criterion):
    """
    Compute Fisher Information for each parameter in the model.
    Fisher Information I_w = E[(dL/dw)^2] approximates the importance of each parameter.
    """
    model.eval()
    fisher_dict = {}
    
    # Initialize fisher dict with zeros
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)
    
    num_samples = 0
    for batch_idx, (x, labels, *additional_info) in enumerate(dataloader):
        x, labels = x.to(device), labels.to(device)
        model.zero_grad()
        
        output = model(x)
        loss = criterion(output, labels.long())
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) * x.size(0)
        
        num_samples += x.size(0)
    
    # Average over all samples
    for name in fisher_dict:
        fisher_dict[name] /= num_samples
    
    return fisher_dict


def generate_zone_masks(model, fisher_dict, top_ratio=0.3, bottom_ratio=0.3, linear_name='linear'):
    """
    Generate three masks based on Fisher Information scores:
    - M_freeze: Top `top_ratio` parameters (Anchor Zone)
    - M_perturb: Middle parameters (Perturbation Zone)  
    - M_reset: Bottom `bottom_ratio` parameters (Purge Zone)
    
    Note: Linear classifier layer is forced to Purge Zone (fully reset).
    """
    # Flatten all Fisher scores
    all_fisher_scores = []
    param_info = []  # Store (name, flat_idx_start, flat_idx_end, shape)
    
    flat_idx = 0
    for name, param in model.named_parameters():
        fisher_flat = fisher_dict[name].flatten()
        all_fisher_scores.append(fisher_flat)
        param_info.append((name, flat_idx, flat_idx + fisher_flat.numel(), param.shape))
        flat_idx += fisher_flat.numel()
    
    all_fisher_scores = torch.cat(all_fisher_scores)
    total_params = all_fisher_scores.numel()
    
    # Sort Fisher scores and find thresholds
    sorted_scores, _ = torch.sort(all_fisher_scores)
    
    bottom_threshold_idx = int(total_params * bottom_ratio)
    top_threshold_idx = int(total_params * (1 - top_ratio))
    
    bottom_threshold = sorted_scores[bottom_threshold_idx].item() if bottom_threshold_idx < total_params else float('inf')
    top_threshold = sorted_scores[top_threshold_idx].item() if top_threshold_idx < total_params else float('inf')
    
    # Generate masks for each parameter
    mask_freeze = {}
    mask_perturb = {}
    mask_reset = {}
    
    for name, param in model.named_parameters():
        fisher_scores = fisher_dict[name]
        
        # Force linear classifier to Purge Zone (fully reset)
        if linear_name in name:
            mask_freeze[name] = torch.zeros_like(fisher_scores).float()
            mask_perturb[name] = torch.zeros_like(fisher_scores).float()
            mask_reset[name] = torch.ones_like(fisher_scores).float()
        else:
            # Top zone (freeze): Fisher >= top_threshold
            mask_freeze[name] = (fisher_scores >= top_threshold).float()
            
            # Bottom zone (reset): Fisher <= bottom_threshold
            mask_reset[name] = (fisher_scores <= bottom_threshold).float()
            
            # Middle zone (perturb): Everything else
            mask_perturb[name] = ((fisher_scores > bottom_threshold) & (fisher_scores < top_threshold)).float()
    
    return mask_freeze, mask_perturb, mask_reset


def apply_zone_initialization(model, mask_freeze, mask_perturb, mask_reset, sigma=0.1):
    """
    Apply mixed initialization based on zone masks:
    - Anchor Zone (freeze): Keep original weights
    - Perturbation Zone (perturb): Add Gaussian noise
    - Purge Zone (reset): Re-initialize with Kaiming/Xavier
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            original_weight = param.data.clone()
            
            # Generate noise for perturbation zone
            noise = torch.randn_like(param.data) * sigma * torch.norm(param.data)
            perturbed_weight = original_weight + noise
            
            # Generate re-initialized weights for reset zone (Kaiming for Conv, Xavier for Linear)
            reset_weight = torch.zeros_like(param.data)
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Kaiming initialization for conv/linear weights
                    nn.init.kaiming_uniform_(reset_weight, a=math.sqrt(5))
                else:
                    # For 1D weights (like BN), use uniform
                    fan_in = param.shape[0]
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(reset_weight, -bound, bound)
            elif 'bias' in name:
                fan_in = param.shape[0]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(reset_weight, -bound, bound)
            
            # Combine: w_new = w_old * M_freeze + (w_old + noise) * M_perturb + init() * M_reset
            param.data = (original_weight * mask_freeze[name] + 
                         perturbed_weight * mask_perturb[name] + 
                         reset_weight * mask_reset[name])


def compute_parameter_consistency_loss(model, teacher_model, mask_perturb, linear_name='linear'):
    """
    Compute L2 loss between student and teacher parameters in the perturbation zone.
    Only constrains parameters that were perturbed (Middle Zone).
    Excludes linear classifier layer, which is fully re-initialized.
    
    L_param = sum_{w in Middle Zone, w not in linear} ||w - w^T0||_2^2
    
    This prevents the perturbed neurons from deviating too far from the original,
    helping them recover benign features while disrupting backdoor patterns.
    """
    param_loss = 0.0
    for name, param in model.named_parameters():
        # Skip linear classifier layer (fully re-initialized, no constraint needed)
        if linear_name in name:
            continue
        if name in mask_perturb:
            teacher_param = teacher_model.state_dict()[name]
            # Only compute loss for parameters in perturbation zone (mask_perturb == 1)
            diff = (param - teacher_param) * mask_perturb[name]
            param_loss += torch.sum(diff ** 2)
    return param_loss


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--ft_mode', type = str, default='all')
    
    parser.add_argument('--attack', type = str)
    parser.add_argument('--attack_label_trans', type=str, default='all2one',
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str,default='../data')
    parser.add_argument('--attack_target', type=int,default=0,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type=int,default=128)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--random_seed', default=0,type=int,
                        help='random_seed')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    
    parser.add_argument('--split_ratio', type=float,
                        help='part of the training set for defense')
    
    parser.add_argument('--log', action='store_true',
                        help='record the log')
    parser.add_argument('--pre', action='store_true', help='load pre-trained weights')
    parser.add_argument('--save', action='store_true', help='save the model checkpoint')
    parser.add_argument('--linear_name', type=str, default='linear', help='name for the linear classifier')
    parser.add_argument('--lb_smooth', type=float, default=None, help='label smoothing')
    parser.add_argument('--alpha', type=float, default=0.2, help='fst')
    
    # Arguments for fzp (Fisher Zone Purification) mode
    parser.add_argument('--fzp_sigma', type=float, default=0.1, help='noise scale for perturbation zone in fzp')
    parser.add_argument('--fzp_lambda', type=float, default=0.01, help='weight for parameter consistency loss in fzp (L2 regularization on perturbed zone)')
    parser.add_argument('--fzp_top_ratio', type=float, default=0.3, help='top ratio for anchor zone (freeze)')
    parser.add_argument('--fzp_bottom_ratio', type=float, default=0.3, help='bottom ratio for purge zone (reset)')
    return parser

def main():

    ### 1. config args, save_path, fix random seed
    
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    
    fix_random(args.random_seed)
    
    
    if args.lb_smooth is not None:
        lbs_criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.lb_smooth)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.ft_mode == 'fe-tuning':
        init = True
    elif args.ft_mode == 'ft-init':
        init = True
    elif args.ft_mode == 'ft':
        init = False
    elif args.ft_mode == 'lp':
        init = False
    elif args.ft_mode == 'fst':
        assert args.alpha is not None
        init = True
    elif args.ft_mode == 'fzp':
        # Fisher Zone Purification - special initialization handled separately
        init = False  # Will be handled by zone-based initialization
    else:
        raise NotImplementedError('Not implemented method.')

    if not args.pre:
        
        folder_path = folder_path = f'../record/{args.dataset}/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}'
        os.makedirs(f'../logs/{args.dataset}/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}', exist_ok=True)
        save_path = f'../logs/{args.dataset}/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}'
    else:
        folder_path = f'../record/{args.dataset}_pre/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}'
        os.makedirs(f'../logs/{args.dataset}_pre/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}', exist_ok=True)
        save_path = f'../logs/{args.dataset}_pre/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}'
        
        
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    if args.log:
        fileHandler = logging.FileHandler(save_path + 'log.txt')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)


    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))


    ### 2. set the clean train data and clean test data
   
    _, train_img_transform, \
                train_label_transfrom, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)
    
    result = load_attack_result(folder_path + '/attack_result.pt')
    clean_dataset = prepro_cls_DatasetBD_v2(result['clean_train'].wrapped_dataset)
    data_all_length = len(clean_dataset)
    ran_idx = choose_index_v2(args.split_ratio, data_all_length) 
    
    clean_dataset.subset(ran_idx)
    data_set_clean = result['clean_train']
    data_set_clean.wrapped_dataset = clean_dataset
    data_set_clean.wrap_img_transform = train_img_transform
    benign_train_ds = data_set_clean

    

    benign_test_ds = prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        )


    model_dict = torch.load(folder_path + '/attack_result.pt')
    adv_test_dataset = model_dict['bd_test']
    
   
    import glob
    image_list = glob.glob(folder_path + '/bd_test_dataset/*/*.png')
    adv_test_dataset = CustomDataset_v2(image_list, args.attack_target, test_img_transform)

    ### 3. generate dataset for backdoor defense and evaluation

    train_data = DataLoader(
            dataset = benign_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    
    test_dataset_dict={
                "test_data" :benign_test_ds,
                "adv_test_data" :adv_test_dataset,
        }

    test_dataloader_dict = {
            name : DataLoader(
                    dataset = test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
            for name, test_dataset in test_dataset_dict.items()
        }   
    
    if not args.pre:
        net  = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
    else:
        
        if args.model == "resnet18":        
            from torchvision.models import resnet18, ResNet18_Weights        
            net = resnet18().to(device)
            net.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True).to(device) 
            
        elif args.model == "resnet50":        
            from torchvision.models import resnet50, ResNet50_Weights        
            net = resnet50().to(device)    
            net.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True).to(device) 
            
        elif args.model == 'swin_b':
            from torchvision.models import swin_b        
            net = swin_b().to(device)
            net.head = nn.Linear(in_features=1024, out_features=args.num_classes, bias=True).to(device) 
            
        elif args.model == 'swin_t':        
            from torchvision.models import swin_t        
            net = swin_t().to(device)
            net.head = nn.Linear(in_features=768, out_features=args.num_classes, bias=True).to(device) 
            
        else:        
            raise NotImplementedError(f"{args.model} is not supported")

    net.load_state_dict(model_dict['model'])   
    net.to(device)

    for dl_name, test_dataloader in test_dataloader_dict.items():
        metrics = test(net, test_dataloader, device)
        metric_info = {
            f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
            f'{dl_name} loss': metrics['test_loss'],
        }
        if 'test_data' == dl_name:
            cur_clean_acc = metric_info['test_data acc']
        if 'adv_test_data' == dl_name:
            cur_adv_acc = metric_info['adv_test_data acc']
    logging.info('*****************************')
    logging.info(f"Load from {folder_path + '/attack_result.pt'}")
    logging.info(f'Fine-tunning mode: {args.ft_mode}')
    logging.info('Original performance')
    logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logging.info('*****************************')


    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')

    # FZP (Fisher Zone Purification) specific initialization
    teacher_model = None
    mask_freeze = None
    mask_perturb = None
    
    if args.ft_mode == 'fzp':
        import copy
        logging.info('='*50)
        logging.info('Fisher Zone Purification (FZP) Mode')
        logging.info('='*50)
        
        # Step 1: Create teacher model (frozen copy of original model)
        logging.info('Step 1: Creating teacher model (frozen copy for parameter consistency)...')
        teacher_model = copy.deepcopy(net)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Step 2: Compute Fisher Information using 5% clean data
        logging.info('Step 2: Computing Fisher Information for all parameters...')
        fisher_criterion = nn.CrossEntropyLoss()
        fisher_dict = compute_fisher_information(net, train_data, device, fisher_criterion)
        
        # Log Fisher statistics
        total_fisher = sum(f.sum().item() for f in fisher_dict.values())
        logging.info(f'Total Fisher Information: {total_fisher:.6f}')
        
        # Step 3: Generate zone masks based on Fisher scores
        logging.info(f'Step 3: Generating zone masks (Top {args.fzp_top_ratio*100:.0f}% freeze, '
                    f'Middle {(1-args.fzp_top_ratio-args.fzp_bottom_ratio)*100:.0f}% perturb, '
                    f'Bottom {args.fzp_bottom_ratio*100:.0f}% reset)...')
        mask_freeze, mask_perturb, mask_reset = generate_zone_masks(
            net, fisher_dict, 
            top_ratio=args.fzp_top_ratio, 
            bottom_ratio=args.fzp_bottom_ratio,
            linear_name=args.linear_name
        )
        
        # Log zone statistics
        total_params = sum(p.numel() for p in net.parameters())
        freeze_count = sum(m.sum().item() for m in mask_freeze.values())
        perturb_count = sum(m.sum().item() for m in mask_perturb.values())
        reset_count = sum(m.sum().item() for m in mask_reset.values())
        logging.info(f'Anchor Zone (freeze): {freeze_count:.0f} params ({freeze_count/total_params*100:.1f}%)')
        logging.info(f'Perturbation Zone (noise): {perturb_count:.0f} params ({perturb_count/total_params*100:.1f}%)')
        logging.info(f'Purge Zone (reset): {reset_count:.0f} params ({reset_count/total_params*100:.1f}%)')
        
        # Step 4: Apply mixed initialization
        logging.info(f'Step 4: Applying mixed initialization (sigma={args.fzp_sigma})...')
        logging.info(f'Note: Linear classifier ({args.linear_name}) is forced to Purge Zone (fully reset)')
        apply_zone_initialization(net, mask_freeze, mask_perturb, mask_reset, sigma=args.fzp_sigma)
        
        logging.info('='*50)
        logging.info('FZP initialization complete. Starting recovery tuning...')
        logging.info(f'Parameter consistency regularization: lambda={args.fzp_lambda}')
        logging.info('='*50)

    param_list = []
    for name, param in net.named_parameters():
        if args.linear_name in name:
            if init:
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1)) 
                    param.data.uniform_(-std, std)
                    
                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
        if args.ft_mode == 'lp':
            if args.linear_name in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif args.ft_mode == 'ft' or args.ft_mode == 'fst' or args.ft_mode == 'ft-init':
            param.requires_grad = True
            param_list.append(param)
        elif args.ft_mode == 'fe-tuning':
            if args.linear_name not in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif args.ft_mode == 'fzp':
            # For FZP: freeze anchor zone, allow updates for perturb and reset zones
            if mask_freeze is not None and mask_freeze[name].sum() == mask_freeze[name].numel():
                # Fully frozen parameter (all in anchor zone)
                param.requires_grad = False
            else:
                # Has some trainable parameters
                param.requires_grad = True
                param_list.append(param)
        
        

    optimizer = optim.SGD(param_list, lr=args.lr,momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        
    
        logging.info(f'Epoch: {epoch}')
        net.train()

        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):


            x, labels = x.to(device), labels.to(device)
            log_probs= net(x)
            if args.lb_smooth is not None:
                loss = lbs_criterion(log_probs, labels)
            else:
                if args.ft_mode == 'fst':
                    loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels.long())
                elif args.ft_mode == 'fzp':
                    # FZP: CE loss + Parameter Consistency Regularization
                    ce_loss = criterion(log_probs, labels.long())
                    
                    # Compute parameter consistency loss for perturbation zone
                    # This prevents perturbed neurons from deviating too far from teacher
                    # Note: Linear classifier is excluded (fully re-initialized)
                    param_loss = torch.tensor(0.0, device=device)
                    if teacher_model is not None and mask_perturb is not None:
                        param_loss = compute_parameter_consistency_loss(net, teacher_model, mask_perturb, args.linear_name)
                    
                    loss = ce_loss + args.fzp_lambda * param_loss
                    
                    # Log losses periodically
                    if batch_idx % 50 == 0:
                        logging.debug(f'Batch {batch_idx}: CE={ce_loss.item():.4f}, ParamLoss={param_loss.item():.4f}')
                else:
                    loss = criterion(log_probs, labels.long())
            loss.backward()
            
            # For FZP: Apply gradient masking to respect zone boundaries
            if args.ft_mode == 'fzp' and mask_freeze is not None:
                with torch.no_grad():
                    for name, param in net.named_parameters():
                        if param.grad is not None and name in mask_freeze:
                            # Zero out gradients for frozen (anchor zone) parameters
                            param.grad.data *= (1 - mask_freeze[name])
            
            optimizer.step()
            optimizer.zero_grad()

            exec_str = f'net.{args.linear_name}.weight.data = net.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(net.{args.linear_name}.weight.data)'
            exec(exec_str)

            _, predicted = torch.max(log_probs, -1)
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            batch_loss = loss.item() * labels.size(0)
            batch_loss_list.append(batch_loss)

    
        scheduler.step()
        one_epoch_loss = sum(batch_loss_list)


        logging.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logging.info('-------------------------------------')
        
        if epoch == args.epochs-1:
            for dl_name, test_dataloader in test_dataloader_dict.items():
                metrics = test(net, test_dataloader, device)
                metric_info = {
                    f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dl_name} loss': metrics['test_loss'],
                }
                if 'test_data' == dl_name:
                    cur_clean_acc = metric_info['test_data acc']
                if 'adv_test_data' == dl_name:
                    cur_adv_acc = metric_info['adv_test_data acc']
            logging.info('Defense performance')
            logging.info(f"Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}") 
            logging.info('-------------------------------------')
    
    if args.save:
        model_save_path = folder_path + f'/defense/{args.ft_mode}'
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(net.state_dict(), f'{model_save_path}/defense_result.pt')
        
    
if __name__ == '__main__':
    main()
    

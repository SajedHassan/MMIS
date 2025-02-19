import torch

from tensorboardX import SummaryWriter

import os
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import shutil
import random
from configs.config import *
from evaluate_dp_npc_stag2_single_path import validate
from utils.logger import Logger
from utils.utils import rand_seed
from dataloader.dataset import RandomGenerator_Multi_Rater, BaseDataSets, ZoomGenerator
from torch.utils.data import DataLoader
from lib.initialize_model_single_path import init_model
from lib.initialize_optimization_single_path import init_optimization
import re

config_path = '/home/sajed/thesis/MMIS/D-Persona/code/configs/params_npc.yaml'
opt = Config(config_path=config_path)

def worker_init_fn(worker_id):
    return random.seed(opt.RANDOM_SEED + worker_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/home/sajed/thesis/MMIS/D-Persona/code/configs/params_npc.yaml', help="config path (*.yaml)")
    parser.add_argument("--save_path", type=str, help="save path", default='')
    parser.add_argument("--model_name", type=str, default='DPersona')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu", type=str, default='0')

    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--prior_sample_num", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--val_num", type=int, default=10)
    parser.add_argument("--mask_num", type=int, default=4)
    parser.add_argument("--RESUME_FROM", type=int, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    rand_seed(opt.RANDOM_SEED)

    # log & model folder
    if args.save_path == '':
        opt.MODEL_DIR += args.model_name + '{}_{}_{}'.format(args.stage, opt.DATASET, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        opt.MODEL_DIR = args.save_path

    if not os.path.exists(opt.MODEL_DIR):
        os.mkdir(opt.MODEL_DIR)

    logger = Logger(args.model_name, path=opt.MODEL_DIR)
    writer = SummaryWriter(opt.MODEL_DIR)

    shutil.copytree('/home/sajed/thesis/MMIS/D-Persona/code/', opt.MODEL_DIR + '/code/', shutil.ignore_patterns(['.git','__pycache__']))

     # dataset
    db_train = BaseDataSets(
        base_dir=opt.DATA_PATH,
        split="train",
        transform=RandomGenerator_Multi_Rater(opt.PATCH_SIZE)
    )

    db_val = BaseDataSets(
        base_dir=opt.DATA_PATH,
        split="val",
        transform=ZoomGenerator(opt.PATCH_SIZE)
    )

    train_loader = DataLoader(db_train, batch_size=opt.TRAIN_BATCHSIZE, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=opt.VAL_BATCHSIZE, shuffle=True, num_workers=1)

    # Training Config
    epochs = args.epochs
    epoch_start = 0

    net = init_model(args, opt)
    loss_fct, optimizer1, optimizer2, optimizer3, optimizer4 = init_optimization(net, args)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer4, T_max=epochs)
    ]

    # Resume
    if args.RESUME_FROM > 0:
        ckpt = torch.load(os.path.join(opt.MODEL_DIR, '{}{}_{}_{}.pth'.format(args.model_name, args.stage, opt.DATASET, args.RESUME_FROM)))
        net.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt.keys():
            optimizers[0].load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt.keys():
            schedulers[0].load_state_dict(ckpt['scheduler'])
        epoch_start = args.RESUME_FROM

    if args.stage == 2:
        # 1 Org
        # ckpt = torch.load(os.path.join('/home/sajed/thesis/MMIS/D-Persona/models/DPersona1_NPC_20241111-060635/DPersona1_NPC_best.pth'))
        # 1 org + 3 generated iou 100
        # ckpt = torch.load(os.path.join('/home/sajed/thesis/MMIS/D-Persona/models/DPersona1_NPC_20241108-235525/DPersona1_NPC_best.pth'))
        # 4 Org
        ckpt = torch.load(os.path.join('/home/sajed/thesis/MMIS/D-Persona/models/DPersona1_NPC_20241102-151012/DPersona1_NPC_best.pth'))

        net.load_state_dict(ckpt['model'], strict=False)

    net.cuda()

    # Training
    best_metric = 0
    for epoch in range(epoch_start, epochs):
        net.train()
        print_str = '-------epoch {}/{}-------'.format(epoch+1, epochs)
        logger.write_and_print(print_str)

        for step, sample in enumerate(tqdm(train_loader)):

            patch = sample['image'].cuda()
            mask = sample['label'].cuda()
            annotators = torch.tensor([int(int(re.search(r"Sample_(\d+)", ann).group(1)) / 25) for ann in sample['idx']])

            # prepare data
            batches_done = len(train_loader) * epoch + step

            patch_per_annotator = []
            mask_per_annotator = []
            for annotator in range(4):  # Annotators are 0, 1, 2, 3
                current_annotator = annotators == annotator  # Boolean mask for the current annotator
                filtered_patch = patch[current_annotator]     # Select the corresponding data
                patch_per_annotator.append(filtered_patch)
                filtered_mask = mask[current_annotator]
                mask_per_annotator.append(filtered_mask)

            # Stack the tensors into a new tensor of size [4, 3, 3, 128, 128]
            patch_per_annotator = patch_per_annotator
            mask_per_annotator = mask_per_annotator

            for p_idx in range(4):
                if len(patch_per_annotator[p_idx]) == 0:
                    continue

                for opt_idx in range(4):
                    optimizers[opt_idx].zero_grad()

                loss, _ = net.train_step(args, patch_per_annotator[p_idx], mask_per_annotator[p_idx], p_idx, loss_fct, stage=args.stage)

                if torch.isnan(loss):
                    logger.write_and_print('***** Warning: loss is NaN *****')
                    loss = torch.tensor(10000).cuda()

                loss.backward()
                optimizers[p_idx].step()

                writer.add_scalar('Loss', loss.item(), batches_done)

        # log learning_rate
        for opt_idx in range(4):
            current_lr = optimizers[opt_idx].param_groups[0]['lr']
            writer.add_scalar('LR', current_lr, epoch)
            schedulers[opt_idx].step()            

        # # save model
        # if epoch % 20 == 0:
        #     ckpt = {'model': net.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict()
        #             }
        #     torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}.pth'.format(args.model_name, opt.DATASET, epoch)))

        # validate each epoch
        metrics_dict = validate(args, net, val_loader, opt, writer, epoch)
        print_str = ''
        for key in metrics_dict.keys():
            print_str += key + ': {:.4f}  '.format(metrics_dict[key])
            writer.add_scalar('Metrics/'+key, metrics_dict[key], epoch)
        logger.write_and_print(print_str)

        if args.stage == 1:
            metric_instance_ = metrics_dict['Dice_match']# + metrics_dict['Dice_soft']
        else:
            metric_instance_ = metrics_dict['Dice_each_mean']

        if metric_instance_ >= best_metric:
            best_metric = metric_instance_
            logger.write_and_print("Best Dice: {}".format(best_metric))
            ckpt = {'model': net.state_dict()}
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_best.pth'.format(args.model_name, opt.DATASET, epoch)))
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}{}_{}_best.pth'.format(args.model_name, args.stage, opt.DATASET)))

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
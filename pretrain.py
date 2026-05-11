import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import random

from model import StudentModel, TeacherModel, EMAUpdater
from losses import total_loss, DINOLoss
from dataset import LUSVideoDataset


seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def current_time():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    if local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, 'train_log.txt'), 'w')
    else:
        log_file = None

    dataset = LUSVideoDataset(args.data_path, args.clip_len, args.image_size, mask_ratio=args.mask_ratio)
    print(len(dataset))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=16, pin_memory=True)

    student = StudentModel(drop_stripe_prob=args.drop_stripe_prob).to(device)
    teacher = TeacherModel(student, pre_train=True).to(device)
    ema_updater = EMAUpdater(momentum=args.ema_momentum)

    student = DDP(student, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student.train()
        total_epoch_loss = 0.0

        for step, (x_student, x_teacher, mask_idx) in enumerate(dataloader):
            x_student = x_student.to(device)
            x_teacher = x_teacher.to(device)
            mask_idx = mask_idx.to(device)

            vc_feat, p_p_c, p_f_c, z_p_c, z_f_c = student(x_student, mask_idx)
            with torch.no_grad():
                teacher_proj = teacher(x_teacher, mask_idx)

            loss, loss_dino_p, loss_dino_f, loss_temp, loss_smooth = total_loss(p_p_c, p_f_c, teacher_proj, z_p_c, z_f_c, vc_feat, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_updater.update(student.module, teacher)

            if step % args.log_interval == 0 and local_rank == 0:
                log_msg = (f"{current_time()} Epoch [{epoch}], Step [{step}], Loss: {loss.item():.4f}, "
                           f"loss_dino_p: {loss_dino_p.item():.4f}, loss_dino_f: {loss_dino_f.item():.4f}, loss_temp: {loss_temp.item():.4f}, "
                           f"smooth: {loss_smooth.item():.4f}")
                print(log_msg)
                if log_file:
                    log_file.write(log_msg + '\n')
                    log_file.flush()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(dataloader)

        if local_rank == 0:
            log_str = f"{current_time()} Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}"
            print(log_str)
            if log_file:
                log_file.write(log_str + '\n')
                log_file.flush()

            save_model(student.module, os.path.join(log_dir, f'student_epoch{epoch+1}.pth'))
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(student.module, os.path.join(log_dir, 'best_student.pth'))

    if log_file:
        log_file.close()
    dist.destroy_process_group()

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--clip_len', type=int, default=24)
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--log_dir', type=str, default='pretrain_logs')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--ema_momentum', type=float, default=0.996)
    parser.add_argument('--drop_stripe_prob', type=float, default=0.2)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()

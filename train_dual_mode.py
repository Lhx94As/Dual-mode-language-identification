import os
import math
import random
import argparse
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from data_load import *
import scoring
import subprocess
from Loss import Loss_kd


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, help='dim of input features', default=80)
    parser.add_argument('--model', type=str, help='model name for saving', default='XSA_E2E')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size', default=64)
    parser.add_argument('--optim', type=str, help='optimizer', default='warmcosine')
    parser.add_argument('--warmup', type=int, help='num of steps', default=24000)
    parser.add_argument('--epochs', type=int, help='num of epochs', default=20)
    parser.add_argument('--lang', type=int, help='num of language classes', default=14)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--device', type=int, help='Device name', default=0)
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--temperature', type=int, help='temperature', default=2)
    parser.add_argument('--window', type=str, help='fix or random window in student model', default='fix')
    parser.add_argument('--winlen', type=int, help='window length', default=15)
    parser.add_argument('--alpha', type=float, help='importance of teacher (full mode)', default=0.33)
    parser.add_argument('--beta', type=float, help='importance of student (short mode)', default=0.33)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = X_Transformer_E2E_LID(input_dim=args.dim,
                                  feat_dim=64,
                                  d_k=64,
                                  d_v=64,
                                  d_ff=2048,
                                  n_heads=8,
                                  dropout=0.1,
                                  n_lang=args.lang,
                                  max_seq_len=10000)

    model.to(device)
    train_txt = args.train
    train_set = RawFeatures(train_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=args.batch,
                            pin_memory=False,
                            num_workers=16,
                            shuffle=True,
                            collate_fn=collate_fn_atten)
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    KD_loss_func = nn.KLDivLoss(reduction='batchmean').to(device)
    total_step = len(train_data)
    optimizer = args.optim
    if optimizer == 'noam':
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.1)
        warm_up_with_noam = lambda step: (512 ** (-0.5)) * min((step + 1) ** (-0.5), (step + 1) * args.warmup ** (-1.5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_noam)
    elif optimizer == 'warmcosine':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        warm_up_with_cosine_lr = lambda step: step / args.warmup \
            if step <= args.warmup \
            else 0.5 * (math.cos((step - args.warmup) / (args.epochs * total_step - args.warmup) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif optimizer == 'cosine':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * total_step)
        
    # ====Train=====
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            atten_mask_student = get_atten_mask_student(seq_len, utt_.size(0),
                                                        mask_type=args.window, win_len=args.winlen)
            atten_mask_student = atten_mask_student.to(device=device)

            labels = labels.to(device=device, dtype=torch.long)
            # Forward pass
            outputs = model(utt_, seq_len, atten_mask)
            outputs_student = model(utt_, seq_len, atten_mask_student)

            loss_teacher = loss_func_CRE(outputs, labels)
            loss_student = loss_func_CRE(outputs_student, labels)
            loss_kd = KD_loss_func(F.log_softmax(outputs_student / args.temperature, dim=1),
                                   F.softmax(outputs / args.temperature, dim=1)) * (args.temperature * args.temperature)

            loss = args.alpha * loss_teacher + args.beta * loss_student + (1 - args.alpha - args.beta) * loss_kd
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} loss_T: {:.4f} loss_S: {:.4f}, loss_KD: {:.4f}".
                      format(epoch + 1, args.epochs, step + 1, total_step, loss.item(), loss_teacher.item(),
                             loss_student.item())), loss_kd.item()))
            scheduler.step()
        print(get_lr(optimizer))

        if epoch >= args.epochs - 5:
            torch.save(model.state_dict(), '/home/hexin/Desktop/models/' + '{}_epoch_{}.ckpt'.format(args.model, epoch))

if __name__ == "__main__":
    main()

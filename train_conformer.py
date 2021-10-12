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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
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
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=1600)
    parser.add_argument('--model', type=str, help='model name',
                        default='Transformer')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    # parser.add_argument('--test', type=str, help='testing data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size',
                        default=32)
    parser.add_argument('--warmup', type=int, help='num of epochs',
                        default=12000)
    parser.add_argument('--epochs', type=int, help='num of epochs',
                        default=20)
    parser.add_argument('--lang', type=int, help='num of language classes',
                        default=14)
    parser.add_argument('--lr', type=float, help='initial learning rate',
                        default=0.0001)
    parser.add_argument('--device', type=int, help='Device name',
                        default=0)
    parser.add_argument('--seed', type=int, help='Device name',
                        default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = Conformer(input_dim=1600,
                      feat_dim=64,
                      d_k=64,
                      d_v=64,
                      n_heads=8,
                      d_ff=2048,
                      max_len=300,
                      dropout=0.1,
                      device=device,
                      n_lang=14)
    model.to(device)

    train_txt = args.train
    train_set = RawFeatures(train_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=args.batch,
                            pin_memory=True,
                            num_workers=16,
                            shuffle=True,
                            collate_fn=collate_fn_atten)
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_step = len(train_data)
    warm_up_with_cosine_lr = lambda step: step / args.warmup \
        if step <= args.warmup \
        else 0.5 * (math.cos((step - args.warmup) / (args.epochs * total_step - args.warmup) * math.pi) + 1)
    # Train the model
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            labels = labels.to(device=device, dtype=torch.long)
            # Forward pass
            outputs = model(utt_, atten_mask)
            loss = loss_func_CRE(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".
                      format(epoch + 1, args.epochs, step + 1, total_step, loss.item()))
        scheduler.step()

        if epoch >= args.epochs - 5:
            torch.save(model.state_dict(), '/home/hexin/Desktop/models/' + '{}_epoch_{}.ckpt'.format(args.model, epoch))



if __name__ == "__main__":
    main()

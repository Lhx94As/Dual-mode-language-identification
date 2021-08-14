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
                        default=437)
    parser.add_argument('--model', type=str, help='model name',
                        default='Transformer')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    # parser.add_argument('--test', type=str, help='testing data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size',
                        default=64)
    parser.add_argument('--warmup', type=int, help='num of epochs',
                        default=5)
    parser.add_argument('--epochs', type=int, help='num of epochs',
                        default=20)
    parser.add_argument('--lang', type=int, help='num of language classes',
                        default=3)
    parser.add_argument('--lr', type=float, help='initial learning rate',
                        default=0.0001)
    parser.add_argument('--device', type=int, help='Device name',
                        default=0)
    parser.add_argument('--seed', type=int, help='Device name',
                        default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = Transformer_E2E_LID(n_lang=args.lang,
                                dropout=0.1,
                                input_dim=args.dim,
                                feat_dim=64,
                                n_heads=8,
                                d_k=64,
                                d_v=64,
                                d_ff=2048,
                                max_seq_len=300,
                                device=device)
    model.to(device)

    train_txt = args.train
    train_set = RawFeatures(train_txt)
    valid_txt_3s = "/home/hexin/Desktop/hexin/datasets/lre17_eval/utt2lan_200ms_3s.txt"
    valid_txt_10s = "/home/hexin/Desktop/hexin/datasets/lre17_eval/utt2lan_200ms_10s.txt"
    valid_txt_30s = "/home/hexin/Desktop/hexin/datasets/lre17_eval/utt2lan_200ms_30s.txt"
    valid_set_3s = RawFeatures(valid_txt_3s)
    valid_set_10s = RawFeatures(valid_txt_10s)
    valid_set_30s = RawFeatures(valid_txt_30s)
    train_data = DataLoader(dataset=train_set,
                            batch_size=args.batch,
                            pin_memory=True,
                            num_workers=16,
                            shuffle=True,
                            collate_fn=collate_fn_atten)
    valid_data_3s = DataLoader(dataset=valid_set_3s,
                               batch_size=1,
                               pin_memory=True,
                               shuffle=False,
                               collate_fn=collate_fn_atten)
    valid_data_10s = DataLoader(dataset=valid_set_10s,
                                batch_size=1,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=collate_fn_atten)
    valid_data_30s = DataLoader(dataset=valid_set_30s,
                                batch_size=1,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=collate_fn_atten)
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup \
        if epoch <= args.warmup \
        else 0.5 * (math.cos((epoch - args.warmup) / (args.epochs - args.warmup) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # Train the model
    total_step = len(train_data)
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            # print(seq_len)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            # print(atten_mask.size())
            labels = labels.to(device=device, dtype=torch.long)
            # Forward pass
            outputs = model(utt_, seq_len, atten_mask)
            # outputs = get_output(outputs, seq_len)
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
            model.eval()
            correct = 0
            total = 0
            scores = 0
            with torch.no_grad():
                for step, (utt, labels, seq_len) in enumerate(valid_data_3s):
                    utt = utt.to(device=device, dtype=torch.float)
                    labels = labels.to(device)
                    # Forward pass\
                    outputs = model(utt, seq_len, atten_mask=None)
                    predicted = torch.argmax(outputs, -1)
                    total += labels.size(-1)
                    correct += (predicted == labels).sum().item()
                    if epoch >= args.epochs - 5:
                        if step == 0:
                            scores = outputs
                        else:
                            scores = torch.cat((scores, outputs), dim=0)
            acc = correct / total
            print('Current Acc.: {:.4f} %'.format(100 * acc))
            scores = scores.squeeze().cpu().numpy()
            print(scores.shape)
            trial_txt = os.path.split(args.train)[0] + '/trial_3s.txt'
            score_txt = os.path.split(args.train)[0] + '/score_3s.txt'
            scoring.get_trials(valid_txt_3s, args.lang, trial_txt)
            scoring.get_score(valid_txt_3s, scores, args.lang, score_txt)
            eer_txt = trial_txt.replace('trial', 'eer')
            subprocess.call(f"/home/hexin/Desktop/hexin/kaldi/egs/subtools/computeEER.sh "
                            f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)

            correct = 0
            total = 0
            scores = 0
            with torch.no_grad():
                for step, (utt, labels, seq_len) in enumerate(valid_data_10s):
                    utt = utt.to(device=device, dtype=torch.float)
                    labels = labels.to(device)
                    # Forward pass\
                    outputs = model(utt, seq_len, atten_mask=None)
                    predicted = torch.argmax(outputs, -1)
                    total += labels.size(-1)
                    correct += (predicted == labels).sum().item()
                    if epoch >= args.epochs - 5:
                        if step == 0:
                            scores = outputs
                        else:
                            scores = torch.cat((scores, outputs), dim=0)
            acc = correct / total
            print('Current Acc.: {:.4f} %'.format(100 * acc))
            scores = scores.squeeze().cpu().numpy()
            print(scores.shape)
            trial_txt = os.path.split(args.train)[0] + '/trial_10s.txt'
            score_txt = os.path.split(args.train)[0] + '/score_10s.txt'
            scoring.get_trials(valid_txt_10s, args.lang, trial_txt)
            scoring.get_score(valid_txt_10s, scores, args.lang, score_txt)
            eer_txt = trial_txt.replace('trial', 'eer')
            subprocess.call(f"/home/hexin/Desktop/hexin/kaldi/egs/subtools/computeEER.sh "
                            f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)

            correct = 0
            total = 0
            scores = 0
            with torch.no_grad():
                for step, (utt, labels, seq_len) in enumerate(valid_data_30s):
                    utt = utt.to(device=device, dtype=torch.float)
                    labels = labels.to(device)
                    # Forward pass\
                    outputs = model(utt, seq_len, atten_mask=None)
                    predicted = torch.argmax(outputs, -1)
                    total += labels.size(-1)
                    correct += (predicted == labels).sum().item()
                    if epoch >= args.epochs - 5:
                        if step == 0:
                            scores = outputs
                        else:
                            scores = torch.cat((scores, outputs), dim=0)
            acc = correct / total
            print('Current Acc.: {:.4f} %'.format(100 * acc))
            scores = scores.squeeze().cpu().numpy()
            print(scores.shape)
            trial_txt = os.path.split(args.train)[0] + '/trial_30s.txt'
            score_txt = os.path.split(args.train)[0] + '/score_30s.txt'
            scoring.get_trials(valid_txt_30s, args.lang, trial_txt)
            scoring.get_score(valid_txt_30s, scores, args.lang, score_txt)
            eer_txt = trial_txt.replace('trial', 'eer')
            subprocess.call(f"/home/hexin/Desktop/hexin/kaldi/egs/subtools/computeEER.sh "
                            f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)


if __name__ == "__main__":
    main()

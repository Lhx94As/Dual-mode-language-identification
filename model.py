import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import *
import conformer as cf
from convolution_module import Conv1dSubampling
from pooling_layers import *
#
class Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1,n_lang=3, max_seq_len=300,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Transformer_E2E_LID, self).__init__()
        # self.subsample = Conv1dSubampling()
        self.transform = nn.Linear(input_dim, feat_dim)
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.d_model = feat_dim*n_heads
        self.n_heads = n_heads
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask):
        batch_size = x.size(0)
        output = self.transform(x) #x [B, T, input_dim] => [B, T feat_dim]
        output = self.layernorm1(output)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)

        stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        # print(stats.size())
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class X_Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1, n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(X_Transformer_E2E_LID, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.fc_xv = nn.Linear(1024, feat_dim)

        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        # self.attention_block3 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        # self.attention_block4 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        # self.attention_block5 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        # self.attention_block6 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)


        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def forward(self, x, seq_len, atten_mask, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = self.dropout(x)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.bn3(F.relu(self.tdnn3(x)))

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        # print("pooling", stats.size())
        embedding = self.fc_xv(stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        # output, _ = self.attention_block3(output, atten_mask)
        # output, _ = self.attention_block4(output, atten_mask)
        stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class Conformer(nn.Module):
    def __init__(self,input_dim, feat_dim, d_k, d_v, n_heads, d_ff, max_len, dropout, device, n_lang):
        super(Conformer, self).__init__()
        self.conv_subsample = Conv1dSubampling(in_channels=input_dim, out_channels= input_dim)
        self.transform = nn.Linear(input_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = feat_dim * n_heads
        self.layernorm1 = LayerNorm(feat_dim)
        self.n_heads = n_heads
        self.attention_block1 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block2 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block3 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block4 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def forward(self, x, atten_mask):
        batch_size = x.size(0)
        output = self.transform(x)  # x [B, T, input_dim] => [B, T feat_dim]
        output = self.layernorm1(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        # print(stats.size())
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


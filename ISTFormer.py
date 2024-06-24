import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

from ISTF_attention import SelfAttentionLayer

class ISTFormer(nn.Module):
    def __init__(
        self, device, input_dim, num_nodes, dropout=0.1,embeding_dim=152
    ):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes#170
        self.output_len = 12
        self.embeding_dim = embeding_dim


        self.tree = ST_Tree(
            device=device,
            num_nodes=self.num_nodes,
            dropout=dropout,embeding_dim=embeding_dim
        )

        self.input_proj = nn.Linear(input_dim, 24)#3->24
        self.tod_embedding = nn.Embedding(288, 24)
        self.dow_embedding = nn.Embedding(7, 24)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.output_len, num_nodes, 80))
        )

        self.glu = GLU(self.embeding_dim, dropout=0.1)  # 96*2
        self.regression = nn.Conv2d(
            self.embeding_dim, self.output_len, kernel_size=(1, self.output_len))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):

        x = input
        # Data Embedding
        tod, dow = x[:, 1, :, :], x[:, 2, :, :]
        tod = tod.permute(0,2,1)
        dow = dow.permute(0,2,1)
        x = x.permute(0,3,2,1)#[64,12,170,3]
        x = self.input_proj(x)#[64,12,170,24] 3—>24
        features = [x]

        #day embedding
        tod_emb = self.tod_embedding((tod * 288).long())
        features.append(tod_emb)
        #week embedding
        dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)[16,12,170,24]
        features.append(dow_emb)


        batch_size = x.shape[0]
        adp_emb = self.adaptive_embedding.expand(
            size=(batch_size, *self.adaptive_embedding.shape)
        )
        features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)[64,12,170,152]

        x = x.permute(0, 3, 2, 1)

        x = self.tree(x)

        #residual and mlp
        res = self.glu(x)+x
        prediction = self.regression(F.relu(res))

        return prediction
    pass

class ST_Tree(nn.Module):
    def __init__(
        self, device, num_nodes=170, dropout=0.1,embeding_dim=152
    ):
        super().__init__()
        # print(channels, diffusion_step, num_nodes, dropout)#S08:192 1 170 0.1
        # sys.exit()

        self.embedding_dim = embeding_dim

        self.memory1 = nn.Parameter(torch.randn(self.embedding_dim, num_nodes, 6))#(192,170,6)
        self.memory2 = nn.Parameter(torch.randn(self.embedding_dim, num_nodes, 3))#(192,170,3)
        self.memory3 = nn.Parameter(torch.randn(self.embedding_dim, num_nodes, 3))#(192,170,3)


        self.ISTL1 = ISTL(
            device=device,
            splitting=True,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory1,embeding_dim=embeding_dim
        )
        self.ISTL2 = ISTL(
            device=device,
            splitting=True,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory2,embeding_dim=embeding_dim
        )
        self.ISTL3 = ISTL(
            device=device,
            splitting=True,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory2,embeding_dim=embeding_dim
        )

    def concat(self, even, odd):

        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]#concat1 or concat2 : len = 3 ,concat0 : len = 6
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x):

        x_even_update1, x_odd_update1 = self.ISTL1(x)
        x_even_update2, x_odd_update2 = self.ISTL2(x_even_update1)
        x_even_update3, x_odd_update3 = self.ISTL3(x_odd_update1)

        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)

        output = concat0 + x
        return output
    pass

class ISTL(nn.Module):
    def __init__(
        self,
        device,
        splitting=True,
        num_nodes=170,
        dropout=0.2, emb = None,embeding_dim=152
    ):
        super(ISTL, self).__init__()

        device = device
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()
        self.embeding_dim = embeding_dim
        #
        # Conv1 = []
        # Conv2 = []
        # Conv3 = []
        # Conv4 = []
        # pad_l = 3
        # pad_r = 3
        #
        # k1 = 5
        # k2 = 3
        # Conv1 += [
        #     nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k1)),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Dropout(self.dropout),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k2)),
        #     nn.Tanh(),
        # ]
        # Conv2 += [
        #     nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k1)),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Dropout(self.dropout),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k2)),
        #     nn.Tanh(),
        # ]
        # Conv4 += [
        #     nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k1)),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Dropout(self.dropout),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k2)),
        #     nn.Tanh(),
        # ]
        # Conv3 += [
        #     nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k1)),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Dropout(self.dropout),
        #     nn.Conv2d(self.embeding_dim, self.embeding_dim, kernel_size=(1, k2)),
        #     nn.Tanh(),
        # ]
        #
        # self.conv1 = nn.Sequential(*Conv1)
        # self.conv2 = nn.Sequential(*Conv2)
        # self.conv3 = nn.Sequential(*Conv3)
        # self.conv4 = nn.Sequential(*Conv4)

        self.STA = STA(num_nodes, dropout, emb,self.embeding_dim)

    def forward(self, x):

        #split
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        # x1 = self.conv1(x_even)
        x1 = x_even
        x1 = self.STA(x1)
        d = x_odd.mul(torch.tanh_(x1))

        #x2 = self.conv2(x_odd)
        x2 = x_odd
        x2 = self.STA(x2)
        c = x_even.mul(torch.tanh_(x2))

        #x3 = self.conv3(c)
        x3 = c
        x3 = self.STA(x3)
        x_odd_update = d+x3

        #x4 = self.conv4(d)
        x4 = d
        x4 = self.STA(x4)
        x_even_update =c + x4

        return (x_even_update, x_odd_update)

class STA(nn.Module):
    def __init__(self, num_nodes=170, dropout=0.1, emb=None,embeding_dim=152):
        super().__init__()
        #PEMS08:channels 192 num_node 170

        self.emb = emb#torch.Size([192, 170, 6])
        num_layers=3
        self.model_dim=embeding_dim
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1)  # [152，256,4，0.1]
                for _ in range(num_layers)  # 3
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(num_layers)  # 3
            ]
        )

    def forward(self, x):

        skip = x
        x = x.permute(0,3,2,1)#[64,6,170,152]

        for attn in self.attn_layers_t:  # 3
            x = attn(x, dim=1)

        for attn in self.attn_layers_s:
            x = attn(x,dim=2)
            pass
        x = x.permute(0,3,2,1)

        return x




class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))
    pass

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Parameter


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]  # 输入的维度，隐层的维度、输出的维度
        for i in range(len(channels) - 1):  # 3层GCN (316->32), (32->64), (64->128)
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):  # todo 前两层就是上一层的输出经过leaky_relu后是下一层的输入
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)  # 中间加一层dropout
        x = self.gcn[-1](x, adj)

        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 参数初始化时进行均匀分布
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # todo 正如论文中提到的，第一个H就是X
        support = torch.mm(input, self.weight)  # (3980, 316)->(3980, 32)->(3980, 64)->(3980, 128)先将所有结点的feature乘上一个w
        output = torch.spmm(adj,
                            support)  # (4980, 32)->(4980, 64)->(3980, 128) 稀疏矩阵乘法，sparse在前，dense在后，不过此处用mm得到的结果也是一样的
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'  # 打印时输出的是GraphConvolution(输入维度->输出维度)


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask  # False
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))  # todo 比较奇怪的是作者在论文中用了两个w，但此处共用一个
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))  # (256, 1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):  # X是Trajectory Flow Map中的结点属性，A来表示边属性
        Wh = torch.mm(X, self.W)  # (4980, 128)每个POI结点属性乘上w

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # 这里+1应该是应用了广播机制，相当于加了个全1矩阵。将从A转化的拉普拉斯矩阵的值范围从0-1 到 1-2
        e = e * A  # 式子6中最终乘出的Phi

        return e  # 形状和邻接矩阵一致，代表从当前行代表的POI访问各列的POI的概率

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (4980, 1)拿同个W乘上a的前段和后段
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (4980, 1)
        e = Wh1 + Wh2.T  # (4980, 4980) 相当于Wh1的每一个值都与Wh2的每一个值加一遍形成一列
        return self.leakyrelu(e)


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )  # dictionary大小为用户的数量，每个向量维度为128

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)  # 就是查询到对应用户的向量
        return embed


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))  # (1, 1)todo w0和b0应该是i=0的情况
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))  # (1, 31)todo 这里-1是因为t2v里concat了
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):  # tau就是标准化后的时间，为该时刻在24小时中的占比
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)  # v1是个(1, 31)的向量，输出值还得sin一下
    v2 = torch.matmul(tau, w0) + b0  # todo 意味着每个tau都会和w0进行计算？
    return torch.cat([v1, v2], 1)


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )  # dictionary大小为所有类别的数量，每个向量维度为32

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed  # 返回该类别对应的向量


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)  # 维度进出都是同个值
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))  # todo 两个fuse用的都是这个类，形参不应该这样命名
        x = self.leaky_relu(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)  # todo 初始化时创建位置编码
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid,
                                                 dropout)  # 总的embedding大小传入transformer的encoder层
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      nlayers)  # encoder_layers相当于forward函数，将forward函数放入模型中
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)  # todo decoder好像就是经transformer得到的h传入不同输出形状的linear层
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 传入的sz是batch_size，triu返回一个上三角，转置后变成下三角
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # 将下三角的0变-inf，1变0
        return mask  # 对角线及其以下是0，对角线以上为-inf

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()  # todo 为啥只init了poi的权重,decoder此处应该有time和poi和cat
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):  # src是一个batch_size的拼接embedding
        src = src * math.sqrt(self.embed_size)  # todo 为啥要乘上embed_size?将里面的每一项都放大了
        src = self.pos_encoder(src)  # 加上位置编码
        x = self.transformer_encoder(src, src_mask)  # (20, x, 320)，这个320维对应几个线性层的输入维度，Transformer的Encoder直接调包
        out_poi = self.decoder_poi(x)  # 4980个POI对应的概率
        out_time = self.decoder_time(x)  # 预测出一个时刻，以小数的形式展示
        out_cat = self.decoder_cat(x)  # 313个类别对应的概率
        return out_poi, out_time, out_cat


class PositionalEncoding(nn.Module):
    """
    Transformer不像RNN这些能直接获得序列的相对位置，因此要人为加入位置编码
    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # todo (500, 320)，max_len可能是序列的最大长度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (500, 1)，从1到500的序号
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 经过幂指数变化的公式
        pe[:, 0::2] = torch.sin(position * div_term)  # 每行的偶数索引列是sin，奇数索引列是cos
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # todo (500, 1, 320)，注意加上这个维度1之后就刚好可以用来广播了
        self.register_buffer('pe', pe)  # todo 位置编码模型训练时不更新

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # (20, x, 320) + (20, 1, 320) todo 应该是位置编码在初始化时做大一点，加的时候就可以对应加多少份
        return self.dropout(x)

import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="FishModel.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')  # 随机种子
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')  # cuda
    # Data
    parser.add_argument('--dataset-adj-mtx',
                        type=str,
                        default='dataset/NYC/graph_A.csv',
                        help='Graph adjacent path')  # 邻接矩阵
    parser.add_argument('--dataset-node-feats',
                        type=str,
                        default='dataset/NYC/graph_X.csv',
                        help='Graph node features path')  # 邻接矩阵中每个结点的属性值
    parser.add_argument('--dataset-train',
                        type=str,
                        default='dataset/NYC/NYC_train.csv',
                        help='Training dataset path')  # 完整的原始训练集
    parser.add_argument('--dataset-val',
                        type=str,
                        default='dataset/NYC/NYC_val.csv',
                        help='Validation dataset path')  # 验证集
    parser.add_argument('--short-traj-threshold',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')  # 轨迹长度的最小阈值
    parser.add_argument('--time-units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')  # todo 啥用？
    parser.add_argument('--time-feature',
                        type=str,
                        default='norm_in_day_time',
                        help='The name of time feature in the dataset')  # TrajectoryDatasetTrain中用到了这个参数

    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')  # 每层gcn的隐层大小不同
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='Scale factor for the time loss term')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=True,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')  # train时的工作路径
    parser.add_argument('--name',
                        default='exp-0',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')

    return parser.parse_args()

import glob
import os
from pathlib import Path
import re
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def increment_path(path, exist_ok=True, sep=''):
    """ 每次代码执行都生成一个对应的工作目录, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc. """
    path = Path(path)  # os-agnostic，目录是runs/train/exp{sep}-n
    if (path.exists() and exist_ok) or (not path.exists()):  # 若不存在直接命名为exp的目录，就直接返回该路径
        return f"{path}"
    else:
        dirs = glob.glob(f"{path}{sep}*")  # 找到该目录下类似的文件，名称为exp-n
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]  # 取出这些类似文件的名称，即exp-n
        i = [int(m.groups()[0]) for m in matches if m]  # 取出exp-n中的n
        n = max(i) + 1 if i else 2  # 若只存在exp文件夹，那新增的文件夹命名为exp-2，否则就依次递增1
        return f"{path}{sep}{n}"  # update path


def load_graph_adj_mtx(path):
    """邻接矩阵"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid',
                             feature3='latitude', feature4='longitude'):
    """和邻接矩阵相对应的结点属性表"""
    df = pd.read_csv(path)  # 该csv是所有POI结点的详细属性表
    rlt_df = df[[feature1, feature2, feature3, feature4]]  # 取出对应列
    X = rlt_df.to_numpy()

    return X


def category2_one_hot(raw_X, num_pois):
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])  # 取出每个POI对应的类别编号，由字母和数字组成
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))  # map就是将每个编号加上一层数组，但还得list转换一下，fit过后每个类别对应一种one-hot编码
    one_hot_rlt = one_hot_encoder.transform(
        list(map(lambda x: [x], cat_list))).toarray()  # (4980, 313) 这4980个POI共有313个类别
    num_cats = one_hot_rlt.shape[-1]  # 总共4980个POI分为313类
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)  # 根据raw_X新建一个X，该X类别编号被转化为了one-hot编码
    X[:, 0] = raw_X[:, 0]  # 复制访问数量
    X[:, 1:num_cats + 1] = one_hot_rlt  # 替换类别为one_hot编码
    X[:, num_cats + 1:] = raw_X[:, 2:]  # 复制经纬度
    return one_hot_encoder, X, num_cats


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]  # 结点数量，即POI数量
    # row sum
    # np.diag虽然形状是个对角矩阵，但类型还是ndarray，调用np.asmatrix后类型就转为了matrix类型
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))  # 将每个POI随后访问的POI数量做成一个对角矩阵，todo 出度矩阵
    # column sum
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)  # 将原始邻接矩阵转为matrix类型
    id_mat = np.asmatrix(np.identity(n_vertex))  # matrix类型的单位矩阵

    if mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv  对应论文4.2.1中公式
        wid_deg_mat = deg_mat + id_mat  # 第一个括号内，度矩阵加上单位矩阵
        wid_adj_mat = adj_mat + id_mat  # 第二个括号内，邻接矩阵加上单位矩阵
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)  # normalized 拉普拉斯矩阵
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def items_re_index(train_df, nodes_df):
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))  # 去重后的POI编号，4980个，数量与邻接矩阵的宽或高一致
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
    cat_ids = list(set(nodes_df['poi_catid'].tolist()))  # 去重后的POI类别编号，313个
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))  # POI类别编号为键，序号为值，将POI类别reindex
    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}  # 遍历每个结点的属性，建立属性中POI和类别的reindex后的映射字典
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row['poi_catid']]
    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]  # todo 去重后的转为字符类型的用户编号，共1047个，这些编号是不连续的
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))  # 用户编号为键，序号为值，进行reindex
    return poi_ids, cat_ids, user_ids, poi_id2idx_dict, cat_id2idx_dict, user_id2idx_dict, poi_idx2cat_idx_dict


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2  # 去掉padding项后的均方损失
    loss = out.mean()
    return loss  # 返回损失的平均


def input_traj_to_embeddings(in_args, sample, poi_embeddings, user_embed_model, time_embed_model, cat_embed_model,
                             embed_fuse_model1, embed_fuse_model2, poi_idx2cat_idx_dict, user_id2idx_dict):
    # Parse sample  # todo 取出轨迹编号，输入数据对应的POI编号、访问时刻、POI类别
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]
    input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

    # User to embedding
    user_id = traj_id.split('_')[0]  # 轨迹编号中含有用户编号
    user_idx = user_id2idx_dict[user_id]  # 用户编号对应的序号
    input = torch.LongTensor([user_idx]).to(device=in_args.device)  # todo 将值转为tensor，用此取出该序号对应的用户embedding
    user_embedding = user_embed_model(input)
    user_embedding = torch.squeeze(user_embedding)  # 返回的用户向量是一个(1, 128)，去掉第0维

    # POI to embedding and fuse embeddings
    input_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]]  # 取出该POI序号对应的embedding
        # todo 取出的embedding本身就是(128, )，所以这个squeeze应没啥用
        poi_embedding = torch.squeeze(poi_embedding).to(device=in_args.device)

        # Time to vector
        time_embedding = time_embed_model(
            torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=in_args.device))
        time_embedding = torch.squeeze(time_embedding).to(device=in_args.device)  # (1, 32) -> 32 去掉第0维

        # Categroy to embedding
        cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=in_args.device)
        cat_embedding = cat_embed_model(cat_idx)
        cat_embedding = torch.squeeze(cat_embedding)  # (1, 32) 去掉第0维

        # Fuse user+poi embeds
        fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)  # (256, )仅是叠起来后经过一个线性层，然后leaky relu一下
        fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)  # (64, )

        # Concat time, cat after user+poi
        concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)  # (320, )

        # Save final embed
        input_seq_embed.append(concat_embedding)

    return input_seq_embed  # 得到一个poi和用户和poi类别和访问时刻全加起来的embedding


# todo Transformer输出的POI预测值会放入这个函数
def adjust_pred_prob_by_graph(y_pred_poi, node_attn_model, X, A, batch_seq_lens, batch_input_seqs):  # (20, x, 4980)
    y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
    attn_map = node_attn_model(X, A)  # todo 模型图中的Probability Map，这个值是未经标准化的从某个POI到某个POI的可能性
    # todo 这个循环就是说从先验的角度上给出从某个POI到其他所有POI的可能性，本身预测值就是从输入的POI到其他4980个POI的概率，对这个概率进行再微调
    for i in range(len(batch_seq_lens)):
        traj_i_input = batch_input_seqs[i]  # 20个样本中每个样本对应的轨迹，即POI编号
        for j in range(len(traj_i_input)):  # 输入轨迹中的每个POI
            # 从概率图中得到的某个POI到其他4980个POI的概率加上预测出的从4980个POI到其他4980个POI的概率
            y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

    return y_pred_poi_adjusted


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]  # 取出序列中最后一项标签及其预测值，为什么要取最后一项不是很清楚
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]  # 由于argsort的返回值是升序的，因此是先取倒数k项再逆序排列，由此得到从大到小的预测概率值
    idx = np.where(top_k_rec == y_true)[0]  # 在topk预测值中是否存在标签值
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]  # 在此步之前和acc也没区别
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)  # 返回预测值的排序的倒数


def save_all_results(in_args, train_epochs_loss_list, train_epochs_poi_loss_list, train_epochs_time_loss_list,
                     train_epochs_cat_loss_list, train_epochs_top1_acc_list, train_epochs_top5_acc_list,
                     train_epochs_top10_acc_list,
                     train_epochs_top20_acc_list, train_epochs_mAP20_list, train_epochs_mrr_list, val_epochs_loss_list,
                     val_epochs_poi_loss_list,
                     val_epochs_time_loss_list, val_epochs_cat_loss_list, val_epochs_top1_acc_list,
                     val_epochs_top5_acc_list, val_epochs_top10_acc_list,
                     val_epochs_top20_acc_list, val_epochs_mAP20_list, val_epochs_mrr_list):
    with open(os.path.join(in_args.save_dir, 'metrics-train.txt'), "w") as f:  # todo 训练集结果
        print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
        print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
        print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
              file=f)
        print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
        print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
        print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
        print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
              file=f)
        print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
              file=f)
        print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
        print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
    with open(os.path.join(in_args.save_dir, 'metrics-val.txt'), "w") as f:  # 验证集结果
        print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
        print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
        print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
        print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
        print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
        print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
        print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
        print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
        print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
        print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)

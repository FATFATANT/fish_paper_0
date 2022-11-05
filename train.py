import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel
from my_dataset import TrajectoryDatasetTrain, TrajectoryDatasetVal
from my_parser import parameter_parser
from my_utils import increment_path, load_graph_adj_mtx, load_graph_node_features, category2_one_hot, \
    calculate_laplacian_matrix, items_re_index, maksed_mse_loss, input_traj_to_embeddings, adjust_pred_prob_by_graph, \
    top_k_acc_last_timestep, mAP_metric_last_timestep, MRR_metric_last_timestep, save_all_results

max_val_score = -np.inf
# For plotting
# train
train_epochs_top1_acc_list = []
train_epochs_top5_acc_list = []
train_epochs_top10_acc_list = []
train_epochs_top20_acc_list = []
# 最大后验概率估计（Maximum a posteriori estimation, 简称MAP）
train_epochs_mAP20_list = []
train_epochs_mrr_list = []
train_epochs_loss_list = []
train_epochs_poi_loss_list = []
train_epochs_time_loss_list = []
train_epochs_cat_loss_list = []
# val
val_epochs_top1_acc_list = []
val_epochs_top5_acc_list = []
val_epochs_top10_acc_list = []
val_epochs_top20_acc_list = []
val_epochs_mAP20_list = []
val_epochs_mrr_list = []
val_epochs_loss_list = []
val_epochs_poi_loss_list = []
val_epochs_time_loss_list = []
val_epochs_cat_loss_list = []


def train(in_args):
    # 对于Path而言很神奇的一点在于可以直接在后面跟着斜杠加字符串拼接路径
    in_args.save_dir = increment_path(Path(in_args.project) / in_args.name, exist_ok=in_args.exist_ok, sep='-')
    if not os.path.exists(in_args.save_dir):
        os.makedirs(in_args.save_dir)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(in_args.save_dir, f"log_training.txt"),
                        filemode='w')  # 日志的基本配置
    console = logging.StreamHandler()  # 日志输出到流
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)  # 日志输出到console
    logging.info(in_args)
    """  ====================== Load dataset ======================  """
    train_df = pd.read_csv(in_args.dataset_train)  # 训练集数据，83230条记录
    val_df = pd.read_csv(in_args.dataset_val)  # 验证集数据，10340条记录
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(in_args.dataset_adj_mtx)  # 过渡图的邻接矩阵
    raw_X = load_graph_node_features(in_args.dataset_node_feats)  # 过渡图中POI结点的特征
    logging.info(f"raw_X.shape: {raw_X.shape};")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]  # poi总数
    logging.info('One-hot encoding poi categories id')  # todo 将category转为one-hot的意义是什么
    one_hot_encoder, X, num_cats = category2_one_hot(raw_X, num_pois)
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(in_args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:  # 将转换后的one-hot的POI类别编码也保存一下
        pickle.dump(one_hot_encoder, f)
    # Normalization
    print('Laplacian matrix...')
    laplacian_matrix = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')  # 使用邻接矩阵形成的标准化拉普拉斯矩阵
    nodes_df = pd.read_csv(in_args.dataset_node_feats)
    poi_ids, cat_ids, user_ids, poi_id2idx_dict, cat_id2idx_dict, user_id2idx_dict, poi_idx2cat_idx_dict = items_re_index(
        train_df, nodes_df)
    print('Prepare dataloader...')
    """ ====================== Define dataloader ======================  """
    train_dataset = TrajectoryDatasetTrain(in_args, train_df, poi_id2idx_dict)
    val_dataset = TrajectoryDatasetVal(in_args, val_df, user_id2idx_dict, poi_id2idx_dict)
    train_loader = DataLoader(train_dataset,
                              batch_size=in_args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=in_args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=in_args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=in_args.workers,
                            collate_fn=lambda x: x)
    """ ====================== Build Models ======================  """
    # %% Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)  # 转换为tensor
        laplacian_matrix = torch.from_numpy(laplacian_matrix)
    X = X.to(device=in_args.device, dtype=torch.float)  # 放到GPU上
    laplacian_matrix = laplacian_matrix.to(device=in_args.device, dtype=torch.float)
    in_args.gcn_nfeat = X.shape[1]  # 316 其中类别转为one-hot
    poi_embed_model = GCN(ninput=in_args.gcn_nfeat,
                          nhid=in_args.gcn_nhid,
                          noutput=in_args.poi_embed_dim,
                          dropout=in_args.gcn_dropout)
    # Node Attn Model
    # 应该是对应Transition Attention Module
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=in_args.node_attn_nhid, use_mask=False)
    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)  # 总用户数
    user_embed_model = UserEmbeddings(num_users, in_args.user_embed_dim)
    # %% Model3: Time Model
    time_embed_model = Time2Vec('sin', out_dim=in_args.time_embed_dim)
    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, in_args.cat_embed_dim)
    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(in_args.user_embed_dim, in_args.poi_embed_dim)  # 对应4.3.1 用户和POI embedding的融合
    embed_fuse_model2 = FuseEmbeddings(in_args.time_embed_dim, in_args.cat_embed_dim)  # 对应4.3.2 时间和POI类别的融合
    # %% Model6: Sequence model
    # 四个embedding维度大小求和，这所有的POI到时就会作为一个整体放入Transformer中
    in_args.seq_input_embed = in_args.poi_embed_dim + in_args.user_embed_dim + in_args.time_embed_dim + in_args.cat_embed_dim
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 in_args.seq_input_embed,
                                 in_args.transformer_nhead,
                                 in_args.transformer_nhid,
                                 in_args.transformer_nlayers,
                                 dropout=in_args.transformer_dropout)
    # Define overall loss and optimizer 将所有模型的参数传入优化器
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(seq_model.parameters()),
                           lr=in_args.lr,
                           weight_decay=in_args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略标签为-1的项，因为-1是padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss  # todo 注意时间用的loss不是直接调包，而是手写了一个
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=in_args.lr_scheduler_factor)  # 当metric变化不多时动态调整学习率
    """  ====================== Train ======================  """
    poi_embed_model = poi_embed_model.to(device=in_args.device)  # todo 将所有模型放入cuda
    node_attn_model = node_attn_model.to(device=in_args.device)
    user_embed_model = user_embed_model.to(device=in_args.device)
    time_embed_model = time_embed_model.to(device=in_args.device)
    cat_embed_model = cat_embed_model.to(device=in_args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=in_args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=in_args.device)
    seq_model = seq_model.to(device=in_args.device)
    # %% Loop epoch

    for epoch in range(in_args.epochs):
        # For saving ckpt，ckpt应该是check point的缩写

        train_or_val_one_epoch(in_args, epoch, 'train', poi_embed_model, node_attn_model, user_embed_model,
                               time_embed_model,
                               cat_embed_model,
                               embed_fuse_model1, embed_fuse_model2, seq_model, train_loader, val_loader, X,
                               laplacian_matrix, poi_id2idx_dict, user_id2idx_dict, cat_id2idx_dict,
                               poi_idx2cat_idx_dict, criterion_poi, criterion_cat, criterion_time, optimizer,
                               lr_scheduler)
        train_or_val_one_epoch(in_args, epoch, 'val', poi_embed_model, node_attn_model, user_embed_model,
                               time_embed_model,
                               cat_embed_model,
                               embed_fuse_model1, embed_fuse_model2, seq_model, train_loader, val_loader, X,
                               laplacian_matrix, poi_id2idx_dict, user_id2idx_dict, cat_id2idx_dict,
                               poi_idx2cat_idx_dict, criterion_poi, criterion_cat, criterion_time, optimizer,
                               lr_scheduler)
        log_results_end_epoch(in_args, logging, epoch)
    save_all_results(in_args, train_epochs_loss_list, train_epochs_poi_loss_list, train_epochs_time_loss_list,
                     train_epochs_cat_loss_list, train_epochs_top1_acc_list, train_epochs_top5_acc_list,
                     train_epochs_top10_acc_list,
                     train_epochs_top20_acc_list, train_epochs_mAP20_list, train_epochs_mrr_list, val_epochs_loss_list,
                     val_epochs_poi_loss_list,
                     val_epochs_time_loss_list, val_epochs_cat_loss_list, val_epochs_top1_acc_list,
                     val_epochs_top5_acc_list, val_epochs_top10_acc_list,
                     val_epochs_top20_acc_list, val_epochs_mAP20_list, val_epochs_mrr_list)


def train_or_val_one_epoch(in_args, epoch_num, mode, poi_embed_model, node_attn_model, user_embed_model,
                           time_embed_model,
                           cat_embed_model,
                           embed_fuse_model1, embed_fuse_model2, seq_model, train_loader, val_loader, X,
                           laplacian_matrix, poi_id2idx_dict, user_id2idx_dict, cat_id2idx_dict,
                           poi_idx2cat_idx_dict, criterion_poi, criterion_cat, criterion_time, optimizer, lr_scheduler,
                           ):
    logging.info(f"{'*' * 50}Epoch:{epoch_num:03d}{'*' * 50}\n")
    global max_val_score
    if mode == 'train':
        poi_embed_model.train()  # 将所有模型都设置训练模式
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()  # 这个是transformer模型
        dataloader = train_loader
    else:
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        dataloader = val_loader
    batches_top1_acc_list = []
    batches_top5_acc_list = []
    batches_top10_acc_list = []
    batches_top20_acc_list = []
    batches_mAP20_list = []
    batches_mrr_list = []
    batches_loss_list = []
    batches_poi_loss_list = []
    batches_time_loss_list = []
    batches_cat_loss_list = []
    # (20, 20)的矩阵，对角线及其往下是0，对角线以上为1
    src_mask = seq_model.generate_square_subsequent_mask(in_args.batch).to(in_args.device)
    # Loop batch
    for b_idx, batch in enumerate(dataloader):  # 一个batch没有concat起来，而是将各项放在一个数组中
        if len(batch) != in_args.batch:
            # 数据不足20个就重新生成对应大小的mask
            src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(in_args.device)

        # For padding
        batch_input_seqs = []
        batch_seq_lens = []
        batch_seq_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_time = []
        batch_seq_labels_cat = []
        # todo 经过若干层GCN后，这样的POI可以体现出generic movement pattern，正如论文里写的，是通过GCN来生成对应的POI_embedding
        poi_embeddings = poi_embed_model(X,
                                         laplacian_matrix)  # (4980, 123)todo 注意X是将raw_X的类别转为one-hot的X，A是将raw_A标准化后的拉普拉斯矩阵

        # Convert input seq to embeddings
        for sample in batch:
            # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
            traj_id = sample[0]  # 轨迹序号
            input_seq = [each[0] for each in sample[1]]  # 输入中的POI_ID
            label_seq = [each[0] for each in sample[2]]  # 标签中的POI_ID，实际上就是往后错开一项的POI
            input_seq_time = [each[1] for each in sample[1]]  # 输入中的时间戳
            label_seq_time = [each[1] for each in sample[2]]  # 标签中的时间戳
            label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]  # 取得POI对应的类别序号
            # (X, 320)，每个POI访问记录被转化为了几个特征拼起来的embedding
            input_seq_embed = torch.stack(
                input_traj_to_embeddings(in_args, sample, poi_embeddings, user_embed_model, time_embed_model,
                                         cat_embed_model,
                                         embed_fuse_model1, embed_fuse_model2, poi_idx2cat_idx_dict, user_id2idx_dict))
            batch_seq_embeds.append(input_seq_embed)  # 保存stack后的embedding
            batch_seq_lens.append(len(input_seq))  # 保存输入序列长度
            batch_input_seqs.append(input_seq)  # 保存POI_ID
            batch_seq_labels_poi.append(torch.LongTensor(label_seq))  # todo 对于标签而言，只需要直接存值就可以了
            batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
            batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

        # Pad seqs for batch training
        # todo (20, X, 320) 原本是一个embedding序列数组，其中序列长度各不相同，此处以最长序列的长度为基准，其他短序列以-1为值进行padding，最终将这20个用户的序列stack起来
        batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)  # (20, x, 320)
        label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)  # (20, x)
        label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)  # (20, x)
        label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)  # (20, x)

        # Feedforward
        x = batch_padded.to(device=in_args.device, dtype=torch.float)  # 输入到Transformer的forward的参数
        y_poi = label_padded_poi.to(device=in_args.device, dtype=torch.long)
        y_time = label_padded_time.to(device=in_args.device, dtype=torch.float)
        y_cat = label_padded_cat.to(device=in_args.device, dtype=torch.long)
        y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

        # Graph Attention adjusted prob
        y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi, node_attn_model, X, laplacian_matrix,
                                                        batch_seq_lens, batch_input_seqs)  # 概率图的概率值未经标准化，此处加上后可能会变得很大

        loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)  # 交叉熵 todo 不是很清楚为什么要转置
        loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)  # 均方损失
        loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)  # 交叉熵
        loss = loss_poi + loss_time * in_args.time_loss_weight + loss_cat
        # 一个batch的Final loss
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # todo 这里retain_graph可能是因为下面在计算acc等
            optimizer.step()

        # Performance measurement 每个batch都会重置
        top1_acc = 0
        top5_acc = 0
        top10_acc = 0
        top20_acc = 0
        mAP20 = 0
        mrr = 0
        batch_label_pois = y_poi.detach().cpu().numpy()  # POI标签
        batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()  # POI预测值
        batch_pred_times = y_pred_time.detach().cpu().numpy()  # 时间预测值
        batch_pred_cats = y_pred_cat.detach().cpu().numpy()  # 类别预测值
        for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
            label_pois = label_pois[:seq_len]  # shape: (seq_len, )  去掉padding后的标签序列
            pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
            top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
            top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
            top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
            top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
            mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
            mrr += MRR_metric_last_timestep(label_pois, pred_pois)  # todo 算loss的时候是会算多个loss，但最后求精度时是只算POI
        batches_top1_acc_list.append(top1_acc / len(batch_label_pois))  # 每次累加这个batch的acc等
        batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
        batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
        batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
        batches_mAP20_list.append(mAP20 / len(batch_label_pois))
        batches_mrr_list.append(mrr / len(batch_label_pois))  # 几个metric的平均值
        batches_loss_list.append(loss.detach().cpu().numpy())  # 总loss
        batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
        batches_time_loss_list.append(loss_time.detach().cpu().numpy())
        batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

        # Report training progress

        if mode == 'train' and (b_idx % (in_args.batch * 5)) == 0:  # 每100个batch打印一次，类似滑动窗口，不过只是右边缘往右滑
            log_results_in_epoch(logging, 'train', y_pred_poi, epoch_num, b_idx, loss, top1_acc, batch_label_pois,
                                 batches_loss_list, batches_poi_loss_list, batches_time_loss_list,
                                 batches_top1_acc_list,
                                 batches_top5_acc_list, batches_top10_acc_list,
                                 batches_top20_acc_list,
                                 batches_mAP20_list,
                                 batches_mrr_list, batch, batch_seq_lens, batch_pred_pois, poi_idx2cat_idx_dict,
                                 batch_pred_cats, batch_seq_labels_time
                                 , batch_pred_times)
        if mode == 'val' and (b_idx % (in_args.batch * 2)) == 0:
            log_results_in_epoch(logging, 'val', y_pred_poi, epoch_num, b_idx, loss, top1_acc, batch_label_pois,
                                 batches_loss_list, batches_poi_loss_list, batches_time_loss_list,
                                 batches_top1_acc_list,
                                 batches_top5_acc_list, batches_top10_acc_list,
                                 batches_top20_acc_list,
                                 batches_mAP20_list,
                                 batches_mrr_list, batch, batch_seq_lens, batch_pred_pois, poi_idx2cat_idx_dict,
                                 batch_pred_cats, batch_seq_labels_time
                                 , batch_pred_times)
    if mode == 'train':
        train_epochs_loss_list.append(np.mean(batches_loss_list))
        train_epochs_poi_loss_list.append(np.mean(batches_poi_loss_list))
        train_epochs_time_loss_list.append(np.mean(batches_time_loss_list))
        train_epochs_cat_loss_list.append(np.mean(batches_cat_loss_list))
        train_epochs_top1_acc_list.append(np.mean(batches_top1_acc_list))
        train_epochs_top5_acc_list.append(np.mean(batches_top5_acc_list))
        train_epochs_top10_acc_list.append(np.mean(batches_top10_acc_list))
        train_epochs_top20_acc_list.append(np.mean(batches_top20_acc_list))
        train_epochs_mAP20_list.append(np.mean(batches_mAP20_list))
        train_epochs_mrr_list.append(np.mean(batches_mrr_list))
    else:
        val_epochs_loss_list.append(np.mean(batches_loss_list))
        val_epochs_poi_loss_list.append(np.mean(batches_poi_loss_list))
        val_epochs_time_loss_list.append(np.mean(batches_time_loss_list))
        val_epochs_cat_loss_list.append(np.mean(batches_cat_loss_list))
        val_epochs_top1_acc_list.append(np.mean(batches_top1_acc_list))
        val_epochs_top5_acc_list.append(np.mean(batches_top5_acc_list))
        val_epochs_top10_acc_list.append(np.mean(batches_top10_acc_list))
        val_epochs_top20_acc_list.append(np.mean(batches_top20_acc_list))
        val_epochs_mAP20_list.append(np.mean(batches_mAP20_list))
        val_epochs_mrr_list.append(np.mean(batches_mrr_list))
        # Monitor loss and score  # todo 这个是验证集几个loss的总和
        monitor_loss = np.mean(batches_loss_list)
        monitor_score = np.mean(np.mean(batches_top1_acc_list) * 4 + np.mean(batches_top5_acc_list))
        # Learning rate scheduler  # todo 根据验证集的loss来更新学习率
        lr_scheduler.step(monitor_loss)
        # Save poi and user embeddings
        if in_args.save_embeds:
            embeddings_save_dir = os.path.join(in_args.save_dir, 'embeddings')
            if not os.path.exists(embeddings_save_dir):
                os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score:
                # Save poi embeddings
                poi_embeddings = poi_embed_model(X, laplacian_matrix).detach().cpu().numpy()
                poi_embedding_list = []
                for poi_idx in range(len(poi_id2idx_dict)):
                    poi_embedding = poi_embeddings[poi_idx]
                    poi_embedding_list.append(poi_embedding)
                save_poi_embeddings = np.array(poi_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_poi_embeddings'), save_poi_embeddings)
                # Save user embeddings
                user_embedding_list = []
                for user_idx in range(len(user_id2idx_dict)):
                    input = torch.LongTensor([user_idx]).to(device=in_args.device)
                    user_embedding = user_embed_model(input).detach().cpu().numpy().flatten()
                    user_embedding_list.append(user_embedding)
                user_embeddings = np.array(user_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_user_embeddings'), user_embeddings)
                # Save cat embeddings
                cat_embedding_list = []
                for cat_idx in range(len(cat_id2idx_dict)):
                    input = torch.LongTensor([cat_idx]).to(device=in_args.device)
                    cat_embedding = cat_embed_model(input).detach().cpu().numpy().flatten()
                    cat_embedding_list.append(cat_embedding)
                cat_embeddings = np.array(cat_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_cat_embeddings'), cat_embeddings)
                # Save time embeddings
                time_embedding_list = []
                for time_idx in range(in_args.time_units):
                    input = torch.FloatTensor([time_idx]).to(device=in_args.device)
                    time_embedding = time_embed_model(input).detach().cpu().numpy().flatten()
                    time_embedding_list.append(time_embedding)
                time_embeddings = np.array(time_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_time_embeddings'), time_embeddings)
        # Save model state dict
        if in_args.save_weights:
            state_dict = {
                'epoch': epoch_num,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
                'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                'node_attn_map': node_attn_model(X, laplacian_matrix),
                'args': in_args,
                'epoch_train_metrics': {
                    'epoch_train_loss': train_epochs_loss_list[epoch_num],
                    'epoch_train_poi_loss': train_epochs_poi_loss_list[epoch_num],
                    'epoch_train_time_loss': train_epochs_time_loss_list[epoch_num],
                    'epoch_train_cat_loss': train_epochs_cat_loss_list[epoch_num],
                    'epoch_train_top1_acc': train_epochs_top1_acc_list[epoch_num],
                    'epoch_train_top5_acc': train_epochs_top5_acc_list[epoch_num],
                    'epoch_train_top10_acc': train_epochs_top10_acc_list[epoch_num],
                    'epoch_train_top20_acc': train_epochs_top20_acc_list[epoch_num],
                    'epoch_train_mAP20': train_epochs_mAP20_list[epoch_num],
                    'epoch_train_mrr': train_epochs_mrr_list[epoch_num]
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': val_epochs_loss_list[epoch_num],
                    'epoch_val_poi_loss': val_epochs_poi_loss_list[epoch_num],
                    'epoch_val_time_loss': val_epochs_time_loss_list[epoch_num],
                    'epoch_val_cat_loss': val_epochs_cat_loss_list[epoch_num],
                    'epoch_val_top1_acc': val_epochs_top1_acc_list[epoch_num],
                    'epoch_val_top5_acc': val_epochs_top5_acc_list[epoch_num],
                    'epoch_val_top10_acc': val_epochs_top10_acc_list[epoch_num],
                    'epoch_val_top20_acc': val_epochs_top20_acc_list[epoch_num],
                    'epoch_val_mAP20': val_epochs_mAP20_list[epoch_num],
                    'epoch_val_mrr': val_epochs_mrr_list[epoch_num]
                }
            }
            model_save_dir = os.path.join(in_args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                max_val_score = monitor_score
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)


def log_results_in_epoch(logger, set_name, y_pred_poi, epoch_num, batch_num, loss, top1_acc, batch_label_pois,
                         batches_loss_list, batches_poi_loss_list, batches_time_loss_list, batches_top1_acc_list,
                         batches_top5_acc_list, batches_top10_acc_list, batches_top20_acc_list, batches_mAP20_list,
                         batches_mrr_list, batch, batch_seq_lens, batch_pred_pois, poi_idx2cat_idx_dict,
                         batch_pred_cats, batch_seq_labels_time
                         , batch_pred_times):
    sample_idx = 0
    batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
    logger.info(f'Epoch:{epoch_num}, batch:{batch_num}, '
                f'{set_name}_batch_loss:{loss.item():.2f}, '
                f'{set_name}_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                f'{set_name}_move_loss:{np.mean(batches_loss_list):.2f} \n'
                f'{set_name}_move_poi_loss:{np.mean(batches_poi_loss_list):.2f} \n'
                f'{set_name}l_move_time_loss:{np.mean(batches_time_loss_list):.2f} \n'
                f'{set_name}l_move_top1_acc:{np.mean(batches_top1_acc_list):.4f} \n'
                f'{set_name}_move_top5_acc:{np.mean(batches_top5_acc_list):.4f} \n'
                f'{set_name}_move_top10_acc:{np.mean(batches_top10_acc_list):.4f} \n'
                f'{set_name}_move_top20_acc:{np.mean(batches_top20_acc_list):.4f} \n'
                f'{set_name}_move_mAP20:{np.mean(batches_mAP20_list):.4f} \n'
                f'{set_name}_move_MRR:{np.mean(batches_mrr_list):.4f} \n'
                f'traj_id:{batch[sample_idx][0]}\n'
                f'input_seq:{batch[sample_idx][1]}\n'
                f'label_seq:{batch[sample_idx][2]}\n'
                f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                '=' * 100)


def log_results_end_epoch(in_args, logger, epoch_num):
    # Print epoch results
    logger.info(f"Epoch {epoch_num}/{in_args.epochs}\n"
                f"train_loss:{train_epochs_loss_list[epoch_num]:.4f}, "
                f"train_poi_loss:{train_epochs_poi_loss_list[epoch_num]:.4f}, "
                f"train_time_loss:{train_epochs_time_loss_list[epoch_num]:.4f}, "
                f"train_cat_loss:{train_epochs_cat_loss_list[epoch_num]:.4f}, "
                f"train_top1_acc:{train_epochs_top1_acc_list[epoch_num]:.4f}, "
                f"train_top5_acc:{train_epochs_top5_acc_list[epoch_num]:.4f}, "
                f"train_top10_acc:{train_epochs_top10_acc_list[epoch_num]:.4f}, "
                f"train_top20_acc:{train_epochs_top20_acc_list[epoch_num]:.4f}, "
                f"train_mAP20:{train_epochs_mAP20_list[epoch_num]:.4f}, "
                f"train_mrr:{train_epochs_mrr_list[epoch_num]:.4f}\n"
                f"val_loss: {val_epochs_loss_list[epoch_num]:.4f}, "
                f"val_poi_loss: {val_epochs_poi_loss_list[epoch_num]:.4f}, "
                f"val_time_loss: {val_epochs_time_loss_list[epoch_num]:.4f}, "
                f"val_cat_loss: {val_epochs_cat_loss_list[epoch_num]:.4f}, "
                f"val_top1_acc:{val_epochs_top1_acc_list[epoch_num]:.4f}, "
                f"val_top5_acc:{val_epochs_top5_acc_list[epoch_num]:.4f}, "
                f"val_top10_acc:{val_epochs_top10_acc_list[epoch_num]:.4f}, "
                f"val_top20_acc:{val_epochs_top20_acc_list[epoch_num]:.4f}, "
                f"val_mAP20:{val_epochs_mAP20_list[epoch_num]:.4f}, "
                f"val_mrr:{val_epochs_mrr_list[epoch_num]:.4f}")


if __name__ == '__main__':
    args = parameter_parser()
    train(args)

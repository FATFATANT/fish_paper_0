""" Build the user-agnostic global trajectory flow map from the sequence dataset """
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()  # 新建有向图，注意是有向图， todo 因为是有向图，所以生成的邻接矩阵不是对称的
    users = list(set(df['user_id'].to_list()))  # 共有1047个用户编号
    if exclude_user in users:
        users.remove(exclude_user)  # 去除指定编号用户，但此处应该是没有
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]  # 取出对应用户id的记录，这些记录是升序排列的

        # 将所有的结点加入到图中
        for i, row in user_df.iterrows():  # i是原df中的索引，row就是对应行的记录
            node = row['POI_id']
            if node not in G.nodes():  # 如果图中没有该POI，就构造出这个POI结点，将POI编号、类别编号、类别名、经纬度作为其属性
                G.add_node(row['POI_id'],
                           checkin_cnt=1,
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:  # 如果图中已有该POI就增加一个访问次数
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            # 若是第一次进入循环或者进入新的一条轨迹，保存当前轨迹编号和poi编号并跳到下个循环 todo 这个主要是避免将两条轨迹之间的终点起点也连一条边，这个边其实还是user-agnostic的，是在全局角度上看两个POI之间连的边的数量
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1  # 若POI之间已经有边，权重加1
            else:
                G.add_edge(previous_poi_id, poi_id, weight=1)  # 若POI之间没有边，新建一条边，权重设置为1
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    """
    Save graph to an adj matrix file and a nodes file
    Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.
    """
    # Save adj matrix
    nodelist = G.nodes()  # 获取图中POI结点的ID
    A = nx.adjacency_matrix(G, nodelist=nodelist)  # 生成图的邻接矩阵，是个稀疏矩阵
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')  # todense就是将稀疏矩阵展开为一个正常矩阵

    # Save nodes list
    nodes_data = list(G.nodes.data())  # 包含详细属性信息的POI结点，每项是个tuple，第0项是结点名，第1项是属性字典
    # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:  # print定向写入文件
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)  # 表头
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},{poi_catname},'
                  f'{latitude},{longitude}', file=f)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))  # 压缩为pkl文件


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}  # POI编号到序号的映射

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:  # 编号为键，序号为值，就是一个reindex操作
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:  # 将结点间每条边用一个三元tuple来表示，结点用序号表示
        for edge in nx.generate_edgelist(G, data=['weight']):  # edge用一条字符串表示，中间用空格分隔
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)  # 将编号映射为序号后将一条条边写入


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    dst_dir = r'dataset/NYC'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, 'NYC_train.csv'))  # (83228, 15)
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)  # 4980个结点，37756条边

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)  # 将图保存为graph.pickle，注意这个图中表示了所有的POI数量和POI之间连续两两访问的边
    save_graph_to_csv(G, dst_dir=dst_dir)  # 邻接矩阵graph_A.csv体现出了图的边信息，graph_X.csv体现出了图的结点信息，结点信息的count可以表现结点的流行度
    save_graph_edgelist(G, dst_dir=dst_dir)

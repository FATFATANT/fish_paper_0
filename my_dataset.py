from torch.utils.data import Dataset
from tqdm import tqdm


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, in_args, train_df, poi_id2idx_dict):  # todo 预处理训练集csv
        self.df = train_df  # 原始训练数据
        self.traj_seqs = []  # todo traj id的组成： [user id]_[traj no]
        self.input_seqs = []
        self.label_seqs = []
        # todo 数据集中已经将用户所属的轨迹做了标记
        for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):  # 说明遍历的轨迹也是user_agnostic的
            traj_df = train_df[train_df['trajectory_id'] == traj_id]  # 找出属于该条轨迹的所有记录
            poi_ids = traj_df['POI_id'].to_list()  # 这些记录的POI编号
            poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]  # POI编号转序号
            # todo 这个轨迹不是以天为单位的，若干天的记录会被看作是一条轨迹，目前还不知道划分标准
            # todo 该条记录的访问时刻处于一天从0点开始的24小时的百分比
            time_feature = traj_df[in_args.time_feature].to_list()
            # todo 但是最终也是两两配对的访问记录对，前一个到后一个，那么这个记录的划分规则可能也不是特别重要
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):  # 后一项作为前一项的标签，所以遍历的次数是序列长度-1
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            if len(input_seq) < in_args.short_traj_threshold:  # 该条轨迹必须大于最小门限值
                continue

            self.traj_seqs.append(traj_id)  # 当前轨迹序号
            self.input_seqs.append(input_seq)  # 由一条轨迹拆出的轨迹访问对
            self.label_seqs.append(label_seq)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index]  # batch中每一项返回轨迹编号、输入、标签


class TrajectoryDatasetVal(Dataset):
    def __init__(self, in_args, val_df, user_id2idx_dict, poi_id2idx_dict):
        self.df = val_df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(val_df['trajectory_id'].tolist())):
            user_id = traj_id.split('_')[0]

            # todo 和训练集的代码的区别:若测试集中出现了验证集中没有的用户，就跳过该用户
            if user_id not in user_id2idx_dict.keys():
                continue

            # Ger POIs idx in this trajectory
            traj_df = val_df[val_df['trajectory_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = []
            time_feature = traj_df[in_args.time_feature].to_list()

            for each in poi_ids:
                if each in poi_id2idx_dict.keys():  # 若POI在训练集中没有出现过也去掉
                    poi_idxs.append(poi_id2idx_dict[each])  # 将POI编号转序号
                else:
                    # 去掉验证集中训练集中该用户未访问的POI
                    continue

            # Construct input seq and label seq
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):  # 同样构建输入和标签的轨迹对
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            # Ignore seq if too short
            if len(input_seq) < in_args.short_traj_threshold:
                continue

            self.input_seqs.append(input_seq)  # 当前轨迹序号
            self.label_seqs.append(label_seq)  # 当前轨迹拆分出的输入
            self.traj_seqs.append(traj_id)  # 当前轨迹拆分出的标签

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index]

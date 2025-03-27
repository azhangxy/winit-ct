import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ClimateDataset(Dataset):
    def __init__(self, X_real, Y_syn, A_syn, subset_name:str, treatment_mode='multiclass', **kwargs):
        # dictinary with unstructured time-series data
        self.data = {
            'X': X_real,
            'Y': Y_syn,
            'A': A_syn
        }

        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.treatment_mode = treatment_mode
        self.exploded = False
        self.norm_const = 1.0
        self.subset_name = subset_name

    def __getitem__(self, index):
        result = {k: v[index] for k, v in self.data.items() if hasattr(v, '__len__') and len(v) == len(self)}
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return self.data['X'].shape[0]

    def get_scaling_params(self):
        X = self.data['X']
        mean = {col: X[:, :, i].mean() for i, col in enumerate(['PM2.5', 'TEMP', 'O3', 'WSPM', 'DEWP', 'station'])}
        std = {col: X[:, :, i].std() for i, col in enumerate(['PM2.5', 'TEMP', 'O3', 'WSPM', 'DEWP', 'station'])}
        return mean, std
    
    def process_data(self, scaling_params):
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            mean = {col: self.data['X'][:, :, i].mean() for i, col in enumerate(['PM2.5', 'TEMP', 'O3', 'WSPM', 'DEWP', 'station'])}
            std = {col: self.data['X'][:, :, i].std() for i, col in enumerate(['PM2.5', 'TEMP', 'O3', 'WSPM', 'DEWP', 'station'])}

            horizon = 1
            offset = 1

            # 标准化特征
            # eps = 1e-8  # 小常数
            features = ['PM2.5', 'TEMP', 'O3', 'WSPM', 'DEWP','station']
            '''for i, col in enumerate(features):
                self.data['X'][:, :, i] = (self.data['X'][:, :, i] - mean[col]) / (std[col] + eps)
            mean_Y = self.data['Y'].mean()
            std_Y = self.data['Y'].std()
            eps = 1e-8

            # 对 Y 进行标准化处理
            self.data['Y'] = (self.data['Y'] - mean_Y) / (std_Y + eps)'''
            X = self.data['X']
            Y = self.data['Y']
            A = self.data['A']
            # sequence_lengths 数组的每个元素都被赋值为 X.shape[1]
            sequence_lengths = np.full(X.shape[0], X.shape[1])

            treatments = A[:, :-offset, np.newaxis]
            if self.treatment_mode == 'multiclass':
                # 这里假设治疗是二分类，可根据实际情况修改
                one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 2))
                for i in range(treatments.shape[0]):
                    one_hot_treatments[i, :, treatments[i, :, 0].astype(int)] = 1
                one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

                self.data['prev_treatments'] = one_hot_previous_treatments
                self.data['current_treatments'] = one_hot_treatments
            
            current_covariates = X[:, :-offset, :-1] # 前5个特征 vitals
            outputs = Y[:, horizon:, np.newaxis]

            # output_means = mean[features[0]]  # 假设以第一个特征为例
            # output_stds = std[features[0]]
            output_means = np.mean(Y)
            output_stds = np.std(Y)

            # 添加有效条目
            active_entries = np.zeros(outputs.shape)
            for i in range(sequence_lengths.shape[0]):
                sequence_length = int(sequence_lengths[i])
                active_entries[i, :sequence_length, :] = 1

            self.data['current_covariates'] = current_covariates
            self.data['outputs'] = outputs
            self.data['active_entries'] = active_entries

            self.data['unscaled_outputs'] = (outputs * output_stds + output_means)

            self.scaling_params = {
                'input_means': np.array([mean[col] for col in features]),
                'inputs_stds': np.array([std[col] for col in features]),
                'output_means': output_means,
                'output_stds': output_stds
            }

            # 统一数据格式
            self.data['prev_outputs'] = Y[:, :-horizon, np.newaxis]

              
            self.data['static_features'] = X[:, 0 , 5:6]

            zero_init_treatment = np.zeros(shape=[current_covariates.shape[0], 1, self.data['prev_treatments'].shape[-1]])
            self.data['prev_treatments'] = np.concatenate([zero_init_treatment, self.data['prev_treatments']], axis=1)
            # 添加 'vitals' 到 data 字典
            self.data['vitals'] = current_covariates
            self.data['next_vitals'] = X[:, 1:, :-1]
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')
            '''# 删除不需要的数据
            del self.data['X']
            del self.data['Y']
            del self.data['A']
            del self.data['unscaled_outputs']
            '''
            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data
    
    def explode_trajectories(self, projection_horizon):
        # 实现时间序列切割
        pass

    def process_sequential(self, encoder_r, projection_horizon):
        # 准备多步预测训练数据
        pass

    def process_sequential_test(self, projection_horizon):
        # 准备多步预测评估数据
        pass

    def process_autoregressive_test(self, encoder_r, encoder_outputs, projection_horizon):
        # 准备自回归多步预测数据
        pass

    def process_sequential_multi(self, projection_horizon):
        # 准备因果Transformer多步预测数据
        pass

from src.data.dataset_collection import RealDatasetCollection
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


from src.data.dataset_collection import SyntheticDatasetCollection
from src.data.climate_data.generate_temporal_data import preprocess_climate_data, generate_climate_outcomes

class ClimateDatasetCollection(SyntheticDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
    """

    def __init__(self, 
                 max_seq_length=168,
                 projection_horizon=0,
                 **kwargs):
        # 执行流程
        self.projection_horizon = projection_horizon
        self.max_seq_length = max_seq_length
        import os
        file_path = os.path.join(os.path.dirname(__file__), 'PRSA_Data_Aotizhongxin.csv')
        df = pd.read_csv(file_path)
        X_real = preprocess_climate_data(df)
        Y_syn, A_syn = generate_climate_outcomes(X_real)
        # print(X_real.shape, Y_syn.shape, A_syn.shape)
        # (202, 168, 5) (202, 168) (202, 168)
        
        test_size = 0.2
        train_size = 0.8
        # 划分训练集、验证集和测试集
        try:
            X_train_val, X_test, Y_train_val, Y_test, A_train_val, A_test = train_test_split(
                X_real, Y_syn, A_syn, test_size=test_size, random_state=42, shuffle=True)
            X_train, X_val, Y_train, Y_val, A_train, A_val = train_test_split(
                X_train_val, Y_train_val, A_train_val, test_size=test_size / (test_size + train_size), random_state=42, shuffle=True)
            # 进一步划分测试集为 test_cf_one_step 和 test_cf_treatment_seq
            X_test_one_step, X_test_treatment_seq, Y_test_one_step, Y_test_treatment_seq, A_test_one_step, A_test_treatment_seq = train_test_split(
                X_test, Y_test, A_test, test_size=0.5, random_state=42, shuffle=True)
        except ValueError as e:
            print(f"数据划分出错: {e}")

        # 初始化训练集、验证集和测试集的 ClimateDataset 对象
        self.train_f = ClimateDataset(X_train, Y_train, A_train, subset_name='train')
        self.val_f = ClimateDataset(X_val, Y_val, A_val, subset_name='val')
        self.test_cf_one_step = ClimateDataset(X_test_one_step, Y_test_one_step, A_test_one_step,subset_name='test')
        self.test_cf_treatment_seq = ClimateDataset(X_test_treatment_seq, Y_test_treatment_seq, A_test_treatment_seq, subset_name='test')
        # 获取训练集的缩放参数
        self.train_scaling_params = self.train_f.get_scaling_params()

        self.autoregressive = True
        self.has_vitals = True
        
        # process_data_multi()



import pandas as pd
import numpy as np

def preprocess_climate_data(df, time_steps=7):
    # 排序并填补缺失值
    df = df.sort_values(['year', 'month', 'day', 'hour']).ffill()
    
    # 站点编码
    vocab = {'Aotizhongxin': 0}
    df['station'] = df['station'].map(vocab)

    # 标准化并保存均值和标准差
    # 特征：PM2.5（颗粒物浓度）、TEMP（温度）、O3（臭氧）、WSPM（风速）、DEWP（露点）、station(站点)
    # feature_stats = {}
    eps = 1e-8
    for col in ['PM2.5', 'TEMP', 'O3', 'DEWP', 'WSPM', 'station']:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + eps)
        '''feature_stats[col] = {'mean': mean, 'std': std}'''
    
    # 生成时间序列样本（每周数据）
    features = ['PM2.5', 'TEMP', 'O3', 'WSPM', 'DEWP','station']
    window_size = time_steps * 24
    data = df[features].values
    
    # 使用滑动窗口生成样本（不重叠）
    num_samples = len(data) // window_size
    data = data[:num_samples * window_size]
    X_real = data.reshape(num_samples, window_size, len(features))
    
    return X_real  #, feature_stats

def generate_climate_outcomes(X, γ=0.85, ω_pred=1.5):
    N, T, d = X.shape
    Y, A = np.zeros((N, T)), np.zeros((N, T))
    
    # 计算标准化后的阈值
    # pm_threshold = (75 - feature_stats['PM2.5']['mean']) / feature_stats['PM2.5']['std']
    # wspm_threshold = (2.0 - feature_stats['WSPM']['mean']) / feature_stats['WSPM']['std']
    
    ℐ_prog = [1]  # 温度作为预后特征
    ℐ_0, ℐ_1 = [0], [2]  # PM2.5和O3作为处理效应特征
    
    for i in range(N):
        Y_prev = 0
        for t in range(T):
            # 动态治疗分配 取决于当前时刻的协变量 pm2.5大于75 && wspm风速小于2.0
            A[i, t] = (X[i, t, 3] < 2.0) & (X[i, t, 0] >75)
            
            # 潜在结果模型
            Y0 = γ * Y_prev + X[i, t, ℐ_prog].mean() + ω_pred * X[i, t, ℐ_0].mean()
            Y1 = γ * Y_prev + X[i, t, ℐ_prog].mean() + ω_pred * X[i, t, ℐ_1].mean()
            
            Y[i, t] = A[i, t] * Y1 + (1 - A[i, t]) * Y0 + np.random.normal(0, 1.2)
            Y_prev = Y[i, t]
    eps = 1e-8
    mean_y = Y.mean()
    std_y = Y.std()
    Y = (Y- mean_y) / (std_y + eps)
    return Y, A

# 执行流程
'''df = pd.read_csv('src\data\climate_data\PRSA_Data_Aotizhongxin.csv')
X_real = preprocess_climate_data(df)
Y_syn, A_syn = generate_climate_outcomes(X_real)

print(X_real.shape, Y_syn.shape, A_syn.shape)'''
# (208, 168, 6) (208, 168) (208, 168)
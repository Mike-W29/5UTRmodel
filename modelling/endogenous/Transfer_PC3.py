# Dataframe
import json
import pandas as pd
import numpy as np
import random
# Visualization
# import plotly.express as px
# Deeplearning
import tensorflow.keras.layers as L
import tensorflow as tf
# Sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Setting seeds
tf.random.set_seed(3407)
np.random.seed(3407)
random.seed(3407)
min_rna_rkpm = float(1)
min_ribo_rkpm = float(1)
import pandas as pd
df = pd.read_csv('/mnt/5UTRDL/data/df_counts_and_len.TE_sorted.pc3.with_annot.txt', sep=" ", index_col=0)
#df = pd.read_csv('/mnt/5UTRDL/data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt', sep=" ", index_col=0)

df = df[(df["rpkm_rnaseq"] > min_rna_rkpm) & (df["rpkm_riboseq"] > min_ribo_rkpm)]



trian_seq = pd.read_csv("~/data/endogenous/Homo_trian_seq_feature_table_maxBPspan30.txt", delimiter='\t')
trian_seq['len'] = trian_seq['utr'].apply(len)
filtered_trian_seq = trian_seq[trian_seq['Transcript_ID'].isin(df['ensembl_tx_id'])]
filtered_df = df[df['ensembl_tx_id'].isin(filtered_trian_seq['Transcript_ID'])]
df_merged = pd.merge(filtered_df, filtered_trian_seq[['Transcript_ID','utr', 'Sequence', 'MFE_UTR', 'structure_UTR', 'MFE_full','structure_full','predicted_loop_type_utr','predicted_loop_type_full_sequence']], 
                     left_on='ensembl_tx_id', right_on='Transcript_ID', how='left')
df_merged['log_te'] = np.log(df_merged['te'])
df_merged['len'] = df_merged['utr'].apply(len)
filtered_df = df_merged[(df_merged['len'] >= 25) & (df_merged['len'] <= 300)]
df_merged = filtered_df
df_merged['scaled_log_te'] = preprocessing.StandardScaler().fit_transform(df_merged['log_te'].values.reshape(-1, 1))
df_merged['scaled_log_te'].describe()

idx = df_merged.groupby('Sequence')['te'].idxmax()
filtered_df = df_merged.loc[idx]
df_merged = filtered_df
indices = df_merged.index.tolist()
np.random.seed(3407)
np.random.shuffle(indices)
train_size = int(len(df_merged) * 0.9)
# 分割索引为训练集和测试集
train_indices = indices[:train_size]
test_indices = indices[train_size:]

e_train = df_merged.loc[train_indices]
e_test = df_merged.loc[test_indices]

e_train['seq300'] = 330*'N' +e_train['Sequence']
e_train['seq300'] = e_train['seq300'].str[-330:]
e_train['stru300'] = 330*'N' +e_train['structure_full']
e_train['stru300'] = e_train['stru300'].str[-330:]

import numpy as np

def one_hot_encode(df, seqcol='utr100', strucol='stru300', seq_len=100):
    # One-hot encoding for nucleotides
    nuc_d = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    
    # One-hot encoding for nucleotide + structure combinations (12 categories)
    stru_d = {
        'A(': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'A)': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'A.': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'C(': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        'C)': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
        'C.': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
        'G(': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
        'G)': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
        'G.': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
        'T(': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
        'T)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
        'T.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        'NN': [0] * 12  # Padding/unknown values
    }
    
    # 创建两个空矩阵
    seq_vectors = np.empty([len(df), seq_len, 4])  # 存放标准的碱基 one-hot 编码
    stru_vectors = np.empty([len(df), seq_len, 12])  # 存放联合编码

    # 遍历每个样本
    for i, (seq, stru) in enumerate(zip(df[seqcol].str[:seq_len], df[strucol].str[:seq_len])):
        seq = seq.upper()
        stru = stru.upper()

        # 碱基的 one-hot 编码
        seq_vectors[i, :, :] = np.array([nuc_d.get(x, [0, 0, 0, 0]) for x in seq])

        # 结构联合 one-hot 编码
        stru_vectors[i, :, :] = np.array([stru_d.get(x + y, [0] * 12) for x, y in zip(seq, stru)])

    return seq_vectors, stru_vectors

seq_e_train,stru_e_train= one_hot_encode(e_train,seqcol='seq300',strucol = 'stru300', seq_len=330)



from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Ref in #https://zhuanlan.zhihu.com/p/97525394
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = K.shape(x)[1]  # 动态获取序列长度

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c,a

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

e_train['len'] = e_train['Sequence'].apply(len)
e_test['len'] = e_test['Sequence'].apply(len)
e_train['Norm_mfe'] = e_train['MFE_full'] / e_train['len']
e_test['Norm_mfe'] = e_test['MFE_full'] / e_test['len']

from sklearn.preprocessing import MinMaxScaler

metadata_cols = [
    'Norm_mfe'
]

# 示例数据
train_meta_data = e_train[metadata_cols].values
test_meta_data = e_test[metadata_cols].values

# 初始化空的 DataFrame 来存储归一化结果
scaled_train_meta = pd.DataFrame(columns=metadata_cols)
scaled_test_meta = pd.DataFrame(columns=metadata_cols)

# 对每个元数据列进行 Min-Max 归一化
for i, col in enumerate(metadata_cols):
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 训练归一化模型并保存
    scaled_train_meta[col] = scaler.fit_transform(train_meta_data[:, i].reshape(-1, 1)).flatten()
    scaled_test_meta[col] = scaler.transform(test_meta_data[:, i].reshape(-1, 1)).flatten()

# 输出归一化结果
print("归一化后的训练元数据：")
print(scaled_train_meta.head())

print("\n归一化后的测试元数据：")
print(scaled_test_meta.head())


from sklearn.preprocessing import StandardScaler
import joblib  # 推荐用于模型或对象的持久化

# 创建并拟合 StandardScaler
scaler = StandardScaler()
e_train['scaled_log_te'] = scaler.fit_transform(e_train['log_te'].values.reshape(-1, 1))

# 保存 scaler 到本地文件（比如 .pkl）
joblib.dump(scaler, '/PC3_train_log_te_scaler.pkl')


from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

metadata_cols = ['Norm_mfe']
train_meta_data = e_train[metadata_cols].values

# 创建并拟合 MinMaxScaler
MinMax = MinMaxScaler(feature_range=(0, 1))
scaled_train_meta = pd.DataFrame(MinMax.fit_transform(train_meta_data), columns=metadata_cols)

# 保存 scaler 到本地文件
joblib.dump(MinMax, '/PC3_norm_mfe_scaler.pkl')


from tensorflow.keras.models import load_model

# 定义自定义层的字典
custom_objects = {
    'Attention': Attention  # 确保你的自定义 Attention 层已定义
}

# 加载模型
model = load_model('/mnt/5UTRDL/model_file/best_model_endogenesis_M4.h5', custom_objects=custom_objects)

from tensorflow.keras.callbacks import EarlyStopping

X_train,X_val, X_train2,X_val2,X_train_meta, X_val_meta, y_train, y_val = train_test_split(stru_e_train, seq_e_train,scaled_train_meta,
                                                                                           e_train['scaled_log_te'], test_size=0.2, random_state=42)


# 早停机制
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,  # 如果验证集loss在10个epoch内没有降低则停止
    restore_best_weights=True  # 恢复验证集loss最低的权重
)

model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(
    [X_train,X_train2, X_train_meta],  # 训练输入
    y_train,              # 训练输出
    validation_data=([X_val,X_val2,X_val_meta], y_val),  # 验证数据
    batch_size=32,
    epochs=999,
    callbacks=[early_stopping]
)

e_test['seq300'] = 330*'N' +e_test['Sequence']
e_test['seq300'] = e_test['seq300'].str[-330:]
e_test['stru300'] = 330*'N' +e_test['structure_full']
e_test['stru300'] = e_test['stru300'].str[-330:]
test_seq,test_stru = one_hot_encode(e_test,seqcol='seq300',strucol = 'stru300', seq_len=330)


import scipy.stats as stats

def test_data(df, model,test_stru,test_seq,test_meta, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''
    
    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1,1))
    
    # Make predictions
    predictions = model.predict([test_stru,test_seq,scaled_test_meta]).reshape(-1)
    
    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:,output_col] = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, 1)
    return df


def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2




e_test = test_data(df=e_test, model=model, obs_col='log_te',test_stru=test_stru,test_seq=test_seq,test_meta=scaled_test_meta)
r = r2(e_test['log_te'], e_test['pred'])
print('r-squared = ', r)
import pandas as pd
from scipy.stats import spearmanr

# 假设e_test是一个Pandas DataFrame，并且已经包含了'rl'和'pred'两列
# 计算斯皮尔曼相关系数
correlation, p_value = spearmanr(e_test['log_te'], e_test['pred'])

# 输出斯皮尔曼相关系数和p值
print("斯皮尔曼相关系数:", correlation)
print("p值:", p_value)
from tensorflow.keras.losses import MeanSquaredError

mse = MeanSquaredError()
mse_value = mse(e_test['log_te'], e_test['pred']).numpy()
print("MSE值:", mse_value)
print("RMSE值:", mse_value ** 0.5)

model.save('/mnt/5UTRDL/model/model_file/final model/best_model_endogenesis_new_PC3.h5')







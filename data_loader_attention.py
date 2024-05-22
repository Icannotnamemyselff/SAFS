# Necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import util
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def load_data(args,seed, normalize = True):
    np.random.seed(seed)
    if args.problem == "Regression":
        if args.data_type == 'csv':
            x = pd.read_csv("./data/{}/data/data.csv".format(args.data))
            y = pd.read_csv("./data/{}/data/label.csv".format(args.data))
        if args.data_type == 'txt':
            df_np = np.loadtxt('./data/{}/data/data.txt'.format(args.data))
            y = pd.DataFrame(df_np[:, -1:])
            x = pd.DataFrame(df_np[:, :-1])
    if args.problem == "Classification":
        x = pd.read_csv("./data/{}/data.csv".format(args.data))
        y = pd.read_csv("./data/{}/label.csv".format(args.data))
    df_X ,df_y = np.array(x), np.array(y)
    raw_data = np.array(df_X)
    # 归一化
    df_X = pd.DataFrame(df_X)
    if normalize:
        x = df_X.values
        min_max_scaler = MinMaxScaler()
        std_scaler = StandardScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    df_X = np.array(df_X)
    Y = np.array(df_y)


    # 原始数据作为权重训练数据
    x_train_classifier, x_test_classifier, y_train_classifier, y_test_classifier = train_test_split(df_X, Y, test_size=0.2)
    # 打乱顺序
    idx = np.random.permutation(len(y_train_classifier))
    x_train = x_train_classifier[idx, :]
    y_train = y_train_classifier[idx]
    # 特征选择：10%训练权重 有监督
    select_y = util.initLabeled(y_train, seed, args)
    if args.problem == 'Regression':
        label_no = int(x_train.shape[0] / 10)
        x_train_weight = x_train[:label_no, :]
        Y_select = y_train[:label_no]
    if args.problem == 'Classification':
        x_train_weight = x_train[select_y]
        Y_select = y_train[select_y]
        if args.aaafs:
            x_train_weight = df_X
            Y_select = Y
            # 假设你的数据集是X(特征)和y(标签)
            # n_samples = len(Y_select)  # 样本数
            # # 需要打乱的标签数量
            # n_shuffle = int(n_samples * 0.05)
            # # 随机选择5%的索引
            # shuffle_indices = np.random.choice(n_samples, n_shuffle, replace=False)
            # shuffle_indices = list(shuffle_indices)
            # # 获取数据集中所有的unique标签
            # unique_labels = np.unique(Y_select)
            # # 对选中的10%样本的标签随机重新赋值
            # select_y = np.random.choice(unique_labels, n_shuffle)
            # for i in range(len(select_y)):
            #     Y_select[shuffle_indices[i]] = select_y[i]
    # 再次打乱顺序
    idx = np.random.permutation(len(Y_select))
    x_train_weight = x_train_weight[idx, :]
    Y_select = Y_select[idx]
    # 按列计算均值
    # column_means = np.mean(x_train_weight, axis=0)
    # # 每列减去相应的均值
    # x_train_weight = x_train_weight - column_means
    # 加噪声
    import skimage
    if args.noise_type == "mask":
        mask_matrix = np.random.binomial(n=1, p=1.0 - args.noise_amount, size=x_train_weight.shape)
        x_train_weight = x_train_weight * mask_matrix
    if args.noise_type == "s&p":
        x_train_weight = skimage.util.random_noise(x_train_weight, mode="s&p", amount=args.noise_amount)
    if args.noise_type == "gaussian":
        x_train_weight = skimage.util.random_noise(x_train_weight, mode="gaussian", var=args.noise_amount, mean=0)
    if args.noise_type == 'speckle':
        x_train_weight = skimage.util.random_noise(x_train_weight, mode="speckle", var=args.noise_amount, mean=0)

    encoder = LabelBinarizer()
    if args.problem == "Classification":
        Y_select = encoder.fit_transform(Y_select)



    return raw_data, x_test_classifier, y_test_classifier, x_train_weight, Y_select, x_train_classifier, y_train_classifier  #use this 原始数据训练所有

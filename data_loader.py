# Necessary packages
import numpy as np
# import skimage
import pandas as pd
from sklearn.model_selection import train_test_split
import util
import argparse
from sklearn import preprocessing
import skimage

def load_data(args, seed, normalize = True):
    # raw data
    np.random.seed(seed)
    if args.problem == "Classification":
        x = pd.read_csv("./data/{}/data.csv".format(args.data))
        y = pd.read_csv("./data/{}/label.csv".format(args.data))
    df_X, df_y = np.array(x), np.array(y)
    if normalize:
        df_X = pd.DataFrame(df_X)
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X, df_y = pd.DataFrame(x_scaled), pd.DataFrame(df_y)
    df_X, Y = np.array(df_X), np.array(df_y)


    x_unlabel = df_X
    print("Raw_data.shape:", x_unlabel.shape)
    if args.problem == 'Classification':
        x_train_classifier, x_test_classifier, y_train_classifier, y_test_classifier = train_test_split(df_X, Y, test_size=0.2,random_state=seed)
        select_y = util.initLabeled(y_train_classifier, seed, args)
        x_train_weight = x_train_classifier[select_y]
        Y_select = y_train_classifier[select_y]
        if args.aaafs:
            x_train_weight = df_X
            Y_select = Y

    idx = np.random.permutation(len(Y_select))
    x_train_weight = x_train_weight[idx, :]
    Y_select = Y_select[idx]

    # add noise
    if args.noise_type == "mask":
        mask_matrix = np.random.binomial(n=1, p=1.0-args.noise_amount, size=x_train_weight.shape)
        x_train_weight = x_train_weight * mask_matrix
    else:
        if args.noise_type == "s&p":
            x_train_weight = skimage.util.random_noise(x_train_weight, mode="s&p", amount=args.noise_amount)
        else:
            x_train_weight = skimage.util.random_noise(x_train_weight,mode="gaussian", var=args.noise_amount, mean=0)

    return x_unlabel, x_test_classifier, y_test_classifier, x_train_weight, Y_select, x_train_classifier, y_train_classifier

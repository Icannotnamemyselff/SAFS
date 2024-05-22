import xgboost as xgb
import data_loader
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier,MLPRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time


def XGB_train(args):
    # load data
    for seed in range(args.iterations):
        np.random.seed(seed)
        x_unlab, x_test, y_test, x_train_weight, y_train_weight, val_X, val_Y = data_loader.load_data(args, seed)

        # define a model
        if args.problem == "Regression":
            XGB = xgb.XGBRegressor()
        if args.problem == "Classification":
            XGB = xgb.XGBClassifier()
        time_start = time.time()
        XGB.fit(x_train_weight, y_train_weight.ravel())
        A = list(XGB.feature_importances_)
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

        dt2 = pd.DataFrame(A)
        dt2.to_csv(args.path_result + "{}_[{}]_weight.csv".format(args.method, seed), index=0)
        AFS_wight_rank = list(np.argsort(A))[::-1]

        ranking = pd.DataFrame(A)
        ac_score_list = []
        MSE_LIST, MAE_LIST = [], []
        pred_Y = []
        real_test_Y = []
        real_test_Y.append(y_test)

        for K in range(1, args.select_features+1, 1):
            use_train_x = val_X[:, AFS_wight_rank[:K]]
            use_test_x = x_test[:, AFS_wight_rank[:K]]
            if args.problem == "Regression":
                if args.classifier == 'lgb':
                    lgb = lgbm.LGBMRegressor()
                if args.classifier == 'cat':
                    lgb = CatBoostRegressor()
            if args.problem == "Classification":
                if args.classifier == 'lgb':
                    lgb = lgbm.LGBMClassifier(n_jobs=10)
                if args.classifier == 'cat':
                    lgb = CatBoostClassifier()
            lgb.fit(use_train_x, val_Y.ravel())
            y_pre = lgb.predict(use_test_x)
            pred_Y.append(y_pre)
            if args.problem == "Classification":
                accuracy = f1_score(y_test, y_pre, average='micro')
                ac_score_list.append(accuracy)
                print('Using Top {} features| accuracy:{:.4f}'.format(K, accuracy))
            if args.problem == "Regression":
                MSE = mean_squared_error(y_test, y_pre)
                MAE = mean_absolute_error(y_test, y_pre)
                print('Using Top {} features| MSE:{:.4f} |MAE:{:.4f}'.format(K, MSE, MAE))
                MSE_LIST.append(MSE)
                MAE_LIST.append(MAE)

        if args.problem == "Classification":
            dt3 = pd.DataFrame(ac_score_list)
            dt3.to_csv(args.path_result + "{}_[{}]_test_ACC.csv".format(args.method, seed), index=0)
        if args.problem == "Regression":
            dt3 = pd.DataFrame(MSE_LIST)
            dt3.to_csv(args.path_result + "{}_[{}]_test_MSE.csv".format(args.method, seed), index=0)
            dt4 = pd.DataFrame(MAE_LIST)
            dt4.to_csv(args.path_result + "{}_[{}]_test_MAE.csv".format(args.method, seed), index=0)
            pre = pd.DataFrame(pred_Y)
            pre.to_csv(args.path_train + "/{}_pred_test_Y_{}.csv".format(args.method, args.data), index=0)
            y_test = y_test.reshape(1, -1)
            real = pd.DataFrame(y_test)
            real.to_csv(args.path_train + "/{}_real_test_Y_{}.csv".format(args.method, args.data), index=0)
    return A, ac_score_list
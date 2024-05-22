import argparse
import attention_train
import os
import cal_MSE_MAE
import xgboost_train

def exp_main(args):
  if (args.method == "attention"):
      print(args.method)
      # print(args)
      attention_train.attention(args)
      cal_MSE_MAE.cal_all_aver_result(args)
  if (args.method == "xgb"):
      print(args.method)
      print(args)
      xgboost_train.XGB_train(args)
      cal_MSE_MAE.cal_all_aver_result(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', help='number of experiments iterations',default=10,  type=int)
    parser.add_argument('--domain', type=str, default='uci')
    parser.add_argument('--data', type=str, default='concrete')  # 'adult','concrete',mnist, breast_cancer
    parser.add_argument('--method', type=str, default="grape_attention")
    parser.add_argument('--problem', type=str, default="Classification")#Regression,Classification, Binary
    parser.add_argument('--data_type', type=str, default='csv')  # csv / txt
    parser.add_argument('--tspecial', type=str, default="123")  #Binary 123
    parser.add_argument('--log_path', type=str, default="./train")
    # noise parameter
    parser.add_argument('--noise_type', type=str, default="mask")  # mask,s&p, gaussian,  speckle,
    parser.add_argument('--diff_noise_type', type=bool, default=False)  # Always false
    parser.add_argument('--noise_amount', type=float, default=0.0)
    parser.add_argument('--select_sample', type=int, default=1)  # fes-shot 1 5 10 else 300
    parser.add_argument('--hidden_dim', type=int, default=64)  # default 64
    parser.add_argument('--select_features', type=int, default=18)  # 3 #30 #10
    parser.add_argument('--Batch_attention_Ablation', type=str, default="GFS")  # GFS / GFS_B
    parser.add_argument('--classifier' , type=str, default="lgb")  # lgb rf mlp
    parser.add_argument('--GINN', type=str, default='Grape')  # self // semi // Grape // VIME
    parser.add_argument('--aaafs', type=bool, default=True)  # True:

    # all labeled data for feature selection
    parser.add_argument('--all_labeldata_train', type=bool, default=False)  # Always False / True : Feature selection use ALL LABEL
    parser.add_argument('--self_supervised_data_classification', type=bool, default=False)  # Always False / True : self_supervised_data_classification wtih RAW LABEL
    #few shot parameter
    parser.add_argument('--k_shot', type=int, default=1)  # few shot learning

    args = parser.parse_args()
    list_datasets = ['isolet']
    list_methods = ['attention']  # attention xgb
    classifier_lists = ['lgb']
    noise_type_lst = ['mask']
    noise_list = [0.0]
    if args.problem == 'Classification': list_samples = [1]
    if args.problem == 'Regression': list_samples = [1]

    for args.select_sample in list_samples:
        for args.data in list_datasets:
            for args.noise_amount in noise_list:
                for args.method in list_methods:
                    for args.noise_type in noise_type_lst:
                        for args.classifier in classifier_lists:
                                args.path_result = './train/{}_{}_{}/{}/result_data_{}/{}/{}/'.format(args.data, args.classifier,args.select_sample, args.method, args.hidden_dim, args.noise_type,
                                                                                           args.noise_amount)
                                if not os.path.exists(args.path_result):
                                    os.makedirs(args.path_result)
                                args.path_train = './train/{}_{}_{}/{}/train_data_{}/{}/{}/'.format(args.data, args.classifier,args.select_sample, args.method, args.hidden_dim, args.noise_type,
                                                                                           args.noise_amount)
                                if not os.path.exists(args.path_train):
                                    os.makedirs(args.path_train)
                                results = exp_main(args)


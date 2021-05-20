import torch
import pandas as pd
import os
import primary_networks
import sub_networks
from config import ConfigMain
from config import ConfigPrimary
from config import ConfigSubModel
import utils
import argparse


def run_task(config, config_primary, config_subs, models_df, iter_num, task_type, sub_nn):
    task_name, max_len, text_col,\
        model_name, models_path, embeddings_version, \
        embeddings_path, label_col, key_col, submodels_list = \
        utils.read_config_main(config)
    models_path += sub_nn + '//' + str(iter_num) + '//'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    # utils.check_input(model_name, task_type)
    train_df = utils.read_df(task_name, 'train')
    val_df = utils.read_df(task_name, 'val')
    test_df = utils.read_df(task_name, 'test')
    if task_type == 'train_primary':
        models_df = primary_networks.run_primary(task_name, model_name, train_df, val_df, test_df,
                                                 max_len, text_col, embeddings_version,
                                                 embeddings_path, config_primary, models_path, models_df,
                                                 label_col, key_col, iter_num)
        models_df.to_csv(models_path + 'models_df.csv')
    elif task_type == "train_sub":
        models_df = sub_networks.train_sub_models(task_name, model_name, train_df, val_df, test_df, max_len,
                                                  text_col, embeddings_version, embeddings_path,
                                                  config_subs, models_path, models_df, label_col,
                                                  key_col, submodels_list, sub_nn, iter_num)
        models_df.to_csv(models_path + 'models_df.csv')


def create_models_df():
    models_dict = {'primary': {}, 'over_under': {}, 'model_12': {},
                   'model_13': {}, 'model_14': {}, 'model_15': {},
                   'model_23': {}, 'model_24': {}, 'model_25': {},
                   'model_34': {}, 'model_35': {},'model_45': {}}
    columns_df = ['hid_dim_lstm', 'dropout', 'lin_output_dim', 'lr',
                  'epochs_num', 'batch_size', 'momentum', 'accuracy']
    for model_name in models_dict.keys():
        models_dict[model_name] = ['' for col_num in range(len(columns_df))]
    models_df = pd.DataFrame.from_dict(models_dict, orient='index')
    models_df.columns = columns_df
    return models_df


def main(iter_num, task_type, sub_nn):
    config = ConfigMain()
    config_primary = ConfigPrimary()
    config_subs = ConfigSubModel()
    models_df = create_models_df()
    run_task(config, config_primary, config_subs, models_df, iter_num, task_type, sub_nn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", type=int, default=1)
    parser.add_argument("--task_type", type=str, default="train_primary")
    parser.add_argument("--sub_nn", type=str, default="regular")

    hp = parser.parse_args()
    main(hp.iter_num, hp.task_type, hp.sub_nn)

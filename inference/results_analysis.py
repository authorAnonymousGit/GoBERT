import numpy as np
import pandas as pd
from utils import create_prox_mat
import csv


def compute_accuracy(df, label_feature, pred_feature):
    return len(df[df[label_feature] == df[pred_feature]]) / len(df)


def compute_mae(df, label_feature, pred_feature):
    return (np.abs(df[label_feature] - df[pred_feature]).sum()) / len(df)


def compute_cem(df, label_feature, pred_feature):
    df_copy = df.copy()
    dist_dict = dict(df[label_feature].value_counts())
    denominator = len(df)
    prox_mat = create_prox_mat(dist_dict, denominator, inv=False)
    df_copy['pred_prox'] = df_copy.apply(lambda row:
                                         prox_mat[row[pred_feature] - 1][row[label_feature] - 1],
                                         axis=1)
    df_copy['truth_prox'] = df_copy.apply(lambda row:
                                          prox_mat[row[label_feature]-1][row[label_feature]-1],
                                          axis=1)
    return df_copy['pred_prox'].sum() / df_copy['truth_prox'].sum()


def evaluate_results(file_path, label_feature, prediction_feature_primary, by_epoch=None):
    df = pd.read_csv(file_path)
    if by_epoch is not None:
        # Filter the df
        df = df[df['epoch'] == by_epoch]
    acc = round(compute_accuracy(df, label_feature, prediction_feature_primary), 4)
    mae = round(compute_mae(df, label_feature, prediction_feature_primary), 4)
    cem = round(compute_cem(df, label_feature, prediction_feature_primary), 4)
    return acc, mae, cem


def compute_relative_improvement(acc_baseline, acc_to_compare,
                                 mae_baseline, mae_to_compare,
                                 cem_baseline, cem_to_compare):
    relative_improvement_acc = 100 * round((acc_to_compare - acc_baseline)/acc_baseline, 4)
    relative_improvement_mae = 100 * round((mae_baseline - mae_to_compare)/mae_baseline, 4)
    relative_improvement_cem = 100 * round((cem_to_compare - cem_baseline)/cem_baseline, 4)
    return relative_improvement_acc, relative_improvement_mae, relative_improvement_cem


if __name__ == '__main__':
    file_paths = ["../trained_models/sst-5/bert-large-uncased/regular/1/",
                  "../trained_models/sst-5/bert-large-uncased/regular/2/",
                  "../trained_models/sst-5/roberta-base/regular/2/",
                  "../trained_models/SemEval2017/bert-large-uncased/regular/1/",
                  "../trained_models/SemEval2017/bert-large-uncased/regular/2/",
                  "../trained_models/AmazonFashion/bert-large-uncased/regular/2/"]
    for file_path in file_paths:
        label_feature = "label"
        prediction_feature_primary = "prediction"
        files_suffixes = ['CrossEntropy', 'OrdinalTextClassification_alpha_0', 'OrdinalTextClassification_alpha_1',
                          'OrdinalTextClassification_alpha_2', 'OrdinalTextClassification_alpha_3',
                          'OrdinalTextClassification_alpha_4', 'OrdinalTextClassification_alpha_5',
                          'OrdinalTextClassification_alpha_6', 'OrdinalTextClassification_alpha_7',
                          'OrdinalTextClassification_alpha_8', 'OrdinalTextClassification_alpha_9',
                          'OrdinalTextClassification_alpha_10']
        updated_rows = [['val_acc', 'val_mae', 'val_cem', 'test_acc', 'test_mae', 'test_cem']]
        for file_suffix in files_suffixes:
            val_acc, val_mae, val_cem = evaluate_results(file_path + 'predictions_validation_primary_' +
                                                         file_suffix + '.csv', label_feature,
                                                         prediction_feature_primary)
            test_acc, test_mae, test_cem = evaluate_results(file_path + 'predictions_test_primary_' +
                                                            file_suffix + '.csv', label_feature,
                                                            prediction_feature_primary)
            updated_rows.extend([[val_acc, val_mae, val_cem, test_acc, test_mae, test_cem]])

        with open(file_path + "updated_summary.csv", "w+", newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(updated_rows)

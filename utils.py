import pandas as pd
import torch
import random
import numpy as np
from process_data import TextDataReader
from torch.utils.data.dataloader import DataLoader
import datetime
import time
from ast import literal_eval


def find_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_config_main(config):
    return config.TASK_NAME, config.MAX_LEN, \
           config.TEXT_FEATURE, config.MODEL_NAME, config.MODELS_PATH, \
           config.EMBEDDINGS_VERSION, config.EMBEDDINGS_PATH, \
           config.LABEL_FEATURE, config.KEY_FEATURE, config.SUB_MODELS


def read_config_networks(config, task_type):
    loss_type = config.LOSS_TYPE
    if task_type == 'train_primary':
        alpha = config.ALPHA
        beta = config.BETA
        model_selection_procedure = config.MODEL_SELECTION_PROCEDURE
    else:
        alpha = None
        beta = None
        model_selection_procedure = None
    labels_num = config.LABELS_NUM
    epochs_num = config.EPOCHS_NUM
    if "train" in task_type:
        lr = random.choice(np.linspace(config.LEARNING_RATE[0],
                                       config.LEARNING_RATE[1],
                                       config.LEARNING_RATE[2]))

        batch_size = int(random.choice(np.linspace(config.BATCH_SIZE[0],
                                                   config.BATCH_SIZE[1],
                                                   config.BATCH_SIZE[2])))
    # elif "test" in task_type:
    #     lr = config.LEARNING_RATE_val
    #     batch_size = config.BATCH_SIZE_val
    else:
        raise TypeError("The task " + task_type + " is not defined")
    return loss_type, alpha, beta, labels_num, epochs_num, lr, batch_size, model_selection_procedure


def read_config_main_for_inference(config):
    return config.TASK_NAME, config.EMBEDDINGS_VERSION, config.MODELS_PATH, config.SUB_MODELS


def read_config_graph_classification(config):
    return config.LABELS_NUM, config.SINGLE_GRAPH_STRUCTURE, config.EDGE_CREATION_PROCEDURE, \
           config.FINAL_PREDICTION_PROCEDURE, config.MODEL_DATA_PROCEDURE,\
           config.LOSS_TYPE, config.ALPHA, config.HIDDEN_CHANNELS, \
           config.BATCH_SIZE, config.NODE_FEATURES_NUM, \
           config.EPOCHS_NUM, config.BIDIRECTIONAL, \
           config.PRIMARY_MODEL_NAME_BASELINE, \
           config.PRIMARY_MODEL_NAME_GNN, \
           config.MODEL_SELECTION_PROCEDURE, \
           config.PRIMARY_LOSS_TYPE


def read_df(task_name, file_type):
    df = pd.read_csv(".//data//" + task_name + "//" + task_name + "_" + file_type + ".csv")
    df.drop([col for col in df.columns if "Unnamed" in col], axis=1, inplace=True)
    return df


def create_dataloaders_train(train_df, val_df, text_feature,
                             embeddings_version, max_len, batch_size,
                             label_col, key_col, sub_nn=None, nn_type='primary'):
    train_datareader = TextDataReader(train_df, embeddings_version, max_len,
                                      text_feature, label_col, key_col, sub_nn, nn_type)
    val_datareader = TextDataReader(val_df, embeddings_version, max_len,
                                    text_feature, label_col, key_col, sub_nn, nn_type)
    train_dataloader = DataLoader(train_datareader, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_datareader, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def create_dataloaders_test(df, embeddings_version, max_len, text_feature, label_col, key_col, batch_size,
                            sub_nn=None, nn_type='primary'):
    test_datareader = TextDataReader(df, embeddings_version, max_len, text_feature,
                                     label_col, key_col, sub_nn, nn_type)
    return DataLoader(test_datareader, batch_size=batch_size, shuffle=True)


def write_to_file(model_name, text):
    print(text)
    fout = open(str(model_name) + ".txt", "a")
    fout.write(text + '\n')
    fout.close()
    return


def write_results(labels, predictions, file_name):
    fout = open(file_name + ".txt", "a")
    for i in range(len(labels)):
        curr_str = str(i) + ":  "
        curr_str += "label: " + str(labels[i])
        curr_str += " , prediction: " + str(predictions[i])
        curr_str += "\n"
        fout.write(curr_str)
    fout.close()


def print_summary(models_path, model_name, target_metric_list):
    write_to_file(models_path + model_name, "Target Metric Dev:")
    write_to_file(models_path + model_name, str(target_metric_list))
    write_to_file(models_path + model_name, "")
    write_to_file(models_path + model_name, "Training complete!")
    write_to_file(models_path + model_name, "-------------------------------------------------------------")
    write_to_file(models_path + model_name, "")


def print_train_epoch_end(t0):
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Calculate val accuracy")
    print("")


def print_test_results(models_path, model_name, target_metric_val):
    write_to_file(models_path + model_name, "Target Metric Value Test:")
    write_to_file(models_path + model_name, str(target_metric_val))
    write_to_file(models_path + model_name, "-------------------------------------------------------------")
    write_to_file(models_path + model_name, "")


def print_train_epoch(epoch, accuracy, train_len, loss_scalar, total_time, loss_dist_sum=None):
    if loss_dist_sum:
        print("Flags train accuracy for epoch " + str(epoch + 1) + " is: %.3f" % float(accuracy))
        print("predictions distribution loss for epoch " + str(epoch + 1) + " is: %.3f" % float(loss_dist_sum))
    else:
        print("train accuracy for epoch " + str(epoch + 1) + " is: %.3f" % float(accuracy))
    print("loss after epoch", epoch + 1, "is: %.3f" % float(loss_scalar))
    print("total time: %.3f" % total_time)
    print()


def print_validation_epoch(acc_num, val_len, loss_scalar, total_time, loss_dist_sum=None):
    if loss_dist_sum:
        print("Flags validation accuracy for this epoch: %.3f" % float(acc_num))
        print("Predictions distribution loss for epoch: %.3f" % float(loss_dist_sum))
    else:
        print("Dev accuracy for this epoch: %.3f" % float(acc_num))
    print("Loss for this epoch %.3f" % float(loss_scalar))
    print("Total time: %.3f" % total_time)
    print()
    print()
    return


def print_epochs_progress(epoch, epochs_num):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs_num))
    print('Training...')


def print_batches_progress(t0, batch_idx, train_dataloader):
    elapsed = format_time(time.time() - t0)
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(batch_idx,
                                                                len(train_dataloader), elapsed))


def save_model(models_df, model_name, labels_num, loss_type, lr,
               epochs_num, batch_size, best_dev_acc):
    models_df.loc[models_df.index == model_name, 'labels_num'] = labels_num
    models_df.loc[models_df.index == model_name, 'loss_function'] = loss_type
    models_df.loc[models_df.index == model_name, 'lr'] = lr
    models_df.loc[models_df.index == model_name, 'epochs_num'] = epochs_num
    models_df.loc[models_df.index == model_name, 'batch_size'] = batch_size
    models_df.loc[models_df.index == model_name, 'accuracy'] = round(best_dev_acc, 3)


def convert_to_torch(inputs_ids, masks, labels):
    return torch.tensor(inputs_ids), torch.tensor(masks), torch.tensor(labels)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def update_predictions_dict(model_name, predictions_dict, key_index, labels, probabilities, predictions=None,
                            labels_dist=None, flags_preds=None, flags=None, sub_nn=None):
    probabilities = list(map(lambda y: list(map(lambda x: round(x, 4), y)), probabilities.tolist()))

    if sub_nn:  # This is a sub-network
        rel_label1 = int(model_name[-2])
        rel_label2 = int(model_name[-1])
        if sub_nn == 'unknown':
            labels = list(map(lambda label:
                              rel_label1 if label == 0
                              else (rel_label2 if label == 1 else "Unknown"),
                              labels.tolist()))
            predictions = list(map(lambda prediction:
                                   rel_label1 if prediction == 0
                                   else (rel_label2 if prediction == 1 else "Unknown"),
                                   predictions.tolist()))

        elif sub_nn == 'with_flags':
            labels = list(map(lambda label: label + 1, labels.tolist()))
            labels_dist = list(map(lambda y: list(map(lambda x: round(x, 4), y)), labels_dist.tolist()))

        else:  # sub_nn == 'regular'
            labels = list(map(lambda label: rel_label1 if label == 0 else rel_label2, labels.tolist()))
            predictions = list(map(lambda prediction:
                                   rel_label1 if prediction == 0
                                   else rel_label2, predictions.tolist()))

    else:  # This is the primary network
        labels = list(map(lambda label: label + 1, labels.tolist()))
        predictions = list(map(lambda prediction: prediction + 1, predictions.tolist()))

    if predictions:  # Valid for the primary network and sub_nn of "regular" or "unknown"
        tmp_dict = dict(zip(key_index.tolist(),
                            zip(predictions,
                                probabilities,
                                labels)))
    else:  # Valid for the sub_nn of "with_flags"
        tmp_dict = dict(zip(key_index.tolist(),
                            zip(probabilities,
                                labels,
                                labels_dist,
                                flags_preds.tolist(),
                                flags.tolist())))
    return {**predictions_dict, **tmp_dict}


def save_predictions_to_df(predictions_dict, models_path, dataset_type, model_name, sub_nn=None):
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    df.reset_index(inplace=True)
    if sub_nn != 'with_flags':
        df.columns = ['key_index', 'prediction', 'probability', 'label']
    else:
        df.columns = ['key_index', 'probability', 'label', 'label_dist', 'flag_pred', 'flag']
    df.to_csv(models_path + 'predictions_' + dataset_type + '_' + model_name + '.csv')


def print_new_sub_model(sub_model):
    print("**************************************************************")
    print(sub_model)
    print("**************************************************************")
    print("")


def OLCLoss(input_dist, target, factor_mat, device):
    r""" Ordered Labels Classification Loss:
    (Sum_{i=1}{labels_num} (input[i]*((i-J)/2 + 1))) - 1
    """
    # print("input_dist ", input_dist)
    to_mult = torch.tensor([factor_mat[elem.item()] for elem in target]).to(device)
    loss_by_item = (input_dist.to(device) * to_mult).sum(axis=1) - 1
    res = torch.mean(loss_by_item)
    # res.requires_grad = True
    return res


def adjust_probabilities(model_name, probs, labels_num=5, sub_nn='regular'):
    if model_name == 'primary':
        return np.array(literal_eval(probs))
    else:
        probs = literal_eval(probs)
        if sub_nn == 'unknown':
            # Ignore this sub nn if it is not confident enough of its prediction
            if np.argmax(probs) == 2:
                return None
            else:
                # Remove the unknown element from the probs vector
                probs = probs[:2]
        rel_label1 = int(model_name[-2]) - 1
        rel_label2 = int(model_name[-1]) - 1
        final_probs = np.zeros(labels_num)
        final_probs[[rel_label1, rel_label2]] = probs
        return final_probs / final_probs.sum()


def create_input_dict(files_path, run_type, sub_models, labels_num, sub_nn, filter_by_origin_pred=None,
                      model_name="primary"):
    primary_df = pd.read_csv(files_path + "predictions_" + run_type + "_" + model_name + ".csv")
    if filter_by_origin_pred:
        primary_df = primary_df[primary_df['prediction'] == filter_by_origin_pred]
        # if run_type == 'train' or run_type == 'validation':
        #     primary_df = primary_df[(primary_df['prediction'] == filter_by_origin_pred) |
        #                             (primary_df['label'] == filter_by_origin_pred)]
        # else:  # run_type = 'validation' or run_type = 'test'
        #     primary_df = primary_df[primary_df['prediction'] == filter_by_origin_pred]
    final_predictions = primary_df.copy()
    final_predictions.drop([col for col in final_predictions.columns
                            if "Unnamed" in col], axis=1, inplace=True)
    primary_df = primary_df[['key_index', 'probability']]
    primary_df.columns = ['key_index', 'primary']
    primary_df['primary'] = primary_df['primary'].apply(lambda probs: adjust_probabilities('primary',
                                                                                           probs,
                                                                                           labels_num))
    df_final = primary_df.copy()
    sub_models_path = files_path.rsplit("//", 2)[0] + '//'
    for sub_model in sub_models:
        file_name = "predictions_" + run_type + "_" + sub_model + ".csv"
        tmp_df = pd.read_csv(sub_models_path + file_name)
        tmp_df = tmp_df[['key_index', 'probability']]
        tmp_df.columns = ['key_index', sub_model]
        tmp_df[sub_model] = tmp_df[sub_model].apply(lambda probs: adjust_probabilities(sub_model,
                                                                                       probs,
                                                                                       labels_num,
                                                                                       sub_nn))
        df_final = df_final.merge(tmp_df, how='left', on='key_index')
    final_dict = df_final.set_index('key_index').to_dict(orient='index')
    return final_dict, final_predictions


def CrossEntropyLoss(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = outputs[range(batch_size), targets]
    del targets, batch_size
    torch.cuda.empty_cache()
    return -torch.sum(outputs)/num_examples


def CrossEntropyProxLoss(outputs, targets, inv_prox_mat, norm=False):
    # Method B- CE * 1/prox ; Method C- prox as targets
    batch_size = outputs.shape[0]
    ce_outputs = outputs[range(batch_size), targets]
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    preds = torch.argmax(outputs, axis=1)
    inv_prox_items = inv_rel_prox[range(batch_size), preds]
    if norm:
        prox_items = 1 / inv_rel_prox[range(batch_size), targets]
    else:
        prox_items = 1
    prox_mul = prox_items * inv_prox_items
    del batch_size, inv_rel_prox, preds
    loss = -torch.sum(ce_outputs * prox_mul) / batch_size  # Update- prox_mul instead of inv_prox_items
    del outputs, targets, inv_prox_mat
    torch.cuda.empty_cache()
    return loss


# def calc_expected_prox(target, pred, dist_dict, denominator, inv=True):
#     int_pred = int(pred)
#     residue = pred - int_pred
#     minlabel, maxlabel = min(int_pred, target), max(int_pred, target)
#     numerator = dist_dict[int_pred] / 2
#     if minlabel == int_pred:
#         for tmp_label in range(minlabel + 1, maxlabel):
#             numerator += dist_dict[tmp_label]
#         numerator += dist_dict[maxlabel] * (1 - residue)
#     else:
#         for tmp_label in range(maxlabel - 1, minlabel, -1):
#             numerator += dist_dict[tmp_label]
#         numerator += dist_dict[minlabel] * (1 - residue)
#     if inv:
#         return denominator / numerator
#     else:
#         return numerator / denominator

def ExpectationProxLoss(outputs, targets, inv_prox_mat):
    # outputs = 1 / outputs
    outputs = outputs**2
    num_examples = targets.shape[0]
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    loss_val = torch.sum(torch.sum(inv_rel_prox * outputs, dim=1)) / num_examples
    # prox_rel_items = 1 / inv_rel_prox[range(num_examples), targets]
    # loss_val = -torch.sum(tmp_loss_val * prox_rel_items) / num_examples
    # loss_val = torch.sum(tmp_loss_val * prox_rel_items) / num_examples
    # del outputs, targets, inv_prox_mat, prox_rel_items, tmp_loss_val
    del outputs, targets, inv_prox_mat
    torch.cuda.empty_cache()
    return loss_val


def CEProxMSELoss(outputs, targets, inv_prox_mat, se_tensor, delta=0.0, phi=25.0):
    outputs = 1 / outputs
    se_tensor = (se_tensor + delta) / phi
    num_examples = targets.shape[0]
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    se_tensor_rel =  se_tensor[:, targets].t()
    loss_val = -torch.sum(torch.sum(inv_rel_prox * outputs * se_tensor_rel, dim=1)) / num_examples
    del outputs, targets, inv_prox_mat
    torch.cuda.empty_cache()
    return loss_val


def getProxLoss(outputs, targets, inv_prox_mat):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    preds = torch.argmax(outputs, axis=1)
    inv_prox_items = rel_prox[range(batch_size), preds]
    del outputs, targets, inv_prox_mat, batch_size, rel_prox, preds
    torch.cuda.empty_cache()
    return torch.sum(inv_prox_items)/num_examples


def MseLoss(outputs, targets, labels_num, device):
    num_examples = targets.shape[0]
    labels = torch.tensor(range(1, labels_num + 1)).to(device)
    expected_preds = torch.sum(outputs * labels, dim=1) - 1
    mse_by_expected = (expected_preds - targets) ** 2
    del outputs, targets, labels, expected_preds
    torch.cuda.empty_cache()
    return torch.sum(mse_by_expected)/num_examples


def MseProxLoss(outputs, targets, inv_prox_mat, labels_num, device):
    labels = torch.tensor(range(1, labels_num + 1)).to(device)
    expected_preds = torch.sum(outputs * labels, dim=1) - 1
    mse_by_expected = (expected_preds - targets) ** 2

    preds_by_expected = [round(label.item()) for label in expected_preds]
    inv_prox_items = inv_prox_mat[preds_by_expected, targets]

    del outputs, targets, inv_prox_mat, labels, expected_preds, preds_by_expected
    torch.cuda.empty_cache()
    return torch.mean(inv_prox_items * mse_by_expected)


def ExpectedMSE(outputs, targets, se_tensor, inv_prox_mat=None):
    # Returns \frac{1}{batchsize}\sum_{i=1}^{batchsize}\sum_{y=1}^{L} p_y \cdot \left (  y-l_i\right )^2
    num_examples = targets.shape[0]
    outputs = 1 / outputs
    if inv_prox_mat is None:
        # This is the case of loss function A
        loss_val = -torch.sum(se_tensor[targets, :] * outputs) / num_examples
    else:
        # This is the case of loss function B
        tmp_loss_val = torch.sum(se_tensor[targets, :] * outputs, dim=1)
        # TODO: Maybe we will change back to argmax
        preds = torch.argmin(outputs, axis=1)
        inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
        inv_prox_items = inv_rel_prox[range(num_examples), preds]
        prox_items = 1 / inv_rel_prox[range(num_examples), targets]
        prox_mul = prox_items * inv_prox_items
        loss_val = -torch.sum(tmp_loss_val * prox_mul) / num_examples
    del outputs, targets, inv_prox_mat
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss(outputs, targets, prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    rel_prox = torch.tensor(prox_mat[:, targets]).t()
    numerator = torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * rel_prox, dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, prox_mat, numerator, denominator, \
        one_hot_target_comp, one_hot_target, rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss2(outputs, targets, inv_prox_mat, prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    rel_inv_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    rel_prox = torch.tensor(prox_mat[:, targets]).t()
    numerator = torch.sum((one_hot_target * outputs) * (rel_prox**2), dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * rel_inv_prox, dim=1)
    loss_val = torch.sum(numerator / denominator)
    del outputs, targets, prox_mat
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss3(outputs, targets, inv_prox_mat, prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    rel_inv_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    rel_prox = torch.tensor(prox_mat[:, targets]).t()
    numerator = torch.sum((one_hot_target * outputs) * (rel_prox**3), dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * rel_inv_prox, dim=1)
    loss_val = torch.sum(numerator / denominator)
    del outputs, targets, prox_mat
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss4(outputs, targets, inv_prox_mat, factor, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    numerator = factor * torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * inv_rel_prox, dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, inv_rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss5(outputs, targets, prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    rel_prox = torch.tensor(prox_mat[:, targets]).t()
    numerator = torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * torch.sqrt(rel_prox), dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, prox_mat, numerator, denominator, \
        one_hot_target_comp, one_hot_target, rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss6(outputs, targets, inv_prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    numerator = torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * torch.sqrt(inv_rel_prox), dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, inv_rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss7(outputs, targets, prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    rel_prox = torch.tensor(prox_mat[:, targets]).t()
    rel_prox = rel_prox**2
    numerator = torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * rel_prox, dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, prox_mat, numerator, denominator, \
        one_hot_target_comp, one_hot_target, rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss8(outputs, targets, inv_prox_mat, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    numerator = torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * torch.pow(inv_rel_prox, 1/3), dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, inv_rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss9(outputs, targets, inv_prox_mat, factor, root, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    numerator = factor * torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * torch.pow(inv_rel_prox, 1/root), dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, inv_rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss10(outputs, targets, norm_inv_prox_mat, factor, root, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(norm_inv_prox_mat[:, targets]).t()
    numerator = factor * torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * torch.pow(inv_rel_prox, 1/root), dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, inv_rel_prox
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss11(outputs, targets, se_tensor, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    se_tensor_rel = se_tensor[:, targets].t()
    numerator = torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * se_tensor_rel, dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, se_tensor
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss12(outputs, targets, inv_prox_mat, se_tensor, factor, root, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    se_tensor_rel = se_tensor[:, targets].t()
    numerator = factor * torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) *
                            torch.pow(inv_rel_prox, 1/root) *
                            se_tensor_rel, dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, se_tensor
    torch.cuda.empty_cache()
    return loss_val


def CrossProxLoss13(outputs, targets, inv_prox_mat, factor, delta, device):
    num_examples = targets.shape[0]
    labels_num = outputs.shape[1]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    normalized_inv_rel_prox = inv_rel_prox / torch.min(inv_rel_prox).item() + delta
    numerator = factor * torch.sum(one_hot_target * outputs, dim=1)
    denominator = torch.sum((one_hot_target_comp * outputs) * torch.log(normalized_inv_rel_prox), dim=1)
    loss_val = torch.sum(numerator / denominator) / num_examples
    del outputs, targets, numerator, denominator, \
        one_hot_target_comp, one_hot_target, inv_rel_prox, \
        normalized_inv_rel_prox
    torch.cuda.empty_cache()
    return loss_val


def PProx(outputs, targets, inv_prox_mat):
    num_examples = targets.shape[0]
    outputs = outputs ** 2
    inv_rel_prox = torch.tensor(inv_prox_mat[:, targets]).t()
    loss_val = torch.sum(torch.sum(inv_rel_prox * outputs, dim=1)) / num_examples
    del outputs, targets, inv_prox_mat
    torch.cuda.empty_cache()
    return loss_val


def create_prox_mat(dist_dict, denominator, inv=True):
    labels = dist_dict.keys()
    prox_mat = np.zeros([len(labels), len(labels)])
    for label1 in labels:
        for label2 in labels:
            minlabel, maxlabel = min(label1, label2), max(label1, label2)
            numerator = dist_dict[label1] / 2
            if minlabel == label1:  # Above the diagonal
                for tmp_label in range(minlabel + 1, maxlabel + 1):
                    numerator += dist_dict[tmp_label]
            else:  # Under the diagonal
                for tmp_label in range(maxlabel - 1, minlabel - 1, -1):
                    numerator += dist_dict[tmp_label]
            if inv:
                prox_mat[label1 - 1][label2 - 1] = (-np.log(numerator / denominator))**-1
            else:
                prox_mat[label1 - 1][label2 - 1] = -np.log(numerator / denominator)
    return torch.tensor(prox_mat)


def get_prox_params(models_path, dataset_type, model_name):
    df_primary = pd.read_csv(models_path + "predictions_" + dataset_type + "_" + model_name + ".csv")
    dist_dict = dict(df_primary['label'].value_counts())
    denominator = len(df_primary)
    return dist_dict, denominator


def get_loss(loss_type, lsm, softmax_values, ground_truth, device,
             alpha, beta, inv_prox_mat, norm_inv_prox_mat,
             labels_num, dist_dict, se_tensor):
    if loss_type == 'CrossEntropy':
        loss_val = CrossEntropyLoss(lsm, ground_truth.to(device))
    else:  # Ordinal loss
        # loss1, loss2, loss3 = 0.0, 0.0, 0.0
        if loss_type == 'OrdinalTextClassification-A':
            loss1 = CrossEntropyLoss(lsm, ground_truth.to(device))
            # loss2 = MseLoss(softmax_values, ground_truth.to(device), labels_num, device)
            loss2 = ExpectedMSE(lsm, ground_truth.to(device), se_tensor.to(device))
            loss3 = getProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        # elif loss_type == 'OrdinalTextClassification-B':
        #     loss1 = CrossEntropyProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        #     # loss2 = MseProxLoss(softmax_values, ground_truth.to(device),
        #     #                     inv_prox_mat, labels_num, device)
        #     loss2 = ExpectedMSE(lsm, ground_truth.to(device),
        #                         se_tensor.to(device), inv_prox_mat)
        # elif loss_type == 'OrdinalTextClassification-C':
        #     loss1 = CrossEntropyProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        #     loss2 = ExpectationProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        # elif loss_type == 'OrdinalTextClassification-D':
        #     loss1 = CrossEntropyLoss(lsm, ground_truth.to(device))
        #     loss2 = ExpectationProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        # elif loss_type == 'OrdinalTextClassification-E':
        #     loss1 = CrossEntropyLoss(lsm, ground_truth.to(device))
        #     loss2 = ExpectationProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        #     loss_val = loss1 * loss2
        #     del loss1,
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-F':
        #     # loss1 = CrossEntropyLoss(lsm, ground_truth.to(device))
        #     # loss2 = getProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        #     # loss_val = loss1 * loss2
        #     # return loss_val
        #     return CrossEntropyProxLoss(lsm, ground_truth.to(device), inv_prox_mat, norm=True)
        # elif loss_type == 'OrdinalTextClassification-G':
        #     return CrossEntropyProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        # elif loss_type == 'OrdinalTextClassification-H':
        #     return ExpectationProxLoss(softmax_values, ground_truth.to(device), inv_prox_mat)
        # elif loss_type == 'OrdinalTextClassification-I':
        #     loss1 = CrossEntropyLoss(lsm, ground_truth.to(device))
        #     loss2 = CrossEntropyProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        #     loss_val = alpha * loss1 + (1 - alpha) * loss2
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-J':
        #     loss1 = ExpectedMSE(lsm, ground_truth.to(device), se_tensor.to(device))
        #     loss2 = CrossEntropyProxLoss(lsm, ground_truth.to(device), inv_prox_mat)
        #     loss_val = alpha * loss1 + (1 - alpha) * loss2
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-K':
        #     loss_val = CrossProxLoss(lsm, ground_truth.to(device), 1 / inv_prox_mat, device)
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-L':
        #     loss_val = CrossProxLoss2(lsm, ground_truth.to(device), inv_prox_mat, 1 / inv_prox_mat, device)
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-M':
        #     loss_val = CrossProxLoss3(lsm, ground_truth.to(device), inv_prox_mat, 1 / inv_prox_mat, device)
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-N':
        #     loss_val = CrossProxLoss4(lsm, ground_truth.to(device), inv_prox_mat, 25, device)
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-O':
        #     loss_val = CrossProxLoss5(lsm, ground_truth.to(device), 1 / inv_prox_mat, device)
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-P':
        #     loss_val = CrossProxLoss6(lsm, ground_truth.to(device), inv_prox_mat, device)
        #     return loss_val
        # elif loss_type == 'OrdinalTextClassification-Q':
        #     loss_val = CrossProxLoss7(lsm, ground_truth.to(device), 1 / inv_prox_mat, device)
        #     return loss_val
        elif loss_type == 'OrdinalTextClassification-R':
            loss_val = CrossProxLoss8(lsm, ground_truth.to(device), 1 / inv_prox_mat, device)
            return loss_val
        elif loss_type == 'OrdinalTextClassification-S':
            loss_val = CrossProxLoss9(lsm, ground_truth.to(device), inv_prox_mat, 1, 4, device)
            return loss_val
        elif loss_type == 'OrdinalTextClassification-T':
            loss_val = CrossProxLoss9(lsm, ground_truth.to(device), inv_prox_mat, labels_num**2, 4, device)
            return loss_val
        elif loss_type == 'OrdinalTextClassification-U':
            loss_val = CrossProxLoss11(lsm, ground_truth.to(device), se_tensor.to(device), device)
            return loss_val
        elif loss_type == 'OrdinalTextClassification-V':
            return CrossProxLoss12(lsm, ground_truth.to(device), inv_prox_mat,
                                   se_tensor.to(device), labels_num**2, 4, device)
        # elif loss_type == 'OrdinalTextClassification-W':
        #     return CrossProxLoss13(lsm, ground_truth.to(device), inv_prox_mat, 1, 0.1, device)
        # elif loss_type == 'OrdinalTextClassification-X':
        #     return CrossProxLoss13(lsm, ground_truth.to(device), inv_prox_mat, 1, 1, device)
        # elif loss_type == 'OrdinalTextClassification-Y':
        #     return CrossProxLoss13(lsm, ground_truth.to(device), inv_prox_mat, labels_num**2, 0.1, device)
        # elif loss_type == 'OrdinalTextClassification-Z':
        #     return CrossProxLoss13(lsm, ground_truth.to(device), inv_prox_mat, labels_num**2, 1, device)

        elif loss_type == 'OrdinalTextClassification-W':
            return CrossProxLoss9(lsm, ground_truth.to(device), inv_prox_mat, labels_num, 4, device)
        elif loss_type == 'OrdinalTextClassification-X':
            return CrossProxLoss9(lsm, ground_truth.to(device), inv_prox_mat, labels_num, 5, device)
        elif loss_type == 'OrdinalTextClassification-Y':
            return CrossProxLoss9(lsm, ground_truth.to(device), inv_prox_mat, labels_num**2, 4, device)
        elif loss_type == 'OrdinalTextClassification-Z':
            return CrossProxLoss9(lsm, ground_truth.to(device), inv_prox_mat, labels_num**2, 5, device)
        elif loss_type == 'OrdinalTextClassification-B':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num, 2, device)
        elif loss_type == 'OrdinalTextClassification-C':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num, 4, device)
        elif loss_type == 'OrdinalTextClassification-D':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num**2, 2, device)
        elif loss_type == 'OrdinalTextClassification-E':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num**2, 4, device)
        elif loss_type == 'OrdinalTextClassification-F':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num**2, labels_num - 1, device)
        elif loss_type == 'OrdinalTextClassification-G':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num**2, labels_num - 2, device)
        elif loss_type == 'OrdinalTextClassification-H':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num, labels_num + 2, device)
        elif loss_type == 'OrdinalTextClassification-I':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num, labels_num + 1, device)
        elif loss_type == 'OrdinalTextClassification-J':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num, labels_num, device)
        elif loss_type == 'OrdinalTextClassification-K':
            return CrossProxLoss10(lsm, ground_truth.to(device), norm_inv_prox_mat, labels_num**2, labels_num + 2, device)
        '''
        if beta is not None:
            loss_val = alpha * loss1 + beta * loss2 + (1 - alpha - beta) * loss3
        else:
            loss_val = alpha * loss1 + (1 - alpha) * loss2
        del loss1, loss2
        '''
    return loss_val




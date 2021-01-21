import utils
import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from FCBERT_classifier import FCBERT_SUB
from FCBERT_classifier import FCBERT_PRIMARY
import numpy as np
import copy


def get_model(model_name, embeddings_path, sub_nn, loss_type):
    if "FCBERT" in model_name:
        if sub_nn == 'with_flags':
            return FCBERT_SUB(embeddings_path, labels_num=2)
        elif sub_nn == 'regular':
            return FCBERT_PRIMARY(embeddings_path, labels_num=2, loss_type=loss_type)
        else:
            # with unknown
            return FCBERT_PRIMARY(embeddings_path, labels_num=3, loss_type=loss_type)
    else:
        raise TypeError("The model " + model_name + " is not defined")


def ov_un_label(row):
    return int(row['prediction'] > row['label'])


def create_over_under(models_path, task_name):
    df_train_primary = pd.read_csv(models_path + 'predictions_train_primary.csv')
    df_val_primary = pd.read_csv(models_path + 'predictions_validation_primary.csv')
    df_train_ov_un = df_train_primary[df_train_primary['label'] != df_train_primary['prediction']]
    df_val_ov_un = df_val_primary[df_val_primary['label'] != df_val_primary['prediction']]
    df_train_ov_un['label'] = df_train_ov_un.apply(lambda row: ov_un_label(row), axis=1)
    df_val_ov_un['label'] = df_val_ov_un.apply(lambda row: ov_un_label(row), axis=1)
    df_train_ov_un = df_train_ov_un[['key_index', 'label']]
    df_val_ov_un = df_val_ov_un[['key_index', 'label']]
    df_train_ov_un.to_csv(task_name + '_sub_over_under_TRAIN.csv')
    df_val_ov_un.to_csv(task_name + '_sub_over_under_VAL.csv')


def fix_values_over_under(train_df_main, val_df_main, test_df_main, task_name, label_col):
    df_train_ov_un = pd.read_csv(task_name + '_sub_over_under_TRAIN.csv')
    df_val_ov_un = pd.read_csv(task_name + '_sub_over_under_VAL.csv')
    df_test_ov_un = pd.read_csv(task_name + '_sub_over_under_TEST.csv')
    train_over_under_elements = set(df_train_ov_un.key_index)
    val_over_under_elements = set(df_val_ov_un.key_index)
    test_over_under_elements = set(df_test_ov_un.key_index)
    train_df = train_df_main[train_df_main.key_index.isin(train_over_under_elements)]
    val_df = val_df_main[val_df_main.key_index.isin(val_over_under_elements)]
    test_df = test_df_main[test_df_main.key_index.isin(test_over_under_elements)]
    dict_train = dict(zip(df_train_ov_un.key_index.tolist(), df_train_ov_un.label.tolist()))
    dict_val = dict(zip(df_val_ov_un.key_index.tolist(), df_val_ov_un.label.tolist()))
    dict_test = dict(zip(df_test_ov_un.key_index.tolist(), df_test_ov_un.label.tolist()))
    train_df[label_col] = train_df['key_index'].apply(lambda x: dict_train[x])
    val_df[label_col] = val_df['key_index'].apply(lambda x: dict_val[x])
    test_df[label_col] = test_df['key_index'].apply(lambda x: dict_test[x])
    return train_df, val_df, test_df


def create_flag(rel_label1, rel_label2, ground_truth_label):
    if rel_label1 <= ground_truth_label <= rel_label2:
        return 0
    else:
        return 1


def create_ground_truth_dist(rel_label1, rel_label2, ground_truth_label):
    if rel_label1 <= ground_truth_label <= rel_label2:
        dist1 = abs(ground_truth_label - rel_label1)
        dist2 = abs(ground_truth_label - rel_label2)
        distances_sum = dist1 + dist2
        val1 = 1 - (dist1 / distances_sum)
        val2 = 1 - (dist2 / distances_sum)
        return [val1, val2]
    else:
        if ground_truth_label < rel_label1:
            return [1, 0]
        else:
            return [0, 1]


def create_sub_df(df_primary, rel_label1, rel_label2, text_col, label_col, sub_nn):
    df_sub = df_primary.copy()
    df_sub = df_sub[['key_index', text_col, label_col]]
    if sub_nn == 'regular':
        df_sub = df_sub[(df_sub[label_col] == rel_label1) | (df_sub[label_col] == rel_label2)]
    elif sub_nn == 'with_flags':
        df_sub['labels_dist'] = df_sub[label_col].apply(lambda label: create_ground_truth_dist(rel_label1,
                                                                                               rel_label2,
                                                                                               label))
        df_sub['flag'] = df_sub[label_col].apply(lambda label: create_flag(rel_label1,
                                                                           rel_label2,
                                                                           label))
    else:
        # with unknown
        df_sub[label_col] = df_sub[label_col].apply(lambda label:
                                                    0 if label == rel_label1 else
                                                    (1 if label == rel_label2 else 2))
    return df_sub


def create_sub_models_dfs(sub_model, train_df, val_df, test_df, text_col, label_col, sub_nn):
    rel_label1 = int(sub_model[-2])
    rel_label2 = int(sub_model[-1])
    df_sub_train = create_sub_df(train_df, rel_label1, rel_label2, text_col, label_col, sub_nn)
    df_sub_val = create_sub_df(val_df, rel_label1, rel_label2, text_col, label_col, sub_nn)
    if sub_nn == 'regular':
        df_sub_test = test_df.copy()
    else:
        df_sub_test = create_sub_df(test_df, rel_label1, rel_label2, text_col, label_col, sub_nn)
    # df_sub_train.to_csv(task_name + '_sub_' + sub_model + '_TRAIN.csv')
    # df_sub_val.to_csv(task_name + '_sub_' + sub_model + '_VAL.csv')
    return df_sub_train, df_sub_val, df_sub_test


def train_sub_models(task_name, model_name, train_df, val_df, test_df, max_len,
                     text_feature, embeddings_version, embeddings_path, config_subs,
                     models_path, models_df, label_col, key_col, submodels_list, sub_nn):
    device = utils.find_device()
    for sub_model in submodels_list:
        loss_type, _, labels_num, epochs_num, lr, batch_size = \
            utils.read_config_networks(config_subs, "train_sub")

        if sub_model == 'over_under':
            create_over_under(models_path, task_name)
            train_df_sub, val_df_sub, test_df_sub = fix_values_over_under(train_df, val_df, test_df, task_name, label_col)
        else:
            train_df_sub, val_df_sub, test_df_sub = create_sub_models_dfs(sub_model, train_df, val_df, test_df,
                                                                          text_feature, label_col, sub_nn)
            # train_df, val_df = get_train_validation(sub_model, train_df_main, val_df_main,
            #                                         label_col, key_col, task_name)

        train_dataloader, val_dataloader = utils.create_dataloaders_train(train_df_sub, val_df_sub,
                                                                          text_feature,
                                                                          embeddings_version,
                                                                          max_len, batch_size,
                                                                          label_col, key_col,
                                                                          sub_nn, sub_model)
        model = get_model(model_name, embeddings_path, sub_nn, loss_type)

        model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5,  eps=1e-8)
        train_len = len(train_df_sub)
        val_len = len(val_df_sub)
        train_accuracy_list, train_loss_list, val_dist_list_loss, loss_list_val = [], [], [], []
        val_acc_list = []
        best_val_dist_loss = np.inf
        best_val_acc = 0.0
        best_epoch = False
        best_model = None
        total_steps = len(train_dataloader) * epochs_num
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        utils.print_new_sub_model(sub_model)

        for epoch in range(epochs_num):
            utils.print_epochs_progress(epoch, epochs_num)
            start_train = time.process_time()
            acc_num_train = 0.0
            loss_scalar = 0.0
            loss_dist_sum = 0.0
            model.train()
            optimizer.zero_grad()
            t0 = time.time()
            predictions_dict_train = dict()
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx % 10 == 0 and not batch_idx == 0:
                    utils.print_batches_progress(t0, batch_idx, train_dataloader)
                input_ids = batch[0].to(device, dtype=torch.long)
                masks = batch[1].to(device, dtype=torch.long)
                labels = batch[2].to(device, dtype=torch.long)
                key_ids = batch[3].to(device, dtype=torch.long)
                model.zero_grad()
                if sub_nn == "with_flags":
                    labels_dist = batch[4].to(device, dtype=torch.float)
                    flags = batch[5].to(device, dtype=torch.long)
                    loss, predictions_dist, predictions_flag, loss_dist = model(input_ids,
                                                                                masks,
                                                                                labels_dist,
                                                                                flags)
                    loss_dist_sum += loss_dist.item()
                    acc_num_train += torch.sum(predictions_flag == flags)

                    predictions_dict_train = utils.update_predictions_dict(sub_model, predictions_dict_train,
                                                                           key_ids,
                                                                           labels,
                                                                           predictions_dist,
                                                                           labels_dist=labels_dist,
                                                                           flags_preds=predictions_flag,
                                                                           flags=flags,
                                                                           sub_nn=sub_nn)
                    del key_ids, input_ids, masks, labels, flags, labels_dist, \
                        predictions_dist, predictions_flag, loss_dist
                else:
                    loss, predictions, probabilities = model(input_ids, masks, labels)
                    predictions_dict_train = utils.update_predictions_dict(sub_model, predictions_dict_train,
                                                                           key_ids,
                                                                           labels,
                                                                           probabilities,
                                                                           predictions,
                                                                           sub_nn=sub_nn)
                    acc_num_train += torch.sum(predictions == labels)
                    del key_ids, input_ids, masks, labels, predictions, probabilities
                torch.cuda.empty_cache()
                loss_scalar += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            loss_scalar /= train_len
            loss_dist_sum /= train_len
            acc_num_train /= train_len
            end_train = time.process_time()
            total_time = end_train - start_train
            utils.print_train_epoch(epoch, acc_num_train, train_len, loss_scalar, total_time, loss_dist_sum)
            train_loss_list.append(round(float(loss_scalar), 3))
            train_accuracy_list.append(round(float(acc_num_train), 3))
            utils.print_train_epoch_end(t0)
            acc_num_val, val_dist_loss, val_loss, predictions_dict_val = evaluate_val(sub_model, model, val_dataloader,
                                                                                      device, val_len, sub_nn)
            val_dist_list_loss.append(val_dist_loss)
            val_acc_list.append(acc_num_val)
            loss_list_val.append(val_loss)
            if sub_nn == 'with_flags' and val_dist_loss < best_val_dist_loss:
                best_epoch = True
                best_val_dist_loss = val_dist_loss
            elif sub_nn != 'with_flags' and acc_num_val > best_val_acc:
                best_epoch = True
                best_val_acc = acc_num_val
            if best_epoch:
                torch.save(model.state_dict(), models_path + sub_model + '.pkl')
                best_model = copy.deepcopy(model)

                # The next two commands are used here in order to allow the GNN to utilize the
                # prediction of each sub model for every sample.If we want to use the previous version
                # These commands have to be ignored, and the following two (save_predictions_to_df) will return
                # to their old role (saving the predictions in a way that only the expertise region of each
                # sub model is taken into account.
                _, predictions_dict_train = evaluate_test(sub_model, train_df, model, device,
                                                          embeddings_version, max_len,
                                                          text_feature, batch_size,
                                                          label_col, key_col, sub_nn,
                                                          sub_model)
                _, predictions_dict_val = evaluate_test(sub_model, val_df, model, device,
                                                        embeddings_version, max_len,
                                                        text_feature, batch_size,
                                                        label_col, key_col, sub_nn,
                                                        sub_model)

                utils.save_predictions_to_df(predictions_dict_train, models_path, 'train', sub_model, sub_nn)
                utils.save_predictions_to_df(predictions_dict_val, models_path, 'validation', sub_model, sub_nn)
                best_epoch = False
        utils.save_model(models_df, sub_model, labels_num, loss_type, lr,
                         epochs_num, batch_size, best_val_dist_loss)
        if sub_nn != "with_flags":
            utils.print_summary(models_path, sub_model, val_acc_list)
        else:
            utils.print_summary(models_path, sub_model, val_dist_list_loss)
        test_rel_metric, predictions_dict_test = evaluate_test(sub_model, test_df_sub, best_model, device,
                                                               embeddings_version, max_len,
                                                               text_feature, batch_size,
                                                               label_col, key_col, sub_nn,
                                                               sub_model)
        utils.save_predictions_to_df(predictions_dict_test, models_path, 'test', sub_model, sub_nn)
        utils.print_test_results(models_path, model_name, test_rel_metric)
    return models_df


def evaluate_val(sub_model, model, val_dataloader, device, val_len, sub_nn):
    start_val = time.process_time()
    acc_num_val = 0.0
    loss_dist_sum = 0.0
    loss_scalar = 0.0
    model.eval()
    predictions_dict_val = dict()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            input_ids = batch[0].to(device, dtype=torch.long)
            masks = batch[1].to(device, dtype=torch.long)
            labels = batch[2].to(device, dtype=torch.long)
            key_ids = batch[3].to(device, dtype=torch.long)
            if sub_nn == "with_flags":
                labels_dist = batch[4].to(device, dtype=torch.float)
                flags = batch[5].to(device, dtype=torch.long)
                loss, predictions_dist, predictions_flag, loss_dist = model(input_ids,
                                                                            masks,
                                                                            labels_dist,
                                                                            flags)
                loss_dist_sum += loss_dist.item()
                acc_num_val += torch.sum(predictions_flag == flags)
                predictions_dict_val = utils.update_predictions_dict(sub_model, predictions_dict_val,
                                                                     key_ids,
                                                                     labels,
                                                                     predictions_dist,
                                                                     labels_dist=labels_dist,
                                                                     flags_preds=predictions_flag,
                                                                     flags=flags,
                                                                     sub_nn=sub_nn)
                del key_ids, input_ids, masks, labels, flags, labels_dist
            else:
                loss, predictions, probabilities = model(input_ids, masks, labels)
                predictions_dict_val = utils.update_predictions_dict(sub_model, predictions_dict_val,
                                                                     key_ids,
                                                                     labels,
                                                                     probabilities,
                                                                     predictions,
                                                                     sub_nn=sub_nn)
                acc_num_val += torch.sum(predictions == labels)
                del key_ids, input_ids, masks, labels
            torch.cuda.empty_cache()
            loss_scalar += loss.item()
    loss_dist_sum /= val_len
    acc_num_val /= val_len
    end_val = time.process_time()
    total_time = end_val - start_val
    utils.print_validation_epoch(acc_num_val, val_len, loss_scalar, total_time, loss_dist_sum)
    return acc_num_val.item(), float(loss_dist_sum), float(loss_scalar), predictions_dict_val


# def read_model_parameters(models_df, model_name):
#     hid_dim_lstm = int(models_df.loc[model_name].loc['hid_dim_lstm'])
#     dropout = models_df.loc[model_name].loc['dropout']
#     lin_output_dim = int(models_df.loc[model_name].loc['lin_output_dim'])
#     lr = models_df.loc[model_name].loc['lr']
#     batch_size = int(models_df.loc[model_name].loc['batch_size'])
#     momentum = models_df.loc[model_name].loc['momentum']
#     loss_function = nn.NLLLoss()  # conf_df.loc[model_name].loc['loss_function']
#     labels_num = int(models_df.loc[model_name].loc['labels_num'])
#     return hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, \
#            batch_size, momentum, labels_num
#
#
# def load_model_parameters(model_name, sub_model, models_path, embeddings_path, models_df, sub_nn):
#     device = utils.find_device()
#     hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, batch_size, momentum, labels_num = \
#         read_model_parameters(models_df, sub_model)
#     model = get_model(model_name, embeddings_path, sub_nn, loss_type)
#     state_dict = torch.load(models_path + '//' + sub_model + '.pkl', map_location=device)
#     model.load_state_dict(state_dict)
#     return model


def evaluate_test(sub_model, test_df, model, device, embeddings_version, max_len,
                  text_feature, batch_size, label_col, key_col,
                  sub_nn=None, nn_type='primary'):
    start_test = time.process_time()
    test_dataloader = utils.create_dataloaders_test(test_df, embeddings_version,
                                                    max_len, text_feature, label_col,
                                                    key_col, batch_size, sub_nn, nn_type)
    test_len = len(test_df)
    acc_num_test = 0.0
    loss_dist_sum = 0.0
    model.eval()
    predictions_dict_test = dict()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            input_ids = batch[0].to(device, dtype=torch.long)
            masks = batch[1].to(device, dtype=torch.long)
            labels = torch.zeros(len(batch[1])).to(device, dtype=torch.long)  # batch[2].to(device, dtype=torch.long)
            key_ids = batch[3].to(device, dtype=torch.long)
            if sub_nn == "with_flags":
                labels_dist = batch[4].to(device, dtype=torch.float)
                flags = batch[5].to(device, dtype=torch.long)
                loss, predictions_dist, predictions_flag, loss_dist = model(input_ids,
                                                                            masks,
                                                                            labels_dist,
                                                                            flags)
                loss_dist_sum += loss_dist.item()
                acc_num_test += torch.sum(predictions_flag == flags)
                predictions_dict_test = utils.update_predictions_dict(sub_model, predictions_dict_test,
                                                                      key_ids,
                                                                      labels,
                                                                      predictions_dist,
                                                                      labels_dist=labels_dist,
                                                                      flags_preds=predictions_flag,
                                                                      flags=flags,
                                                                      sub_nn=sub_nn)
                del key_ids, input_ids, masks, labels, flags, labels_dist
            else:
                loss, predictions, probabilities = model(input_ids, masks, labels)
                predictions_dict_test = utils.update_predictions_dict(sub_model, predictions_dict_test,
                                                                      key_ids, labels,
                                                                      probabilities,
                                                                      predictions,
                                                                      sub_nn=sub_nn)
                acc_num_test += torch.sum(predictions == labels)
                del key_ids, input_ids, masks, labels
            torch.cuda.empty_cache()
    loss_dist_sum /= test_len
    acc_num_test /= test_len
    end_test = time.process_time()
    total_time = end_test - start_test
    print("total time: ", total_time)
    if sub_nn != "with_flags":
        return float(acc_num_test), predictions_dict_test
    else:
        return float(loss_dist_sum), predictions_dict_test


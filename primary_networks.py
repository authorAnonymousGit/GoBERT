import utils
import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from FCBERT_classifier import FCBERT_PRIMARY, FCBERT_REGRESSION
import csv
import copy
from inference import results_analysis


def get_model(model_name, embeddings_path, labels_num, loss_type, dist_dict, denominator, alpha):
    if model_name == "FCBERT":
        return FCBERT_PRIMARY(embeddings_path, labels_num=labels_num, loss_type=loss_type,
                              dist_dict=dist_dict, denominator=denominator, alpha=alpha)
    elif model_name == "FCBERT_REGRESSION":
        return FCBERT_REGRESSION(embeddings_path, labels_num)
    else:
        raise TypeError("The model " + model_name + " is not defined")


def train_primary_model(models_df, train_df, val_df, test_df, text_feature,
                        embeddings_version, max_len, batch_size, label_col,
                        key_col, model_name, embeddings_path, labels_num,
                        loss_type, device, epochs_num, models_path, lr,
                        alpha=None):
    model_file_name = 'primary_' + loss_type

    if alpha is not None:
        model_file_name += '_alpha_' + str(int(alpha * 10))

    print("model_file_name")
    print(model_file_name)

    train_dataloader, val_dataloader = utils.create_dataloaders_train(train_df, val_df,
                                                                      text_feature,
                                                                      embeddings_version,
                                                                      max_len, batch_size,
                                                                      label_col, key_col)

    dist_dict = train_df[label_col].value_counts().to_dict()
    denominator = len(train_df)
    model = get_model(model_name, embeddings_path, labels_num, loss_type,
                      dist_dict, denominator, alpha=alpha)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    train_len = len(train_df)
    val_len = len(val_df)
    train_accuracy_list, train_loss_list, accuracy_list_val, loss_list_val = [], [], [], []
    best_val_acc = 0.0
    total_steps = len(train_dataloader) * epochs_num
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    best_model = None

    for epoch in range(epochs_num):
        utils.print_epochs_progress(epoch, epochs_num)
        start_train = time.process_time()
        acc_num_train = 0.0
        loss_scalar = 0.0
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
            key_ids = batch[3]
            model.zero_grad()
            loss, predictions, probabilities = model(input_ids, masks, labels)
            loss_scalar += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # acc_num_train += utils.add_correct_num(predictions, labels)
            acc_num_train += torch.sum(predictions == labels)
            optimizer.step()
            scheduler.step()
            predictions_dict_train = utils.update_predictions_dict('primary', predictions_dict_train,
                                                                   key_ids,
                                                                   labels,
                                                                   probabilities,
                                                                   predictions)
            del input_ids, masks, labels, loss, predictions, probabilities
            torch.cuda.empty_cache()

        loss_scalar /= train_len
        acc_num_train /= train_len
        end_train = time.process_time()
        total_time = end_train - start_train
        utils.print_train_epoch(epoch, acc_num_train, train_len, loss_scalar, total_time)
        train_loss_list.append(round(float(loss_scalar), 3))
        train_accuracy_list.append(round(float(acc_num_train), 3))
        utils.print_train_epoch_end(t0)
        val_acc, val_loss, predictions_dict_val = evaluate_val(model, val_dataloader,
                                                               device, val_len)
        accuracy_list_val.append(val_acc)
        loss_list_val.append(val_loss)
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), models_path + model_file_name + '.pkl')
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            utils.save_predictions_to_df(predictions_dict_train, models_path, 'train', model_file_name)
            utils.save_predictions_to_df(predictions_dict_val, models_path, 'validation', model_file_name)
    # utils.save_model(models_df, 'primary', labels_num, loss_type, lr,
    #                  epochs_num, batch_size, best_val_acc)
    utils.print_summary(models_path, model_file_name, accuracy_list_val)
    test_acc, predictions_dict_test = evaluate_test(test_df, best_model, device,
                                                    embeddings_version,
                                                    max_len, text_feature,
                                                    batch_size, label_col, key_col)
    utils.save_predictions_to_df(predictions_dict_test, models_path, 'test', model_file_name)
    utils.print_test_results(models_path, model_file_name, test_acc)
    return model_file_name, best_val_acc, test_acc


def run_primary(task_name, model_name, train_df, val_df, test_df,
                max_len, text_feature, embeddings_version,
                embeddings_path, config_primary, models_path, models_df,
                label_col, key_col):
    device = utils.find_device()

    loss_types, alphas, labels_num, epochs_num, lr, batch_size = \
        utils.read_config_networks(config_primary, "train_primary")

    primary_summary_rows = [['model_file_name', 'best_val_acc', 'val_mae', 'val_cem',
                             'test_acc', 'test_mae', 'test_cem']]
    for loss_type in loss_types:
        if loss_type == 'CrossEntropy':
            print('~' * 10, "primary", loss_type, '~' * 10)
            model_file_name, best_val_acc, test_acc = train_primary_model(models_df, train_df,
                                                                          val_df, test_df, text_feature,
                                                                          embeddings_version, max_len,
                                                                          batch_size, label_col,
                                                                          key_col, model_name,
                                                                          embeddings_path, labels_num,
                                                                          loss_type, device,
                                                                          epochs_num, models_path, lr)
            _, val_mae, val_cem = results_analysis.evaluate_results(models_path + 'predictions_validation_' +
                                                                    model_file_name + '.csv',
                                                                    'label', 'prediction')
            _, test_mae, test_cem = results_analysis.evaluate_results(models_path + 'predictions_test_' +
                                                                      model_file_name + '.csv',
                                                                      'label', 'prediction')
            primary_summary_rows.extend([[model_file_name, best_val_acc, val_mae, val_cem,
                                          test_acc, test_mae, test_cem]])

        else:  # loss_type == 'OrdinalTextClassification'
            for alpha in alphas:
                print('~' * 10, "primary", loss_type, alpha, '~' * 10)
                model_file_name, best_val_acc, test_acc = train_primary_model(models_df, train_df,
                                                                              val_df, test_df, text_feature,
                                                                              embeddings_version, max_len,
                                                                              batch_size, label_col,
                                                                              key_col, model_name,
                                                                              embeddings_path, labels_num,
                                                                              loss_type, device, epochs_num,
                                                                              models_path, lr, alpha)

                _, val_mae, val_cem = results_analysis.evaluate_results(models_path + 'predictions_validation_' +
                                                                        model_file_name + '.csv',
                                                                        'label', 'prediction')
                _, test_mae, test_cem = results_analysis.evaluate_results(models_path + 'predictions_test_' +
                                                                          model_file_name + '.csv',
                                                                          'label', 'prediction')
                primary_summary_rows.extend([[model_file_name, best_val_acc, val_mae, val_cem,
                                              test_acc, test_mae, test_cem]])

    with open(models_path + "primary_results_summary.csv", "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(primary_summary_rows)

    return models_df


def evaluate_val(model, val_dataloader, device, val_len):
    start_val = time.process_time()
    acc_num_val = 0.0
    loss_scalar = 0.0
    model.eval()
    predictions_dict_val = dict()
    for batch_idx, batch in enumerate(val_dataloader):
        input_ids = batch[0].to(device, dtype=torch.long)
        masks = batch[1].to(device, dtype=torch.long)
        labels = batch[2].to(device, dtype=torch.long)
        key_ids = batch[3]
        with torch.no_grad():
            loss, predictions, probabilities = model(input_ids, masks, labels)
            predictions_dict_val = utils.update_predictions_dict('primary', predictions_dict_val,
                                                                 key_ids,
                                                                 labels,
                                                                 probabilities,
                                                                 predictions)
            loss_scalar += loss.item()
            # acc_num_val += utils.add_correct_num(predictions, labels)
            acc_num_val += torch.sum(predictions == labels)
        del input_ids, masks, labels, loss, predictions, probabilities
        torch.cuda.empty_cache()
    loss_scalar /= val_len
    acc_num_val /= val_len
    end_val = time.process_time()
    total_time = end_val - start_val
    utils.print_validation_epoch(acc_num_val, val_len, loss_scalar, total_time)
    return float(acc_num_val), float(loss_scalar), predictions_dict_val


def evaluate_test(test_df, model, device, embeddings_version, max_len,
                  text_feature, batch_size, label_col, key_col):
    start_val = time.process_time()
    test_dataloader = utils.create_dataloaders_test(test_df, embeddings_version,
                                                    max_len, text_feature, label_col,
                                                    key_col, batch_size)
    test_len = len(test_df)
    acc_num_test = 0.0
    model.eval()
    predictions_dict_test = dict()
    for batch_idx, batch in enumerate(test_dataloader):
        input_ids = batch[0].to(device, dtype=torch.long)
        masks = batch[1].to(device, dtype=torch.long)
        labels = batch[2].to(device, dtype=torch.long)
        key_ids = batch[3]
        with torch.no_grad():
            _, predictions, probabilities = model(input_ids, masks, labels)
            predictions_dict_test = utils.update_predictions_dict('primary', predictions_dict_test,
                                                                  key_ids, labels,
                                                                  probabilities,
                                                                  predictions)
            # acc_num_test += utils.add_correct_num(predictions, labels)
            acc_num_test += torch.sum(predictions == labels)
        del input_ids, masks, labels, predictions, probabilities
        torch.cuda.empty_cache()
    end_val = time.process_time()
    total_time = end_val - start_val
    print("total time: ", total_time)
    return float(acc_num_test / test_len), predictions_dict_test

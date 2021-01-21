from inference.models import GCN
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from inference.process_graph_input import GraphInputReader
from torch_geometric.data import DataLoader as GeometricDataLoader
import csv
import copy
from transformers import AdamW, get_linear_schedule_with_warmup


def create_dataloaders_inference(sub_models, files_path, labels_num, sub_nn,
                                 edge_creation_procedure, bidirectional, batch_size=32,
                                 filter_by_origin_pred=None, primary_model_name="primary"):
    train_datareader = GraphInputReader(sub_models, files_path,
                                        "train", labels_num, sub_nn,
                                        edge_creation_procedure,
                                        bidirectional,
                                        filter_by_origin_pred,
                                        primary_model_name=primary_model_name)
    val_datareader = GraphInputReader(sub_models, files_path,
                                      "validation", labels_num, sub_nn,
                                      edge_creation_procedure,
                                      bidirectional,
                                      filter_by_origin_pred,
                                      primary_model_name=primary_model_name)
    test_datareader = GraphInputReader(sub_models, files_path,
                                       "test", labels_num, sub_nn,
                                       edge_creation_procedure,
                                       bidirectional,
                                       filter_by_origin_pred,
                                       primary_model_name=primary_model_name)

    train_dataloader = GeometricDataLoader(train_datareader, batch_size=batch_size, shuffle=True)
    val_dataloader = GeometricDataLoader(val_datareader, batch_size=batch_size, shuffle=False)
    test_dataloader = GeometricDataLoader(test_datareader, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def epoch_train(model, train_dataloader, inv_prox_mat, loss_type,
                optimizer, scheduler, device, final_prediction_procedure,
                labels_num, alpha=None):
    model.train()
    optimizer.zero_grad()
    for data_batch in train_dataloader:  # Iterate in batches over the training dataset.
        data = data_batch[0]
        primary_nodes = data_batch[4]
        num_of_nodes = data_batch[5]
        label_encoding = data_batch[6].to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        label = data.y.to(device)

        primary_idxs_in_batch = None
        if final_prediction_procedure == 'primary':
            primary_idxs_in_batch = [(sum(num_of_nodes[:enum_graph]) + primary_node).item()
                                     for enum_graph, primary_node in enumerate(primary_nodes)]
        model.zero_grad()
        out, softmax_vals = model(x, edge_index, batch, primary_idxs_in_batch)  # Perform a single forward pass.

        if loss_type == 'CrossEntropy':
            loss = utils.CrossEntropyLoss(out, label)
        else:
            loss1 = utils.CrossEntropyProxLoss(out, label, inv_prox_mat)
            loss2 = utils.MseProxLoss(softmax_vals, label, inv_prox_mat, labels_num, device)
            loss = alpha * loss1 + (1 - alpha) * loss2
        loss.backward()  # Derive gradients.

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        optimizer.zero_grad()  # Clear gradients.
        del x, edge_index, batch, label, label_encoding
        torch.cuda.empty_cache()
    return model


def test(epoch, model, dataloader, device, final_prediction_procedure,
         inv_prox_mat, loss_type, labels_num, save_results=False, alpha=None,
         prox_mat=None):
    model.eval()
    correct = 0
    all_new_rows = []
    total_loss = 0.0

    # It is computed only for the validation
    cem_num = 0.0
    cem_den = 0.0

    with torch.no_grad():
        for data_batch in dataloader:  # Iterate in batches over the training/test dataset.
            data = data_batch[0]
            key_idxs = data_batch[1]
            primary_pred = data_batch[2]
            primary_dist = data_batch[3]
            primary_nodes = data_batch[4]
            num_of_nodes = data_batch[5]
            label_encoding = data_batch[6].to(device)
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            label = data.y.to(device)

            primary_idxs_in_batch = None
            if final_prediction_procedure == 'primary':
                primary_idxs_in_batch = [(sum(num_of_nodes[:enum_graph]) + primary_node).item()
                                         for enum_graph, primary_node in enumerate(primary_nodes)]

            out, softmax_vals = model(x, edge_index, batch, primary_idxs_in_batch)  # Perform a single forward pass.

            if loss_type == 'CrossEntropy':
                loss = utils.CrossEntropyLoss(out, label)
            else:
                loss1 = utils.CrossEntropyProxLoss(out, label, inv_prox_mat)
                loss2 = utils.MseProxLoss(softmax_vals, label, inv_prox_mat, labels_num, device)
                loss = alpha * loss1 + (1 - alpha) * loss2
            total_loss += loss.item()

            pred = out.argmax(dim=1)  # Use the class with highest probability.
            softmax_vals = [[round(elem.item(), 4) for elem in dist] for dist in softmax_vals]
            if save_results:
                new_rows = [[epoch, key_idxs[i].item(), label[i].item() + 1,
                             primary_pred[i].item(), pred[i].item() + 1,
                             primary_dist[i], softmax_vals[i]] for i in range(len(pred))]
                all_new_rows.extend(new_rows)
            correct += int((pred == label).sum())  # Check against ground-truth labels.

            # Compute cem if prox_mat exists
            if prox_mat is not None:
                cem_num += sum(prox_mat[pred, label])
                cem_den += sum(prox_mat[label, label])

            del x, edge_index, batch, label, label_encoding
            torch.cuda.empty_cache()

    if prox_mat is not None:
        cem = cem_num / cem_den
    else:
        cem = None

    return correct, total_loss/len(dataloader), all_new_rows, cem


def is_selected_model(val_loss, val_corr, val_cem, test_corr, best_loss,
                      best_corr, best_cem, model_selection_procedure):
    selected_model = False
    updated_corr = best_corr
    updated_loss = best_loss
    updated_cem = best_cem

    if model_selection_procedure == 'test_acc':
        if test_corr > best_corr:
            selected_model = True
            updated_corr = test_corr
    elif model_selection_procedure == 'val_acc':
        if val_corr > best_corr:
            selected_model = True
            updated_corr = val_corr
    elif model_selection_procedure == 'val_loss':
        if val_loss < best_loss:
            selected_model = True
            updated_loss = val_loss
    elif model_selection_procedure == 'val_cem':
        if val_cem > best_cem:
            selected_model = True
            updated_cem = val_cem
    elif model_selection_procedure == 'each_epoch':
        selected_model = True
    return selected_model, updated_loss, updated_corr, updated_cem


def run_graph_classification(primary_model_name, labels_num, edge_creation_procedure,
                             final_prediction_procedure, model_data_procedure,
                             hidden_channels, batch_size, node_features_num,
                             epochs_num, bidirectional, models_path, sub_models,
                             inference_path, sub_nn, loss_type, alpha, model_selection_procedure):
    device = utils.find_device()

    dist_dict_train, denominator_train = utils.get_prox_params(models_path, 'train', primary_model_name)
    inv_prox_mat = torch.tensor(utils.create_prox_mat(dist_dict_train, denominator_train, inv=True)).to(device)

    dist_dict_val, denominator_val = utils.get_prox_params(models_path, 'validation', primary_model_name)
    prox_mat_val = torch.tensor(utils.create_prox_mat(dist_dict_val, denominator_val, inv=False)).to(device)

    dataset_types = ['train', 'val', 'test']
    results_dict = {dataset_type:
                    [['epoch', 'key_idx', 'label', 'primary_pred', 'graph_pred', 'primary_dist', 'graph_dist']]
                    for dataset_type in dataset_types}

    if model_data_procedure == 'by_origin_pred':
        train_dataloaders, val_dataloaders, test_dataloaders = dict(), dict(), dict()
        models, optimizers, schedulers = dict(), dict(), dict()
        best_models, best_corr_by_model, best_cem_by_model, best_loss_by_model = dict(), dict(), dict(), dict()

        for label in range(1, labels_num + 1):
            train_dataloaders[label], val_dataloaders[label], test_dataloaders[label] = \
                create_dataloaders_inference(sub_models, models_path, labels_num,
                                             sub_nn, edge_creation_procedure,
                                             bidirectional, batch_size,
                                             filter_by_origin_pred=label,
                                             primary_model_name=primary_model_name)
            models[label] = GCN(node_features_num, hidden_channels, labels_num)
            models[label].to(device)
            optimizers[label] = torch.optim.Adam(models[label].parameters(), lr=0.0075)
            total_steps = len(train_dataloaders[label]) * epochs_num
            schedulers[label] = get_linear_schedule_with_warmup(optimizers[label],
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_steps)

            best_models[label] = GCN(node_features_num, hidden_channels, labels_num)
            best_corr_by_model[label] = 0.0
            best_cem_by_model[label] = 0.0
            best_loss_by_model[label] = float('inf')

        print('*' * 20)
        for epoch in range(1, epochs_num + 1):
            print(f'Epoch: {epoch:03d}')
            print('*'*20)
            epoch_best_corr_train, epoch_best_corr_val, epoch_best_corr_test = 0.0, 0.0, 0.0
            total_train, total_val, total_test = 0.0, 0.0, 0.0
            for label in range(1, labels_num + 1):
                models[label] = epoch_train(models[label], train_dataloaders[label],
                                            inv_prox_mat, loss_type, optimizers[label],
                                            schedulers[label], device, final_prediction_procedure,
                                            labels_num, alpha)
                train_corr, train_loss, _, _ = test(epoch, models[label], train_dataloaders[label], device,
                                                    final_prediction_procedure, inv_prox_mat, loss_type,
                                                    labels_num, alpha=alpha)
                train_len = len(train_dataloaders[label].dataset)
                total_train += train_len
                train_acc = train_corr / train_len

                val_corr, val_loss, _, val_cem = test(epoch, models[label], val_dataloaders[label], device,
                                                      final_prediction_procedure, inv_prox_mat, loss_type,
                                                      labels_num, alpha=alpha, prox_mat=prox_mat_val)
                val_len = len(val_dataloaders[label].dataset)
                total_val += val_len
                val_acc = val_corr / val_len

                test_corr, test_loss, _, _ = test(epoch, models[label], test_dataloaders[label], device,
                                                  final_prediction_procedure, inv_prox_mat, loss_type,
                                                  labels_num, alpha=alpha)
                test_len = len(test_dataloaders[label].dataset)
                total_test += test_len
                test_acc = test_corr / test_len

                print(f'Model of Label: {label:03d}')
                print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
                print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')

                selected_model, updated_loss, \
                    updated_corr, updated_cem = is_selected_model(val_loss, val_corr, val_cem,
                                                                  test_corr, best_loss_by_model[label],
                                                                  best_corr_by_model[label],
                                                                  best_cem_by_model[label],
                                                                  model_selection_procedure)
                if selected_model:
                    best_corr_by_model[label] = updated_corr
                    best_loss_by_model[label] = updated_loss
                    best_cem_by_model[label] = updated_cem
                    best_models[label] = copy.deepcopy(models[label])

                # Update csv_rows by the best model to the current epoch
                train_corr, _, new_rows_train, _ = test(epoch, best_models[label], train_dataloaders[label], device,
                                                        final_prediction_procedure, inv_prox_mat, loss_type,
                                                        labels_num, save_results=True, alpha=alpha)
                epoch_best_corr_train += train_corr
                results_dict['train'] += new_rows_train

                val_corr, _, new_rows_val, _ = test(epoch, best_models[label], val_dataloaders[label], device,
                                                    final_prediction_procedure, inv_prox_mat, loss_type,
                                                    labels_num, save_results=True, alpha=alpha)
                epoch_best_corr_val += val_corr
                results_dict['val'] += new_rows_val

                test_corr, _, new_rows_test, _ = test(epoch, best_models[label], test_dataloaders[label], device,
                                                      final_prediction_procedure, inv_prox_mat, loss_type,
                                                      labels_num, save_results=True, alpha=alpha)
                epoch_best_corr_test += test_corr
                results_dict['test'] += new_rows_test

            best_train_acc_epoch = epoch_best_corr_train / total_train
            best_val_acc_epoch = epoch_best_corr_val / total_val
            best_test_acc_epoch = epoch_best_corr_test / total_test
            print('-'*20)
            print(f'Train Acc by best conf: {best_train_acc_epoch:.4f}, '
                  f'Val Acc by best conf: {best_val_acc_epoch:.4f}, '
                  f'Test Acc by best conf: {best_test_acc_epoch:.4f}')

    else:  # model_data_procedure == 'all'
        train_dataloader, val_dataloader, test_dataloader = \
            create_dataloaders_inference(sub_models, models_path, labels_num,
                                         sub_nn, edge_creation_procedure,
                                         bidirectional, batch_size,
                                         primary_model_name=primary_model_name)

        model = GCN(node_features_num, hidden_channels, labels_num)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0075)
        total_steps = len(train_dataloader) * epochs_num
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        best_acc = 0.0
        best_loss = float('inf')
        best_cem = 0.0
        best_model = ""
        for epoch in range(1, epochs_num + 1):
            model = epoch_train(model, train_dataloader, inv_prox_mat, loss_type,
                                optimizer, scheduler, device, final_prediction_procedure,
                                labels_num, alpha)
            train_corr, train_loss, _, _ = test(epoch, model, train_dataloader, device,
                                                final_prediction_procedure, inv_prox_mat,
                                                loss_type, labels_num, alpha=alpha)
            train_len = len(train_dataloader.dataset)
            train_acc = train_corr / train_len

            val_corr, val_loss, _, val_cem = test(epoch, model, val_dataloader, device,
                                                  final_prediction_procedure, inv_prox_mat,
                                                  loss_type, labels_num, alpha=alpha)
            val_len = len(val_dataloader.dataset)
            val_acc = val_corr / val_len

            test_corr, test_loss, _, _ = test(epoch, model, test_dataloader, device,
                                              final_prediction_procedure, inv_prox_mat,
                                              loss_type, labels_num, alpha=alpha)
            test_len = len(test_dataloader.dataset)
            test_acc = test_corr / test_len

            print(f'Epoch: {epoch:03d}')
            print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')

            selected_model, updated_loss, \
                updated_corr, updated_cem = is_selected_model(val_loss, val_acc, val_cem,
                                                              test_acc, best_loss, best_acc,
                                                              best_cem, model_selection_procedure)
            if selected_model:
                best_acc = updated_corr
                best_loss = updated_loss
                best_cem = updated_cem
                best_model = copy.deepcopy(model)

            # Update csv_rows by the best model to the current epoch
            train_corr, _, new_rows_train, _ = test(epoch, best_model, train_dataloader, device,
                                                    final_prediction_procedure, inv_prox_mat, loss_type,
                                                    labels_num, save_results=True, alpha=alpha)
            results_dict['train'] += new_rows_train

            val_corr, _, new_rows_val, _ = test(epoch, best_model, val_dataloader, device,
                                                final_prediction_procedure, inv_prox_mat, loss_type,
                                                labels_num, save_results=True, alpha=alpha)
            results_dict['val'] += new_rows_val

            test_corr, _, new_rows_test, _ = test(epoch, best_model, test_dataloader, device,
                                                  final_prediction_procedure, inv_prox_mat, loss_type,
                                                  labels_num, save_results=True, alpha=alpha)
            results_dict['test'] += new_rows_test

            best_train_acc_epoch = train_corr / train_len
            best_val_acc_epoch = val_corr / val_len
            best_test_acc_epoch = test_corr / test_len
            print('-' * 20)
            print(f'Train Acc by best conf: {best_train_acc_epoch:.4f}, '
                  f'Val Acc by best conf: {best_val_acc_epoch:.4f}, '
                  f'Test Acc by best conf: {best_test_acc_epoch:.4f}')

    # Write the best results (by test accuracy) to files
    for dataset_type in dataset_types:
        file_name = inference_path + dataset_type + ".csv"
        with open(file_name, "w+") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(results_dict[dataset_type])


from inference import graph_classification
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import itertools
import csv
from config import ConfigMain, ConfigGraphClassification
from inference import results_analysis


def run_inference_by_options(primary_model_name_gnn, labels_num, edge_creation_procedure,
                             final_prediction_procedure, model_data_procedure, hidden_channels,
                             batch_size, node_features_num, epochs_num, bidirectional,
                             models_path, sub_models, inference_path,
                             sub_nn, loss_type, alpha, model_selection_procedure):

    inference_path += edge_creation_procedure + '//' + final_prediction_procedure + '//' + \
        model_data_procedure + '//' + loss_type + '//'

    if alpha is not None:
        inference_path += 'alpha_' + str(int(alpha * 10)) + '//'

    if not os.path.exists(inference_path):
        os.makedirs(inference_path)

    graph_classification.run_graph_classification(primary_model_name_gnn, labels_num, edge_creation_procedure,
                                                  final_prediction_procedure, model_data_procedure,
                                                  hidden_channels, batch_size, node_features_num,
                                                  epochs_num, bidirectional, models_path, sub_models,
                                                  inference_path, sub_nn, loss_type, alpha,
                                                  model_selection_procedure)

    measures_by_epochs_val = []
    measures_by_epochs_test = []
    for epoch in range(1, epochs_num + 1):
        val_file_name = inference_path + "val.csv"
        acc_gnn_val, mae_gnn_val, cem_gnn_val = \
            results_analysis.evaluate_results(val_file_name, "label", "graph_pred", by_epoch=epoch)
        measures_by_epochs_val.append((acc_gnn_val, mae_gnn_val, cem_gnn_val))

        test_file_name = inference_path + "test.csv"
        acc_gnn_test, mae_gnn_test, cem_gnn_test = \
            results_analysis.evaluate_results(test_file_name, "label", "graph_pred", by_epoch=epoch)
        measures_by_epochs_test.append((acc_gnn_test, mae_gnn_test, cem_gnn_test))

    return measures_by_epochs_val, measures_by_epochs_test


def run_inference(iter_num, sub_nn):
    config_main = ConfigMain()
    config_graph_classification = ConfigGraphClassification()
    task_name, embeddings_version, models_path, sub_models, inference_type = \
        utils.read_config_main_for_inference(config_main)
    models_path += sub_nn + '//' + str(iter_num) + '//'

    if inference_type == "graph_classification":
        labels_num, single_graph_structure, edge_creation_procedure, \
            final_prediction_procedure, model_data_procedure, \
            loss_types, alphas, hidden_channels, \
            batch_size, node_features_num, \
            epochs_num, bidirectional, \
            primary_model_name_baseline, \
            primary_model_name_gnn,\
            model_selection_procedures = \
            utils.read_config_graph_classification(config_graph_classification)

        primary_baseline_path = models_path + "predictions_test_" + primary_model_name_baseline + ".csv"
        acc_primary_baseline, mae_primary_baseline, cem_primary_baseline = \
            results_analysis.evaluate_results(primary_baseline_path, "label", "prediction")
        # primary_gnn_path = models_path + "predictions_test_" + primary_model_name_gnn + ".csv"
        # acc_primary_gnn, mae_primary_gnn, cem_primary_gnn = \
        #     results_analysis.evaluate_results(primary_gnn_path, "label", "prediction")

        if single_graph_structure:
            edge_creation_procedure = edge_creation_procedure[0]
            final_prediction_procedure = final_prediction_procedure[0]
            model_data_procedure = model_data_procedure[0]
            loss_type = loss_types[0]
            alpha = alphas[0]
            model_selection_procedure = model_selection_procedures[0]

            inference_path = ".//GNN_results//" + task_name + \
                             '//' + embeddings_version + '//' + \
                             sub_nn + '//' + str(iter_num) + '//' + \
                             model_selection_procedure + '//' + \
                             edge_creation_procedure + '//' + \
                             final_prediction_procedure + '//' + \
                             model_data_procedure + '//' + \
                             loss_type + '//'

            if alpha is not None:
                inference_path += 'alpha_' + str(int(alpha * 10)) + '//'

            if not os.path.exists(inference_path):
                os.makedirs(inference_path)

            graph_classification.run_graph_classification(primary_model_name_gnn,
                                                          labels_num, edge_creation_procedure,
                                                          final_prediction_procedure,
                                                          model_data_procedure,
                                                          hidden_channels, batch_size,
                                                          node_features_num,
                                                          epochs_num, bidirectional,
                                                          models_path, sub_models,
                                                          inference_path, sub_nn,
                                                          loss_type, alpha,
                                                          model_selection_procedure)
        else:
            for model_selection_procedure in model_selection_procedures:
                inference_path = ".//GNN_results//" + task_name + \
                                 '//' + embeddings_version + '//' + \
                                 sub_nn + '//' + str(iter_num) + '//' + \
                                 model_selection_procedure + '//'
                if not os.path.exists(inference_path):
                    os.makedirs(inference_path)

                # Run graph_classification on any possible graph structure
                options_summary_rows = [['edge_creation_procedure', 'final_prediction_procedure',
                                         'model_data_procedure', 'loss_type', 'alpha', 'epoch',
                                         'acc_primary_baseline', 'acc_gnn_val', 'acc_gnn_test',
                                         'relative_improvement_acc_test',
                                         'mae_primary_baseline', 'mae_gnn_val', 'mae_gnn_test',
                                         'relative_improvement_mae_test',
                                         'cem_primary_baseline', 'cem_gnn_val', 'cem_gnn_test',
                                         'relative_improvement_cem_test']]
                for option in itertools.product(*[edge_creation_procedure,
                                                  final_prediction_procedure,
                                                  model_data_procedure,
                                                  loss_types]):
                    loss_type = option[3]
                    if loss_type == 'CrossEntropy':
                        print('~' * 10, option[0], option[1], option[2], loss_type, '~' * 10)
                        measures_by_epochs_val, measures_by_epochs_test = run_inference_by_options(
                            primary_model_name_gnn, labels_num, option[0], option[1], option[2],
                            hidden_channels, batch_size, node_features_num, epochs_num, bidirectional,
                            models_path, sub_models, inference_path, sub_nn, loss_type, None,
                            model_selection_procedure)

                        for epoch, measures_val, measures_test in zip(range(1, epochs_num + 1),
                                                                      measures_by_epochs_val,
                                                                      measures_by_epochs_test):
                            rel_improvement_acc_test, rel_improvement_mae_test, rel_improvement_cem_test = \
                                results_analysis.compute_relative_improvement(acc_primary_baseline, measures_test[0],
                                                                              mae_primary_baseline, measures_test[1],
                                                                              cem_primary_baseline, measures_test[2])

                            new_row = [[option[0], option[1], option[2], loss_type, 'None', epoch,
                                        acc_primary_baseline, measures_val[0], measures_test[0],
                                        rel_improvement_acc_test,
                                        mae_primary_baseline, measures_val[1], measures_test[1],
                                        rel_improvement_mae_test,
                                        cem_primary_baseline, measures_val[2], measures_test[2],
                                        rel_improvement_cem_test]]

                            options_summary_rows.extend(new_row)
                    else:  # loss_type == 'OrdinalTextClassification'
                        for alpha in alphas:
                            print('~' * 10, option[0], option[1], option[2], option[3], alpha, '~' * 10)
                            measures_by_epochs_val, measures_by_epochs_test = run_inference_by_options(
                                primary_model_name_gnn, labels_num, option[0], option[1], option[2],
                                hidden_channels, batch_size, node_features_num, epochs_num,
                                bidirectional, models_path, sub_models, inference_path,
                                sub_nn, loss_type, alpha, model_selection_procedure)

                            for epoch, measures_val, measures_test in zip(range(1, epochs_num + 1),
                                                                          measures_by_epochs_val,
                                                                          measures_by_epochs_test):
                                rel_improvement_acc_test, rel_improvement_mae_test, rel_improvement_cem_test = \
                                    results_analysis.compute_relative_improvement(acc_primary_baseline,
                                                                                  measures_test[0],
                                                                                  mae_primary_baseline,
                                                                                  measures_test[1],
                                                                                  cem_primary_baseline,
                                                                                  measures_test[2])

                                new_row = [[option[0], option[1], option[2], loss_type, alpha, epoch,
                                            acc_primary_baseline, measures_val[0], measures_test[0],
                                            rel_improvement_acc_test,
                                            mae_primary_baseline, measures_val[1], measures_test[1],
                                            rel_improvement_mae_test,
                                            cem_primary_baseline, measures_val[2], measures_test[2],
                                            rel_improvement_cem_test]]

                                options_summary_rows.extend(new_row)

                with open(inference_path + "results_summary_" + str(iter_num) + '_model_selection_' +
                          model_selection_procedure + ".csv", "w+") as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    csv_writer.writerows(options_summary_rows)


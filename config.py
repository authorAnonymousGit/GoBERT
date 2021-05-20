import numpy as np


class ConfigMain:
    TASK_NAME = 'AmazonFashion'  # 'sst-5' / 'SemEval2017' / 'AmazonFashion'
    MAX_LEN = 256
    TEXT_FEATURE = 'text'
    MODEL_NAME = 'FCBERT'
    EMBEDDINGS_VERSION = 'bert-large-uncased'  # "bert-base-uncased"/"bert-large-uncased"/"roberta-base"/"albert-large-v2"
    EMBEDDINGS_PATH = 'bert-large-uncased'  # "bert-base-uncased"/"bert-large-uncased"/"roberta-base"/"albert-large-v2"
    MODELS_PATH = './/trained_models//' + TASK_NAME + '//' + EMBEDDINGS_VERSION + '//'
    LABEL_FEATURE = "overall"
    KEY_FEATURE = "key_index"
    SUB_MODELS = ['model_12', 'model_13', 'model_14', 'model_15',
                  'model_23', 'model_24', 'model_25', 'model_34',
                  'model_35', 'model_45']
    # 5 For 'sst-5' and 'AmazonFashion', 3 For 'SemEval2017' (['model_12', 'model_13', 'model_23'])


class ConfigPrimary:
    # Try different variations of loss functions out of "CrossEntropy"/ "OrdinalTextClassification-?".
    LOSS_TYPE = ["CrossEntropy",
                 # "OrdinalTextClassification-F",
                 # "OrdinalTextClassification-G",
                 # "OrdinalTextClassification-H",
                 # "OrdinalTextClassification-I",
                 # "OrdinalTextClassification-J",
                 # "OrdinalTextClassification-K",
                 ]
    ALPHA = np.linspace(0.0, 0.0, 1)  # Irrelevant
    BETA = np.linspace(0.0, 0.0, 1)  # Irrelevant
    LABELS_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and 'AmazonFashion'
    EPOCHS_NUM = 4
    LEARNING_RATE = [0.06, 0.15, 10]  # Irrelevant
    BATCH_SIZE = [16, 16, 1]
    MODEL_SELECTION_PROCEDURE = 'cem'  # acc / cem


class ConfigSubModel:
    LOSS_TYPE = "CrossEntropy"
    LABELS_NUM = 2
    EPOCHS_NUM = 3
    LEARNING_RATE = [0.01, 0.15, 15]  # Irrelevant
    BATCH_SIZE = [16, 16, 1]


class ConfigGraphClassification:
    LABELS_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and AmazonFashion
    SINGLE_GRAPH_STRUCTURE = False # Irrelevant
    EDGE_CREATION_PROCEDURE = ['pyramid_narrow_disconn_dist_1', 'pyramid_narrow_conn_dist_1',
                               'pyramid_wide_disconn_dist_1', 'pyramid_wide_conn_dist_1']
    # 'star', 'star_conn_kids', 'complete' are also available but we did not use them in out experiments

    # Select the method of yielding the GCN's prediction
    FINAL_PREDICTION_PROCEDURE = ['primary', ]  # 'primary' / 'avg_all'

    MODEL_DATA_PROCEDURE = ['by_origin_pred', ]  # 'by_origin_pred' / 'all'. We used only by_origin_pred.

    # Try different variations of loss functions out of "CrossEntropy"/ "OrdinalTextClassification-?".
    LOSS_TYPE = ["CrossEntropy", "OrdinalTextClassification-E",
                 "OrdinalTextClassification-F", "OrdinalTextClassification-G",
                 "OrdinalTextClassification-H", "OrdinalTextClassification-I",
                 "OrdinalTextClassification-J", "OrdinalTextClassification-K"]

    ALPHA = np.linspace(0.0, 0.0, 1)  # Irrelevant
    BIDIRECTIONAL = True
    HIDDEN_CHANNELS = 64
    BATCH_SIZE = 32
    NODE_FEATURES_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and 'AmazonFashion'
    EPOCHS_NUM = 5

    # Select the primary model by its loss function. The GCN will be built upon this model.
    # The possible options are "primary_CrossEntropy"/ "primary_OrdinalTextClassification-?".
    # The used model has to be fine-tuned in advance as a primary model.
    PRIMARY_MODEL_NAME_BASELINE = 'primary_CrossEntropy'
    PRIMARY_MODEL_NAME_GNN = PRIMARY_MODEL_NAME_BASELINE
    PRIMARY_LOSS_TYPE = PRIMARY_MODEL_NAME_BASELINE.split('_')[1]
    MODEL_SELECTION_PROCEDURE = ['val_cem', ]  # 'test_acc'/'each_epoch'/'val_loss'/'val_cem'/'val_acc'


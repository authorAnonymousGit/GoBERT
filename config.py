import numpy as np


class ConfigMain:
    TASK_NAME = 'sst-5'  # 'sst-5'/'SemEval2017'/'AmazonFashion'
    MAX_LEN = 256
    TEXT_FEATURE = 'text'
    MODEL_NAME = 'FCBERT'
    EMBEDDINGS_VERSION = 'roberta-base'  # "bert-base-uncased"/"bert-large-uncased"/"roberta-base"/"albert-large-v2"
    EMBEDDINGS_PATH = 'roberta-base'  # "bert-base-uncased"/"bert-large-uncased"/"roberta-base"/"albert-large-v2"
    MODELS_PATH = './/trained_models//' + TASK_NAME + '//' + EMBEDDINGS_VERSION + '//'
    LABEL_FEATURE = "overall"
    KEY_FEATURE = "key_index"
    SUB_MODELS = ['model_12', 'model_13', 'model_14', 'model_15',
                  'model_23', 'model_24', 'model_25', 'model_34',
                  'model_35', 'model_45']  # For 'sst-5' and 'AmazonFashion',
    # For 'SemEval2017' Change the SUB_MODELS to ['model_12', 'model_13', 'model_23']
    INFERENCE_TYPE = "graph_classification"


class ConfigPrimary:
    LOSS_TYPE = ["CrossEntropy", ]  # "CrossEntropy"/"OrdinalTextClassification"
    ALPHA = np.linspace(0.0, 1.0, 11)
    LABELS_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and 'AmazonFashion'
    EPOCHS_NUM = 5
    LEARNING_RATE = [0.06, 0.15, 10]
    BATCH_SIZE = [16, 16, 1]


class ConfigSubModel:
    LOSS_TYPE = "CrossEntropy"
    LABELS_NUM = 2
    EPOCHS_NUM = 5
    LEARNING_RATE = [0.01, 0.15, 15]
    BATCH_SIZE = [16, 16, 1]


class ConfigGraphClassification:
    LABELS_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and AmazonFashion
    SINGLE_GRAPH_STRUCTURE = False
    EDGE_CREATION_PROCEDURE = ['pyramid_narrow_conn_dist_1', ]  # 'star'/'complete'/'pyramid_narrow_disconn_dist_1'/
    # 'pyramid_wide_conn_dist_1'/'pyramid_wide_disconn_dist_1'
    FINAL_PREDICTION_PROCEDURE = ['primary', ]  # 'primary' / 'avg_all'
    MODEL_DATA_PROCEDURE = ['by_origin_pred', ]  # 'by_origin_pred' / 'all'
    LOSS_TYPE = ["OrdinalTextClassification", ]  # "CrossEntropy" / "OrdinalTextClassification"
    ALPHA = np.linspace(0.0, 1.0, 11)  # Relevant Only for OrdinalTextClassification Loss
    BIDIRECTIONAL = True
    HIDDEN_CHANNELS = 32
    BATCH_SIZE = 32
    NODE_FEATURES_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and 'AmazonFashion'
    EPOCHS_NUM = 20
    PRIMARY_MODEL_NAME_BASELINE = 'primary_CrossEntropy'
    PRIMARY_MODEL_NAME_GNN = 'primary_CrossEntropy'
    MODEL_SELECTION_PROCEDURE = ['val_cem', 'val_acc']  # 'test_acc'/'each_epoch'/'val_loss'/'val_cem'/'val_acc'

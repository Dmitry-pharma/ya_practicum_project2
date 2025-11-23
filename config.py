class Config:
    # для воспроизводимости
    SEED = 42

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"
#   TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"  
#     IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"  


    # Гиперпараметры
    BATCH_SIZE = 256 
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    EPOCHS = 50
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 4

    # Пути
    TRAIN_DF_PATH = "data/train_data.csv"
    VAL_DF_PATH = "data/test_data.csv"
    SAVE_PATH = "best_model.pth"

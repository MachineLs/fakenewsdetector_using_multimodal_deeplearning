import torch

from data.config import Config
from data.reddit.data_loader import TwitterDatasetLoader


class RedditConfig(Config):
    name = 'reddit'
    DatasetLoader = TwitterDatasetLoader
    # change your local path here
    data_path = './'
    # data_path = '../../../../../media/external_3TB/3TB/ghorbanpoor/twitter/'
    # data_path = '/home/faeze/PycharmProjects/fake_news_detection/data/twitter/'
    
    # output_path = '../../../../../media/external_10TB/10TB/ghorbanpoor/'
    # useless output path
    output_path = './output'

    
    # image direction, be careful
    train_image_path = data_path + 'images/'
    validation_image_path = data_path + 'images/'
    test_image_path = data_path + 'images/'

    # csv path, local file path, the relation between text and image
    # csv have three features, text, image, label. image is the file name with suffix. label 1/0.
    train_text_path = data_path + 'training_data.csv'
    validation_text_path = data_path + 'test_data.csv'
    test_text_path = data_path + 'vaildation_data.csv'

    # change the batch size to fit your device
    batch_size = 128
    # batch_size = 32
    # epochs = 100 epochs, 
    epochs = 20
    # be careful, magic
    num_workers = 0

    head_lr = 1e-03
    image_encoder_lr = 1e-04
    text_encoder_lr = 1e-04
    attention_lr = 1e-3
    classification_lr = 1e-03

    head_weight_decay = 0.001
    attention_weight_decay = 0.001
    classification_weight_decay = 0.001

    # depends on your device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #image_model_name = '../../../../../media/external_10TB/10TB/ghorbanpoor/vit-base-patch16-224'
    image_model_name = 'google/vit-base-patch16-224'
    image_embedding = 768
    # the public pretrained model parameters
    text_encoder_model = "bert-base-uncased"
    # text_encoder_model = "/home/faeze/PycharmProjects/new_fake_news_detectioin/bert/bert-base-uncased"
    # text_tokenizer = "../../../../../media/external_10TB/10TB/ghorbanpoor/bert-base-uncased"
    text_tokenizer = "bert-base-uncased"
    # text_tokenizer = "/home/faeze/PycharmProjects/new_fake_news_detectioin/bert/bert-base-uncased"
    text_embedding = 768
    max_length = 32

    pretrained = True
    trainable = False
    temperature = 1.0

    classes = [1, 0]
    # classes = ['real', 'fake']
    class_weights = [1, 1]

    wanted_accuracy = 0.85

    def optuna(self, trial):
        self.head_lr = trial.suggest_loguniform('head_lr', 1e-5, 1e-1)
        self.image_encoder_lr = trial.suggest_loguniform('image_encoder_lr', 1e-6, 1e-3)
        self.text_encoder_lr = trial.suggest_loguniform('text_encoder_lr', 1e-6, 1e-3)
        self.classification_lr = trial.suggest_loguniform('classification_lr', 1e-5, 1e-1)

        self.head_weight_decay = trial.suggest_loguniform('head_weight_decay', 1e-5, 1e-1)
        # self.attention_weight_decay = trial.suggest_loguniform('attention_weight_decay', 1e-5, 1e-1)
        self.classification_weight_decay = trial.suggest_loguniform('classification_weight_decay', 1e-5, 1e-1)

        # self.projection_size = trial.suggest_categorical('projection_size', [256, 128, 64])
        # self.hidden_size = trial.suggest_categorical('hidden_size', [256, 128, 64, ])
        # self.dropout = trial.suggest_categorical('drop_out', [0.1, 0.3, 0.5, ])

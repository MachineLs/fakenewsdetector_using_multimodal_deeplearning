import torch
from transformers import BertTokenizer, BertModel, BertConfig

from data.config import Config
from data.weibo.data_loader import WeiboDatasetLoader


class WeiboConfig(Config):
    name = 'weibo'
    DatasetLoader = WeiboDatasetLoader

    data_path = '../../../../../media/external_3TB/3TB/ghorbanpoor/weibo/'
    # data_path = '/home/faeze/PycharmProjects/fake_news_detection/data/weibo/'
    output_path = '../../../../../media/external_10TB/10TB/ghorbanpoor/'
    # output_path = ''

    rumor_image_path = data_path + 'rumor_images/'
    nonrumor_image_path = data_path + 'nonrumor_images/'

    train_text_path = data_path + 'weibo_train.csv'
    validation_text_path = data_path + 'weibo_train.csv'
    test_text_path = data_path + 'weibo_test.csv'

    batch_size = 64
    epochs = 100
    num_workers = 4
    head_lr = 1e-03
    image_encoder_lr = 1e-02
    text_encoder_lr = 1e-05
    weight_decay = 0.001
    classification_lr = 1e-02

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    image_model_name = '../../../../../media/external_10TB/10TB/ghorbanpoor/vit-base-patch16-224'
    image_embedding = 768
    text_encoder_model = "../../../../../media/external_10TB/10TB/ghorbanpoor/bert-base-uncased"
    # text_encoder_model = "/home/faeze/PycharmProjects/new_fake_news_detectioin/bert/bert-base-uncased"
    text_tokenizer = "../../../../../media/external_10TB/10TB/ghorbanpoor/bert-base-uncased"
    # text_tokenizer = "/home/faeze/PycharmProjects/new_fake_news_detectioin/bert/bert-base-uncased"
    text_embedding = 768
    max_length = 200

    pretrained = True
    trainable = False
    temperature = 1.0

    labels = ['real', 'fake']

    wanted_accuracy = 0.80

    def optuna(self, trial):
        self.head_lr = trial.suggest_loguniform('head_lr', 1e-5, 1e-1)
        self.image_encoder_lr = trial.suggest_loguniform('image_encoder_lr', 1e-6, 1e-3)
        self.text_encoder_lr = trial.suggest_loguniform('text_encoder_lr', 1e-6, 1e-3)
        self.classification_lr = trial.suggest_loguniform('classification_lr', 1e-5, 1e-1)

        self.head_weight_decay = trial.suggest_loguniform('head_weight_decay', 1e-5, 1e-1)
        self.classification_weight_decay = trial.suggest_loguniform('classification_weight_decay', 1e-5, 1e-1)

        self.projection_size = trial.suggest_categorical('projection_size', [256, 128, 64])
        self.hidden_size = trial.suggest_categorical('hidden_size', [256, 128, 64, ])
        self.dropout = trial.suggest_categorical('drop_out', [0.1, 0.3, 0.5, ])


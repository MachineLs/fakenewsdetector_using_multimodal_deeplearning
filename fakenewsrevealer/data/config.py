import torch
import ast


class Config:
    name = ''
    DatasetLoader = None
    debug = False
    data_path = ''
    output_path = ''

    train_image_path = ''
    validation_image_path = ''
    test_image_path = ''

    train_text_path = ''
    validation_text_path = ''
    test_text_path = ''

    train_text_embedding_path = ''
    validation_text_embedding_path = ''

    train_image_embedding_path = ''
    validation_image_embedding_path = ''

    batch_size = 256
    epochs = 50
    num_workers = 2
    head_lr = 1e-03
    image_encoder_lr = 1e-04
    text_encoder_lr = 1e-04
    attention_lr = 1e-3
    classification_lr = 1e-03

    max_grad_norm = 5.0

    head_weight_decay = 0.001
    attention_weight_decay = 0.001
    classification_weight_decay = 0.001

    patience = 30
    delta = 0.0000001
    factor = 0.8

    image_model_name = '../../../../../media/external_10TB/10TB/ghorbanpoor/vit-base-patch16-224'
    image_embedding = 768
    num_img_region = 64  # 16 #TODO
    text_encoder_model = "../../../../../media/external_10TB/10TB/ghorbanpoor/bert-base-uncased"
    text_tokenizer = "../../../../../media/external_10TB/10TB/ghorbanpoor/bert-base-uncased"

    text_embedding = 768
    max_length = 32

    pretrained = True  # for both image encoder and text encoder
    trainable = False  # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    num_projection_layers = 1
    projection_size = 64
    dropout = 0.3
    hidden_size = 128
    num_region = 64  # 16 #TODO
    region_size = projection_size // num_region

    classes = ['real', 'fake']
    class_num = 2

    loss_weight = 1
    class_weights = [1, 1]

    writer = None

    has_unlabeled_data = False
    step = 0
    T1 = 10
    T2 = 150
    af = 3

    wanted_accuracy = 0.76

    def optuna(self, trial):
        self.head_lr = trial.suggest_loguniform('head_lr', 1e-5, 1e-1)
        self.image_encoder_lr = trial.suggest_loguniform('image_encoder_lr', 1e-6, 1e-3)
        self.text_encoder_lr = trial.suggest_loguniform('text_encoder_lr', 1e-6, 1e-3)
        self.classification_lr = trial.suggest_loguniform('classification_lr', 1e-5, 1e-1)
        self.attention_lr = trial.suggest_loguniform('attention_lr', 1e-5, 1e-1)

        self.attention_weight_decay = trial.suggest_loguniform('attention_weight_decay', 1e-5, 1e-1)
        self.head_weight_decay = trial.suggest_loguniform('head_weight_decay', 1e-5, 1e-1)
        self.classification_weight_decay = trial.suggest_loguniform('classification_weight_decay', 1e-5, 1e-1)

        self.projection_size = trial.suggest_categorical('projection_size', [256, 128, 64])
        self.hidden_size = trial.suggest_categorical('hidden_size', [256, 128, 64, ])
        self.dropout = trial.suggest_categorical('drop_out', [0.1, 0.3, 0.5, ])

    def __str__(self):
        s = ''
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for member in members:
            s += member + '\t' + str(getattr(self, member)) + '\n'
        return s

    def assign_hyperparameters(self, s):
        for line in s.split('\n'):
            s = line.split('\t')
            try:
                attr = getattr(self, s[0])
                if type(attr) not in [list, set, dict]:
                    setattr(self, s[0], type(attr)(s[1]))
                else:
                    setattr(self, s[0], ast.literal_eval(s[1]))

            except:
                print(s[0], 'doesnot exist')

import torch
import torch.nn.functional as F
from torch import nn
from keras import backend as K
import tensorflow as tf
import numpy as np

from image import ImageTransformerEncoder, ImageResnetEncoder
from text import TextEncoder


class ProjectionHead(nn.Module):
    def __init__(
            self,
            config,
            embedding_dim,
    ):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(embedding_dim, config.projection_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(config.projection_size, config.projection_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.projection_size)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(2 * config.projection_size)
        self.linear_layer = nn.Linear(2 * config.projection_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(config.dropout)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.classifier_layer = nn.Linear(config.hidden_size, config.class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.linear_layer(x)
        x = self.gelu(x)
        x = self.drop_out(x)
        self.embeddings = x = self.layer_norm_2(x)
        x = self.classifier_layer(x)
        x = self.softmax(x)
        return x


class FakeNewsModel(nn.Module):
    def __init__(
            self, config
    ):
        super().__init__()
        self.config = config
        if 'resnet' in self.config.image_model_name:
            self.image_encoder = ImageResnetEncoder(config)
        else:
            self.image_encoder = ImageTransformerEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.image_projection = ProjectionHead(config, embedding_dim=config.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=config.text_embedding)
        self.classifier = Classifier(config)
        class_weights = torch.FloatTensor(config.class_weights)
        self.classifier_loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

        self.text_embeddings = None
        self.image_embeddings = None
        self.multimodal_embeddings = None

    def forward(self, batch):
        image_features = self.image_encoder(ids=batch['id'], image=batch["image"])
        text_features = self.text_encoder(ids=batch['id'],
                                          input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        self.image_embeddings = self.image_projection(image_features)
        self.text_embeddings = self.text_projection(text_features)

        self.multimodal_embeddings = torch.cat((self.image_embeddings, self.text_embeddings), dim=1)
        self.logits = (self.text_embeddings @ self.image_embeddings.T)

        score = self.classifier(self.multimodal_embeddings)
        probs, output = torch.max(score.data, dim=1)

        return output, score



def calculate_loss(model, score, label):
    similarity = calculate_similarity_loss(model.config, model.image_embeddings, model.text_embeddings)
    # fake = (label == 1).nonzero()
    # real = (label == 0).nonzero()
    # s_loss = 0 * similarity[fake].mean() + similarity[real].mean()
    c_loss = model.classifier_loss_function(score, label)
    loss = model.config.loss_weight * c_loss + similarity.mean()
    return loss


def calculate_similarity_loss(config, image_embeddings, text_embeddings):
    # Calculating the Loss
    logits = (text_embeddings @ image_embeddings.T)
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (images_loss + texts_loss) / 2.0
    return loss


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

################### our method
def mean_square_error(y_true, y_pred):
    return K.mean(K.abs(y_true-y_pred), axis=-1)


def Hinge(yHat, y):
    #print(yHat, y)
    return np.max([0., 1. - yHat * y])


################################################
def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
    return loss


def lsr(y_true, y_pred):
    epsilon = 0.1
    y_smoothed_true = y_true * (1 - epsilon - epsilon / 10.0)
    y_smoothed_true = y_smoothed_true + epsilon / 10.0

    y_pred_1 = tf.clip_by_value(y_pred, 1e-7, 1.0)

    return tf.reduce_mean(-tf.reduce_sum(y_smoothed_true * tf.log(y_pred_1), axis=-1))

def generalized_cross_entropy(y_true, y_pred):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    q = 0.7
    t_loss = (1 - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1), q)) / q
    return tf.reduce_mean(t_loss)

def joint_optimization_loss(y_true, y_pred):
    """
    2018 - cvpr - Joint optimization framework for learning with noisy labels.
    """
    y_pred_avg = K.mean(y_pred, axis=0)
    p = np.ones(10, dtype=np.float32) / 10.
    l_p = - K.sum(K.log(y_pred_avg) * p)
    l_e = K.categorical_crossentropy(y_pred, y_pred)
    return K.categorical_crossentropy(y_true, y_pred) + 1.2 * l_p + 0.8 * l_e


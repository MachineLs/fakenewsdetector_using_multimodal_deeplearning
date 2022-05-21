import random

import numpy as np
import torch
from tqdm import tqdm

from data_loaders import make_dfs, build_loaders
from evaluation import metrics, report_per_class, roc_auc_plot, precision_recall_plot, plot_tsne, plot_pca
from learner import batch_constructor
from model import FakeNewsModel


def test(config, test_loader, trial_number=None):
    if trial_number:
        try:
            checkpoint = torch.load(str(config.output_path) + '/checkpoint_' + str(trial_number) + '.pt')
        except:
            checkpoint = torch.load(str(config.output_path) + '/checkpoint.pt')
    else:
        checkpoint = torch.load(str(config.output_path) + '/checkpoint.pt', map_location=torch.device(config.device))

    try:
        parameters = checkpoint['parameters']
        config.assign_hyperparameters(parameters)
    except:
        pass

    model = FakeNewsModel(config).to(config.device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(checkpoint)

    model.eval()

    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)

    image_features = []
    text_features = []
    multimodal_features = []
    concat_features = []

    targets = []
    predictions = []
    scores = []
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for i, batch in enumerate(tqdm_object):
        batch = batch_constructor(config, batch)
        with torch.no_grad():
            output, score = model(batch)

            prediction = output.detach()
            predictions.append(prediction)

            score = score.detach()
            scores.append(score)

            target = batch['label'].detach()
            targets.append(target)

            image_feature = model.image_embeddings.detach()
            image_features.append(image_feature)

            text_feature = model.text_embeddings.detach()
            text_features.append(text_feature)

            multimodal_feature = model.multimodal_embeddings.detach()
            multimodal_features.append(multimodal_feature)

            concat_feature = model.classifier.embeddings.detach()
            concat_features.append(concat_feature)

    # config.writer.add_graph(model, input_to_model=batch, verbose=True)

    s = ''
    s += report_per_class(targets, predictions) + '\n'
    s += metrics(targets, predictions, scores, file_path=str(config.output_path) + '/fpr_tpr.csv') + '\n'
    with open(config.output_path + '/results.txt', 'w') as f:
        f.write(s)

    roc_auc_plot(targets, scores, fname=str(config.output_path) + "/roc.png")
    precision_recall_plot(targets, scores, fname=str(config.output_path) + "/pr.png")

    # saving_in_tensorboard(config, image_features, targets, 'image_features')
    plot_tsne(config, image_features, targets, fname=str(config.output_path) + '/image_features_tsne.png')
    plot_pca(config, image_features, targets, fname=str(config.output_path) + '/image_features_pca.png')

    # saving_in_tensorboard(config, text_features, targets, 'text_features')
    plot_tsne(config, text_features, targets, fname=str(config.output_path) + '/text_features_tsne.png')
    plot_pca(config, text_features, targets, fname=str(config.output_path) + '/text_features_pca.png')
    #
    # saving_in_tensorboard(config, multimodal_features, targets, 'multimodal_features')
    plot_tsne(config, multimodal_features, targets, fname=str(config.output_path) + '/multimodal_features_tsne.png')
    plot_pca(config, multimodal_features, targets, fname=str(config.output_path) + '/multimodal_features_pca.png')

    # saving_in_tensorboard(config, concat_features, targets, 'concat_features')
    plot_tsne(config, concat_features, targets, fname=str(config.output_path) + '/concat_features_tsne.png')
    plot_pca(config, concat_features, targets, fname=str(config.output_path) + '/concat_features_pca.png')

    config_parameters = str(config)
    with open(config.output_path + '/parameters.txt', 'w') as f:
        f.write(config_parameters)
    print(config)


def test_main(config, trial_number=None):
    train_df, test_df, validation_df = make_dfs(config, )
    test_loader = build_loaders(config, test_df, mode="test")
    test(config, test_loader, trial_number)

import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_loaders import make_dfs, build_loaders
from learner import batch_constructor
from model import FakeNewsModel

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from lime.lime_text import LimeTextExplainer

from utils import make_directory


def lime_(config, test_loader, model):
    for i, batch in enumerate(test_loader):
        batch = batch_constructor(config, batch)
        with torch.no_grad():
            output, score = model(batch)

            score = score.detach().cpu().numpy()

            logit = model.logits.detach().cpu().numpy()

    return score, logit


def lime_main(config, trial_number=None):
    _, test_df, _ = make_dfs(config, )
    test_df = test_df[:1]

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

    text_explainer = LimeTextExplainer(class_names=config.classes)
    image_explainer = LimeImageExplainer()

    make_directory(config.output_path + '/lime/')
    make_directory(config.output_path + '/lime/score/')
    make_directory(config.output_path + '/lime/logit/')
    make_directory(config.output_path + '/lime/text/')
    make_directory(config.output_path + '/lime/image/')

    def text_predict_proba(text):
        scores = []
        print(len(text))
        for i in text:
            row['text'] = i
            test_loader = build_loaders(config, row, mode="lime")
            score, _ = lime_(config, test_loader, model)
            scores.append(score)
        return np.array(scores)

    def image_predict_proba(image):
        scores = []
        print(len(image))
        for i in image:
            test_loader = build_loaders(config, row, mode="lime")
            test_loader['image'] = i.reshape((3, 224, 224))
            score, _ = lime_(config, test_loader, model)
            scores.append(score)
        return np.array(scores)

    for i, row in test_df.iterrows():
        test_loader = build_loaders(config, row, mode="lime")
        score, logit = lime_(config, test_loader, model)
        np.savetxt(config.output_path + '/lime/score/' + str(i) + '.csv', score, delimiter=",")
        np.savetxt(config.output_path + '/lime/logit/' + str(i) + '.csv', logit, delimiter=",")

        text_exp = text_explainer.explain_instance(row['text'], text_predict_proba, num_features=5)
        text_exp.save_to_file(config.output_path + '/lime/text/' + str(i) + '.html')
        print('text', i, 'finished')

        data_items = config.DatasetLoader(config, dataframe=row, mode='lime').__getitem__(0)
        img_exp = image_explainer.explain_instance(data_items['image'].reshape((224, 224, 3)), image_predict_proba,
                                                   top_labels=2, hide_color=0, num_samples=1)
        temp, mask = img_exp.get_image_and_mask(img_exp.top_labels[0], positive_only=False, num_features=5,
                                                hide_rest=False)
        img_boundry = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry)
        plt.savefig(config.output_path + '/lime/image/' + str(i) + '.png')
        print('image', i, 'finished')

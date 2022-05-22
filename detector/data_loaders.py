import numpy as np
import pandas as pd

from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader


def make_dfs(config):
    train_dataframe = pd.read_csv(config.train_text_path)
    train_dataframe.dropna(subset=['text'], inplace=True)
    train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
    train_dataframe.label = train_dataframe.label.apply(lambda x: config.classes.index(x))
    config.class_weights = get_class_weights(train_dataframe.label.values)

    if config.test_text_path is None:
        offset = int(train_dataframe.shape[0] * 0.80)
        test_dataframe = train_dataframe[offset:]
        train_dataframe = train_dataframe[:offset]
    else:
        test_dataframe = pd.read_csv(config.test_text_path)
        test_dataframe.dropna(subset=['text'], inplace=True)
        test_dataframe = test_dataframe.sample(frac=1).reset_index(drop=True)
        test_dataframe.label = test_dataframe.label.apply(lambda x: config.classes.index(x))

    if config.validation_text_path is None:
        offset = int(train_dataframe.shape[0] * 0.90)
        validation_dataframe = train_dataframe[offset:]
        train_dataframe = train_dataframe[:offset]
    else:
        validation_dataframe = pd.read_csv(config.validation_text_path)
        validation_dataframe.dropna(subset=['text'], inplace=True)
        validation_dataframe = validation_dataframe.sample(frac=1).reset_index(drop=True)
        validation_dataframe.label = validation_dataframe.label.apply(lambda x: config.classes.index(x))

    return train_dataframe, test_dataframe, validation_dataframe


def build_loaders(config, dataframe, mode):
    print('-------------------------------------------------------')
    print(dataframe)
    dataset = config.DatasetLoader(config, dataframe=dataframe, mode=mode)
    if mode != 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size // 2 if mode == 'lime' else 1,
            num_workers=config.num_workers // 2,
            pin_memory=False,
            shuffle=False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    return dataloader


def get_class_weights(y):
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    return class_weights


from data_loaders import build_loaders, make_dfs
from learner import supervised_train
from test_main import test


def torch_main(config):
    train_df, test_df, validation_df = make_dfs(config, )
    train_loader = build_loaders(config, train_df, mode="train")
    validation_loader = build_loaders(config, validation_df, mode="validation")
    test_loader = build_loaders(config, test_df, mode="test")

    supervised_train(config, train_loader, validation_loader)
    test(config, test_loader)






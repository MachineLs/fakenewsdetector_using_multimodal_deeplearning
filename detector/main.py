import argparse

from lime_main import lime_main
from optuna_main import optuna_main
from test_main import test_main
from torch_main import torch_main
from data.reddit.config import RedditConfig
from torch.utils.tensorboard import SummaryWriter

from utils import make_directory

if __name__ == '__main__':
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--use_optuna', type=int, required=False)
    parser.add_argument('--use_lime', type=int, required=False)
    parser.add_argument('--just_test', type=int, required=False)
    parser.add_argument('--batch', type=int, required=False)
    parser.add_argument('--epoch', type=int, required=False)
    parser.add_argument('--extra', type=str, required=False)

    args = parser.parse_args()

    if args.data == 'reddit':
        config = RedditConfig()
    else:
        raise Exception('Enter a valid dataset name', args.data)

    if args.batch:
        config.batch_size = args.batch
    if args.epoch:
        config.epochs = args.epoch

    if args.use_optuna:
        config.output_path += 'logs/' + args.data + '_optuna' + '_' + str(args.extra)
    else:
        config.output_path += 'logs/' + args.data + '_' + str(args.extra)

    use_optuna = True
    if not args.extra or 'temp' in args.extra:
        config.output_path = str(args.extra)
        use_optuna = False

    make_directory(config.output_path)

    config.writer = SummaryWriter(config.output_path)

    if args.use_optuna and use_optuna:
        optuna_main(config, args.use_optuna)
    elif args.just_test:
        test_main(config, args.just_test)
    elif args.use_lime:
        lime_main(config, args.use_lime)
    else:
        torch_main(config)

    config.writer.close()

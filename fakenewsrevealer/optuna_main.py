

import joblib
import optuna
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from data_loaders import make_dfs, build_loaders
from learner import supervised_train
from test_main import test


def objective(trial, config, train_loader, validation_loader):
    config.optuna(trial=trial)
    print('Trial', trial.number, 'parameters', trial.params)
    accuracy = supervised_train(config, train_loader, validation_loader, trial=trial)
    return accuracy


def optuna_main(config, n_trials=100):
    train_df, test_df, validation_df = make_dfs(config)
    train_loader = build_loaders(config, train_df, mode="train")
    validation_loader = build_loaders(config, validation_df, mode="validation")
    test_loader = build_loaders(config, test_df, mode="test")

    study = optuna.create_study(study_name=config.output_path.split('/')[-1],
                                sampler=optuna.samplers.TPESampler(),
                                storage=f'sqlite:///{config.output_path + "/optuna.db"}',
                                load_if_exists=True,
                                direction="maximize",
                                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10)
                                )
    study.optimize(lambda trial: objective(trial, config, train_loader, validation_loader), n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    joblib.dump(study, str(config.output_path) + '/study_optuna_model' + '.pkl')

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    s = ''
    print("Best trial:")
    trial = study.best_trial
    print(' Number: ', trial.number)
    print("  Value: ", trial.value)
    s += 'number: ' + str(trial.number) + '\n'
    s += 'value: ' + str(trial.value) + '\n'

    print("  Params: ")
    s += 'params: \n'
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        s += "    {}: {}\n".format(key, value)

    with open(config.output_path+'/optuna_results.txt', 'w') as f:
        f.write(s)

    test(config, test_loader, trial_number=trial.number)


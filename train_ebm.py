# main script that launches a training run
from training.trainer import Trainer
import argparse
import utils
import wandb


def launch_training_run(cfg):
    print('PRINTING CONFIG\n\n')
    print(cfg)
    print('\n\n\n\n')
    trainer = Trainer(cfg)
    trainer.run_train_loop()
    # trainer.launch_simulation_run()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config and hyperparameters')
    parser.add_argument('cfg', type=str, help='config file path')
    # any argument given as --kwarg=x after the config file will be parsed
    # and added to the config dict or overwrite the parameters in the config dict if they are already present

    args, remaining_args = parser.parse_known_args()
    cfg = utils.load_config(args.cfg, remaining_args)
    launch_training_run(cfg)

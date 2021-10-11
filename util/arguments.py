import argparse
from datetime import datetime
from pathlib import Path
from random import randint
import os
from config import config_handler


def generate_experiment_name(config):
    if not os.environ.get('experiment'):
        config['experiment'] = f"{datetime.now().strftime('%d%m%H%M')}_{config['task']}_{config['dataset_train']['dataset_name']}_{config['experiment']}"
        if config['resume'] is not None and not config['new_exp_for_resume']:
            config['experiment'] = Path(config['resume']).parents[0].name
        os.environ['experiment'] = config['experiment']
    else:
        config['experiment'] = os.environ['experiment']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config path')
    parser.add_argument('--sanity_steps', type=int, default=0, help='sanity_steps')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--new_exp_for_resume', action='store_true', help='create new experiment for resume')
    parser.add_argument('--val_check_percent', type=float, default=1.0, help='percentage of val checked')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='check val every fraction of epoch')
    parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
    parser.add_argument('--experiment', type=str, default='fast_dev', help='experiment directory')
    parser.add_argument('--suffix', type=str, default='', help='wandb project suffix')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--current_phase', type=int, default=0, help='current phase')
    parser.add_argument('--phase_change_epochs', type=int, nargs='+', default=[30, 25, 5], help='phases')
    parser.add_argument('--wandb_main', action='store_true')
    parser.add_argument('--no_retrievals', action='store_true')
    parser.add_argument('--retrieval_ckpt', type=str, default=None)
    parser.add_argument('--unet_backbone_decoder_ckpt', type=str, default=None)
    parser.add_argument('--retrieval_backbone_ckpt', type=str, default=None)
    parser.add_argument('--attention_block_ckpt', type=str, default=None)
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = randint(0, 999)

    if args.val_check_interval > 1:
        args.val_check_interval = int(args.val_check_interval)

    if not args.wandb_main and args.suffix == '':
        args.suffix = '-dev'

    config = config_handler.read_config(args.config, args)
    print('config parsed')
    generate_experiment_name(config)

    return config

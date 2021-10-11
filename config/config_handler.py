from pathlib import Path
import yaml


def read_config(path, args):
    _config = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)
    config = {}
    if "inherit_from" in _config:
        config = yaml.load(Path(f'config/{_config["inherit_from"]}').read_text(), Loader=yaml.FullLoader)
    update_recursive(config, _config)
    update_dataset_configs(config)
    if 'dataset' in config:
        del config['dataset']
    if args:
        override_config_with_args(config, args)
    return config


def override_config_with_args(config, args):
    var_args = vars(args)
    for k in var_args:
        if (k not in config) or (var_args[k] is not None and var_args[k] != -100):
            config[k] = var_args[k]


def update_dataset_configs(config):
    if 'dataset' in config:
        for c in config['dataset']:
            for d in ['dataset_train', 'dataset_val']:
                if c not in config[d]:
                    config[d][c] = config['dataset'][c]


def update_recursive(dict1, dict2):
    # borrowed from autonomousvision
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

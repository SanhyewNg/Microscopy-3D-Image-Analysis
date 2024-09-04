import collections
import copy
import functools as ft
import os

import yaml
from attrdict import AttrDict


def yaml_file(filepath):
    """
    Return the path to the corresponding yaml file
    Args:
        filepath: path to file like h5, pkl which has corresponding yaml

    Returns:
        path to the corresponding yaml file
    """
    return os.path.splitext(filepath)[0] + ".yaml"


def save_args_to_yaml(output_path, args):
    """Save program arguments into yaml format.

    Args:
        output_path: path to the output yaml file (without extension)
        args: argparse.Namespace() object or dictionary or AttrDict with parameters the program has
              been called with
    """
    if not output_path.endswith('.yaml'):
        output_path = output_path + '.yaml'

    # Convert to simple dictionary so file is clean.
    if isinstance(args, AttrDict) or isinstance(args, dict):
        args_dict = dict(args)
    else:
        args_dict = vars(args)
    with open(output_path, "w") as f:
        yaml.dump(args_dict, f)


def load_yaml_args(config_path):
    """Load yaml configuration file.

    Args:
        config_path: path to yaml file with dictionary containing training
                     arguments in dict format.
    Returns:
        training dictionary in AttrDict format - such approach works
        seamlessly with the rest of the application.
    """
    with open(config_path, 'r') as f:
        return AttrDict(yaml.load(f))


def merge_cli_to_yaml(config_path, cli_args, params_to_merge):
    """Load and merge yaml config with CLI arguments.

    Information from CLI overwrites config (if same keys exist).

    Args:
        config_path: path to yaml config file
        cli_args (dict): object with arguments from CLI
        params_to_merge: if not None then only the provided list of parameters is merged
    Returns:
        config updated with values from CLI
    """
    config = load_yaml_args(config_path)

    if params_to_merge is not None:
        cli_args = {k: cli_args[k] for k in params_to_merge}

    # Update config with args.
    config.update(cli_args)

    return config


def merge_yaml_to_cli(config_path, cli_args, params_to_merge):
    """Load and merge yaml config with CLI arguments.

    Information from yaml config overwrites args (if same keys exist).

    If specific key doesn't exist in args, it will be automatically added.
    Note that whole operation with getting args as dict operates in place - that's why it's been decided to work on copy of it.
    Args:
        config_path: path to yaml config file
        cli_args (dict): object with arguments from CLI
        params_to_merge: if not None then only the provided list of parameters is merged
    Returns:
        args_copy: args with information from config_path
    """

    args_dict = copy.deepcopy(cli_args)
    config = load_yaml_args(config_path)

    # Below loop modifies args in-place.
    for key, value in config.items():
        if params_to_merge is None or key in params_to_merge:
            args_dict[key] = value

    return AttrDict(args_dict)


def update_args(args, **kwargs):
    """Update dictionary with given arguments if they are not None.

    Args:
        args (dict): Arguments read before.
        **kwargs: Arguments updating `args` if they are not None.
    """
    updating_args = {name: value
                     for name, value in kwargs.items() if value is not None}
    args.update(updating_args)


class ArgumentNameError(Exception):
    """Raised when name used by decorator is already function parameter."""


def load_args(arg_name):
    """Load arguments from .yaml file before executing wrapped function.

    It works only with functions that take keyword-only arguments. Note that if it's used
    together with `save_args` decorator they should be used it following order:

    @load_args(name)
    @save_args(name)
    def f():
        ...

    Otherwise `save_args` won't save loaded arguments, because they will be loaded later.

    Args:
        arg_name (str): Name of the argument used for passing path to file with
                        arguments. It can't be used as an ordinary function argument.
    """
    def wrapper(f):
        @ft.wraps(f)
        def inner_wrapper(**given_args):
            load_path = given_args.pop(arg_name, None)
            all_args = given_args
            if load_path is not None:
                yaml_args = load_yaml_args(load_path)
                yaml_args.update(given_args)
                all_args = yaml_args

            return f(**all_args)

        return inner_wrapper

    return wrapper


def save_args(arg_name, ignored_args=frozenset()):
    """Save arguments to .yaml file before executing wrapped function.

    It works only with functions that take keyword-only arguments. Note that if it's used
    together with `load_args` decorator they should be used it following order:

    @load_args(name)
    @save_args(name)
    def f():
        ...

    Otherwise `save_args` won't save loaded arguments, because they will be loaded later.

    Args:
        arg_name (str): Name of the argument used for passing path to file with
                        arguments. It can't be used as an ordinary function argument.
        ignored_args (set): Parameters that shouldn't be saved.
    """
    def wrapper(f):
        @ft.wraps(f)
        def inner_wrapper(**given_args):
            default_args = f.__kwdefaults__
            if default_args is None:
                default_args = {}

            save_path = given_args.pop(arg_name, None)

            if save_path is not None:
                all_args = {**default_args, **given_args}
                args_to_save = {name: value
                                for name, value in all_args.items()
                                if name not in ignored_args}

                save_args_to_yaml(save_path, args_to_save)

            return f(**given_args)

        return inner_wrapper

    return wrapper


def load_and_save_args(f):
    """Add possibility to save and load function arguments from .yaml file.
    When f is decorated function, let's say:
    @save_and_load_args
    def f(*, a=1, b=2, c=3)
       ...
    in addition to 'a', 'b' and 'c' keyword arguments you can also pass
    'args_load_path' and 'args_save_path' to function. First one will load
    arguments from .yaml file, so if it contains all other parameters you don't
    have to pass them. Second one will save parameters to given location. Only
    explicit pass of the arguments has a priority before .yaml, so default
    arguments won't be used if there is a .yaml file.
    Args:
        f (Callable): Decorated function, it is assumed to use only keyword
                      arguments. Also it shouldn't use 'args_load_path' or
                      'args_save_path' arguments.
    Returns:
        Callable: Wrapped f with added 'args_load_path' and 'args_save_path'
                  arguments.
    Raises:
        ArgumentNameError: When 'args_load_path' or 'args_save_path' is used
                           as decorated function parameter.
    """

    @ft.wraps(f)
    def wrapper(args_load_path=None, args_save_path=None, **kwargs):
        # kwargs doesn't include default arguments here, so I add them.
        default_args = f.__kwdefaults__
        if default_args is None:
            default_args = {}
        given_args = kwargs
        all_args = dict(collections.ChainMap(given_args, default_args))

        if args_load_path is not None:
            yaml_args = load_yaml_args(args_load_path)
            yaml_args.update(given_args)
            all_args = yaml_args

        if 'args_load_path' in all_args:
            raise ArgumentNameError(
                "'args_load_path' is already used as function parameter.")
        if 'args_save_path' in all_args:
            raise ArgumentNameError(
                "'args_save_path' is already used as function parameter.")

        if args_save_path is not None:
            dirname = os.path.dirname(os.path.abspath(args_save_path))
            os.makedirs(dirname, exist_ok=True)
            save_args_to_yaml(args_save_path, all_args)

        f(**all_args)

    return wrapper

"""Make a Python script suitable for a Slurm array.
"""

import itertools
import collections
import subprocess
import os

class ArgparseArray:
    """Generate array of argparse arguments to call using a slurm-type cluster.
    You can provide three types of named arguments:
        - List: This list is taken to specify the values over which the named argument should be iterated.
        - Function: This is taken to specify the value of the named argument in dependence on specific values for the
            arguments from the array of arguments.
        - Auxiliary arguments: are only used to spread the array. They should be preceded by 'aux_'.
        - Value: Anything else is taken to be a simple value for this script. If you wish to make sure that something is
            taken to be this (say, because it is a list), you should precede the argument name by 'niarg_'. If the script
            contains positional arguments you should name them 'posarg', followed by a suffix that allows them to be sorted
            in the right manner. So if you have three positional arguments, you could name them posarg0, posarg1, and
            posarg2.
    If you have a flag, you can turn this flag an by setting the argument to True, and you can turn it off by setting it to False.
    """
    def __init__(self, **kwargs):
        self.base_args = {}
        for key, value in kwargs.items():
            if not isinstance(value, (list, collections.abc.Callable)) or key[:5] == 'niarg':
                if key[:5] == 'niarg':
                    key = key[6:]
                self.base_args[key] = value
        self.array_args = {}
        self.aux_keys = []
        for key, value in kwargs.items():
            if isinstance(value, (list,)) and key[:5] != 'niarg':
                if key[:3] == 'aux':
                    key = key[4:]
                    self.aux_keys.append(key)
                self.array_args[key] = value
        self.callable_args = {
            key: value for key, value in kwargs.items() if isinstance(value, (collections.abc.Callable,)) and key[:5] != 'niarg'
        }

    def get_args(self, array_id):
        values = list(itertools.product(*self.array_args.values()))[array_id]
        id_args = {
            key: value for key, value in zip(self.array_args.keys(), values)
        }
        called_args = {
            key: value(array_id=array_id, **id_args) for key, value in self.callable_args.items()
        }
        called_id_args = {
            key: value for key, value in id_args.items() if key not in self.aux_keys
        }
        args = {**called_id_args, **called_args, **self.base_args}
        return args

    def call_script(self, script, array_id, python_cmd='python'):
        args = self.get_args(array_id)
        str_args = [python_cmd, script]
        positional_keys = [key for key in args.keys() if key[:6]=='posarg']
        positional_keys.sort()
        positional_args = [args[key] for key in positional_keys]
        str_args = str_args + positional_args
        for key, value in args.items():
            if isinstance(value, (bool,)):
                if value:
                    str_args.append('--{}'.format(key,))
            else:
                if key[:6] != 'posarg':
                    if isinstance(value, (list,)):
                        str_args.append('--{}'.format(key))
                        for val in value:
                            str_args.append(str(val))
                    else:
                        str_args.append('--{}'.format(key,))
                        str_args.append(str(value))
        subprocess.run(str_args)

def name_instance(*keys, slash_replacement='>', separator='--', base_folder=''):
    def fun(**kwargs):
        name = [f'{key}={value}' for key, value in kwargs.items() if key in keys]
        name = separator.join(name)
        name = name.replace('/', slash_replacement)
        return os.path.join(base_folder, name)
    return fun

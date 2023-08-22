"""Train DNNs at different scales and track their predictions throughout time.
"""

import argparse
import sys
import os
import re

sys.path.append('')

import numpy as np

from python_functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    folder=name_instance('model_seed', 'scaling', 'mode', 'task', base_folder='data-raw/simulations/14_crossentropy-tp'),
    criterion='crossentropy',
    epochs=int(1e6),
    hdims=10000,
    lr=(lambda array_id, mode, scaling, **kwargs: {
        ('linear_readout', 1e-6): 1e4,
        ('backprop', 1e-6): 1e6,
        ('backprop', 1.): 1e-3,
        ('linear_readout', 1.): 1.
    }[(mode, scaling)]),
    model_seed=list(range(20)),
    scaling=[1e-6, 1.],
    test_at_end_only=False,
    task=['transverse', 'banded_transverse'],
    mode=['backprop', 'linear_readout'],
    niarg_tested_epochs=list(range(99))+[10**i-1 for i in range(2, 7)],
    n=7,
    test_every_n_epochs=1,
    no_early_stopping=True
)

def main(args):
    argparse_array.call_script('train.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
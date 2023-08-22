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
    folder=name_instance('depth', 'model_seed', 'task', base_folder='data-raw/simulations/13_depth-tp'),
    criterion='mse',
    epochs=int(1e6),
    hdims=(lambda array_id, depth, **kwargs: [1000]*depth),
    lr=1e37,
    model_seed=list(range(20)),
    scaling=(lambda array_id, depth, **kwargs: 10**(-3*(depth+1))),
    test_at_end_only=True,
    task=['transverse', 'banded_transverse'],
    aux_depth=[1,2,3],
    n=7
)

def main(args):
    argparse_array.call_script('train.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
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
    folder=name_instance('n', 'model_seed', base_folder='data-raw/simulations/07_n-ti'),
    criterion='mse',
    epochs=int(1e6),
    hdims=10000,
    lr=1e37,
    model_seed=list(range(20)),
    scaling=1e-32,
    test_at_end_only=True,
    task='transitive',
    n=[5,7,9,11,13,15]
)

def main(args):
    argparse_array.call_script('train.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
"""Train DNNs at different scales and track their predictions throughout time.
"""

import argparse
import sys
import os
import re

import numpy as np

sys.path.append('')

from python_functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    folder=name_instance('scaling', 'model_seed', base_folder='data-raw/simulations/03_ti-scaling'),
    criterion='mse',
    epochs=int(1e6),
    hdims=50000,
    lr=1e37,
    model_seed=list(range(20)),
    scaling=list(np.logspace(0, -32, 65)),
    test_at_end_only=True,
    task='transitive',
    n=7
)

def main(args):
    argparse_array.call_script('train.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
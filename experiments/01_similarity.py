"""Determine similarities for slim (1,000) and wide (10,000) networks.

This is depicted in Fig. 5b.
"""

import argparse
import sys
import os
import re

sys.path.append('')

from python_functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    folder=name_instance('hdims', 'model_seed', base_folder='data-raw/simulations/01_similarity'),
    criterion='mse',
    epochs=int(1e6),
    hdims=[100, 1000, 10000, 50000],
    lr=1e10,
    model_seed=list(range(20)),
    test_at_end_only=True,
    task='transitive',
    n=7,
    mode='linear_readout'
)

def main(args):
    argparse_array.call_script('train.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
"""Determine the impact of noisy representations (i.e. violation of exchangeability) on network behavior.
"""

import argparse
import sys
import os
import re

sys.path.append('')

from python_functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    folder=name_instance('hdims', 'model_seed', base_folder='data-raw/simulations/18_noise'),
    criterion='mse',
    epochs=int(1e6),
    hdims=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    lr=1e20,
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
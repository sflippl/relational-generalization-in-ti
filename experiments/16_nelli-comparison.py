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
    folder=name_instance('hdims', 'model_seed', 'g_factor', 'symmetric_input_weights', 'batch_size', 'epochs', base_folder='data-raw/simulations/16_nelli-comparison'),
    criterion='mse',
    epochs=[int(1e5), 40],
    hdims=[[20], [1000]],
    task='transitive',
    batch_size=[1, 12],
    lr=0.025,
    n=7,
    symmetric_input_weights=[True, False],
    g_factor=[[0.025, 0.025], [0.025, 1], [1, 1]],
    model_seed=list(range(20)),
    test_at_end_only=True,
    mode='backprop',
    backprop_similarity=True
)

def main(args):
    argparse_array.call_script('train_2.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
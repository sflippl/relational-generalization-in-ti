"""Train DNNs at different scales and track their predictions throughout time.
"""

import argparse
import sys
import os
import re

sys.path.append('')

from python_functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    folder=name_instance('scaling', 'model_seed', base_folder='data-raw/simulations/02_time-trajectory-ti'),
    criterion='mse',
    epochs=int(1e6),
    hdims=50000,
    lr=1e37,
    model_seed=list(range(20)),
    scaling=[1., 1e-6, 1e-32],
    test_at_end_only=False,
    task='transitive',
    n=7,
    test_every_n_epochs=1,
    backprop_similarity=True
)

def main(args):
    argparse_array.call_script('train.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
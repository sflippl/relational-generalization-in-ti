"""Determine similarities for slim (1,000) and wide (10,000) networks.

This is depicted in Fig. 5b.
"""

import argparse
import sys
import os
import re

sys.path.append('')

from python_functions.array_training import ArgparseArray, name_instance

base_folder = 'data-raw/simulations/16_nelli-comparison'

argparse_array = ArgparseArray(
    aux_folder=os.listdir(base_folder),
    path=(lambda array_id, folder, **kwargs: os.path.join(base_folder, folder, 'similarity.feather')),
    save_path=(lambda array_id, folder, **kwargs: os.path.join(base_folder, folder, 'rsa.feather'))
)

def main(args):
    argparse_array.call_script('rsa.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
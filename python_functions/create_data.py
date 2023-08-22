import itertools

import numpy as np
import torch

def get_test_data(n):
    """Generate test data for relational tasks.
    """
    test_x = np.zeros((n, n, 2*n))
    for i, j in itertools.product(range(n), range(n)):
        test_x[i,j,i] = 1
        test_x[i,j,n+j] = 1
    test_x = torch.from_numpy(test_x).float()
    test_x = test_x/torch.sqrt(torch.tensor(2))
    return test_x

def get_transitive_data(n):
    """Generate training data for TI task.
    """
    test_x = get_test_data(n)
    x = test_x[tuple(zip(*([(i, i+1) for i in range(n-1)] + [(i+1, i) for i in range(n-1)])))]
    y = torch.tensor([1.]*(n-1)+[-1.]*(n-1))
    return x, y

def get_transverse_data(n):
    """Generate training data for TP task.
    """
    test_x = get_test_data(n)
    x = test_x[tuple(zip(*([(i, (i+1)%n) for i in range(n)] + [((i+1)%n, i) for i in range(n)])))]
    y = torch.tensor([1.]*n+[-1.]*n)
    return x, y

def get_banded_transverse_data(n):
    """Generate training data for BTP task.
    """
    test_x = get_test_data(n)
    x = test_x[tuple(zip(*(
        [(i, (i+1)%n) for i in range(n)] + 
        [(i, (i+3)%n) for i in range(n)] +
        [((i+1)%n, i) for i in range(n)] +
        [((i+3)%n, i) for i in range(n)]
    )))]
    y = torch.tensor([1.]*(2*n)+[-1.]*(2*n))
    return x, y

def add_argparse_arguments(parser):
    parser.add_argument('--task', choices=['transitive', 'transverse', 'banded_transverse'], default='transitive')
    parser.add_argument('--n', type=int, default=10)
    return parser

def get_data(args):
    return {
        'transitive': get_transitive_data,
        'transverse': get_transverse_data,
        'banded_transverse': get_banded_transverse_data
    }[args.task](args.n)

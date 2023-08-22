import os
import argparse

import numpy as np
import torch

import python_functions.create_data as cd
import python_functions.train as tr
import python_functions.networks as nt

def main(args):
    torch.manual_seed(args.model_seed)
    train_data = cd.get_data(args)
    test_x = cd.get_test_data(args.n)
    model = nt.get_network(args)
    os.makedirs(args.folder, exist_ok=True)
    if (args.mode == 'backprop') and not args.backprop_similarity:
        df_losses, df_test_preds, model = tr.run_training(args, model, train_data, test_x)
    else:
        df_losses, df_test_preds, df_sim, model = tr.run_training(args, model, train_data, test_x)
        df_sim.to_feather(os.path.join(args.folder, 'similarity.feather'))
    df_losses.to_feather(os.path.join(args.folder, 'losses.feather'))
    df_test_preds.to_feather(os.path.join(args.folder, 'test_preds.feather'))
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.folder, 'model.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--save_model', action='store_true')
    parser = cd.add_argparse_arguments(parser)
    parser = nt.add_argparse_arguments(parser)
    parser = tr.add_argparse_arguments(parser)
    args = parser.parse_args()
    main(args)

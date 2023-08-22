import functools
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

class LinearWithSqueeze(nn.Linear):
    def __init__(self, inp_dim):
        super().__init__(inp_dim, 1, bias=False)
        nn.init.zeros_(self.weight)
    
    def forward(self, x):
        return torch.squeeze(super().forward(x), -1)

def array_to_dataframe(array):
    dims = array.shape
    flat_array = array.flatten()
    dct_flat = {
        "dim%d"%i: np.array(
            np.repeat(
                range(dims[i]), functools.reduce(lambda x,y:x*y, dims[i+1:], 1)
            ).tolist() * functools.reduce(lambda x,y:x*y, dims[:i], 1)
        ) for i in range(len(dims))
    }
    dct_flat['array'] = flat_array
    df_flat = pd.DataFrame(dct_flat)
    return df_flat

def mse_loss(y_hat, y):
    return torch.mean((y-y_hat)**2)

def crossentropy(y_hat, y):
    return torch.mean(torch.log(1+torch.exp(-y_hat*y)))

def get_loss(arg):
    return {
        'mse': F.mse_loss,
        'crossentropy': crossentropy
    }[arg]

def test_this_epoch(epoch, test_every_n_epochs, tested_epochs):
    if tested_epochs is None:
        return epoch%test_every_n_epochs==0
    else:
        return (epoch in tested_epochs)

def train(model, train_data, test_x, criterion, test_every_n_epochs=50, epochs=1000, lr=0.01,
          momentum=0., lr_tuning=True, early_stopping=True, early_stopping_count=50,
          test_at_end_only=False, max_loss=1e-5, tested_epochs=None):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    x, y = train_data
    if early_stopping:
        min_loss = float('Inf')
        counter = 0
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if test_this_epoch(i, test_every_n_epochs, tested_epochs) and (not test_at_end_only):
            with torch.no_grad():
                new_df = get_test_pred_df(model(test_x))
                new_df['epoch'] = i
                test_preds.append(new_df)
        if lr_tuning and ((loss > 1) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train(or_model, train_data, test_x, criterion, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, early_stopping=early_stopping, early_stopping_count=early_stopping_count, test_at_end_only=test_at_end_only)
        if early_stopping:
          if (loss < min_loss) | (loss > max_loss):
              min_loss = loss
              counter = 0
          else:
              counter += 1
          if counter >= early_stopping_count:
              print(f'Early stopping at epoch {i}. Minimal loss: {min_loss}')
              with torch.no_grad():
                  new_df = get_test_pred_df(model(test_x))
                  new_df['epoch'] = i
                  test_preds.append(new_df)
              break
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['lr'] = lr
    return losses, pd.concat(test_preds).reset_index(), model
  
def get_test_pred_df(test_preds):
    df_test_preds = array_to_dataframe(test_preds.numpy())
    df_test_preds['j'] = df_test_preds['dim0']
    df_test_preds['k'] = df_test_preds['dim1']
    df_test_preds['margin'] = df_test_preds['array']
    return df_test_preds[['j', 'k', 'margin']]

def add_argparse_arguments(parser):
    parser.add_argument('--criterion', type=get_loss, default=F.mse_loss)
    parser.add_argument('--epochs', type=int, default=int(1e4))
    parser.add_argument('--test_every_n_epochs', type=int, default=50)
    parser.add_argument('--tested_epochs', type=int, nargs='+', default=None)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0., type=float)
    parser.add_argument('--test_at_end_only', action='store_true')
    parser.add_argument('--max_loss', default=1e-5, type=float)
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('--backprop_similarity', action='store_true')
    return parser

def run_training(args, model, train_data, test_x):
    x, y = train_data
    if args.mode == 'ntk':
        x = model.ntk_features(x)
        var_x = torch.mean(torch.sum(x**2, dim=-1))
        #x = x/var_x
        test_x = model.ntk_features(test_x)
        #test_x = test_x/var_x
        model = LinearWithSqueeze(x.shape[-1])
    elif args.mode == 'linear_readout':
        x = model.features(x).clone().detach()
        var_x = torch.mean(torch.sum(x**2, dim=-1))
        #x = x/var_x
        test_x = model.features(test_x).clone().detach()
        #test_x = test_x/var_x
        x = x.detach()
        test_x = test_x.detach()
        model = LinearWithSqueeze(x.shape[-1])
    if args.mode != 'backprop':
        sim = torch.einsum('ijk,lmk->ijlm', test_x, test_x)
        i = torch.arange(7).reshape(7,1,1,1)
        i_prime = torch.arange(7).reshape(1,1,7,1)
        j = torch.arange(7).reshape(1,7,1,1)
        j_prime = torch.arange(7).reshape(1,1,1,7)
        same = (i==i_prime)&(j==j_prime)
        overlap = ((i==i_prime)|(j==j_prime))&(~same)
        distinct = (i!=i_prime)&(j!=j_prime)
        sim_df = []
        for _type, _sim in zip(
            ['same', 'overlap', 'distinct'],
            [same, overlap, distinct]
        ):
            new_df = pd.DataFrame({
                'similarity': sim[_sim&(i!=j)&(i_prime!=j_prime)].detach()
            })
            new_df['type'] = _type
            sim_df.append(new_df)
        sim_df = pd.concat(sim_df).reset_index(drop=True)
    if args.backprop_similarity:
        feats = model.module.features(test_x).clone().detach()
        sim = torch.einsum('ijk,lmk->ijlm', feats, feats)
        i = torch.arange(7).reshape(7,1,1,1).repeat((1,7,7,7))
        i_prime = torch.arange(7).reshape(1,1,7,1).repeat((7,7,1,7))
        j = torch.arange(7).reshape(1,7,1,1).repeat((7,1,7,7))
        j_prime = torch.arange(7).reshape(1,1,1,7).repeat((7,7,7,1))
        sim_df_1 = pd.DataFrame({
            'i1': i.flatten().numpy(),
            'i2': i_prime.flatten().numpy(),
            'j1': j.flatten().numpy(),
            'j2': j_prime.flatten().numpy(),
            'sim': sim.flatten().numpy(),
            'features': 'linear readout',
            'type': 'before training'
        })
        feats = model.ntk_features(test_x)
        sim = torch.einsum('ijk,lmk->ijlm', feats, feats)
        sim_df_2 = pd.DataFrame({
            'i1': i.flatten().numpy(),
            'i2': i_prime.flatten().numpy(),
            'j1': j.flatten().numpy(),
            'j2': j_prime.flatten().numpy(),
            'sim': sim.flatten().numpy(),
            'features': 'ntk',
            'type': 'before training'
        })
        feats = model.module.features[:(-1)](test_x).clone().detach()
        sim = torch.einsum('ijk,lmk->ijlm', feats, feats)
        i = torch.arange(7).reshape(7,1,1,1).repeat((1,7,7,7))
        i_prime = torch.arange(7).reshape(1,1,7,1).repeat((7,7,1,7))
        j = torch.arange(7).reshape(1,7,1,1).repeat((7,1,7,7))
        j_prime = torch.arange(7).reshape(1,1,1,7).repeat((7,7,7,1))
        sim_df_3 = pd.DataFrame({
            'i1': i.flatten().numpy(),
            'i2': i_prime.flatten().numpy(),
            'j1': j.flatten().numpy(),
            'j2': j_prime.flatten().numpy(),
            'sim': sim.flatten().numpy(),
            'features': 'preactivation',
            'type': 'before training'
        })
        sim_df = pd.concat([sim_df_1, sim_df_2, sim_df_3])
    df_losses, df_test_preds, model = train(
        model, (x, y), test_x,
        criterion=args.criterion, test_every_n_epochs=args.test_every_n_epochs,
        epochs=args.epochs, lr=args.lr, momentum=args.momentum, test_at_end_only=args.test_at_end_only,
        max_loss=args.max_loss, early_stopping=(not args.no_early_stopping), tested_epochs=args.tested_epochs
    )
    if args.backprop_similarity:
        feats = model.module.features(test_x).clone().detach()
        sim = torch.einsum('ijk,lmk->ijlm', feats, feats)
        i = torch.arange(7).reshape(7,1,1,1).repeat((1,7,7,7))
        i_prime = torch.arange(7).reshape(1,1,7,1).repeat((7,7,1,7))
        j = torch.arange(7).reshape(1,7,1,1).repeat((7,1,7,7))
        j_prime = torch.arange(7).reshape(1,1,1,7).repeat((7,7,7,1))
        sim_df_1 = pd.DataFrame({
            'i1': i.flatten().numpy(),
            'i2': i_prime.flatten().numpy(),
            'j1': j.flatten().numpy(),
            'j2': j_prime.flatten().numpy(),
            'sim': sim.flatten().numpy(),
            'features': 'linear readout',
            'type': 'after training'
        })
        feats = model.ntk_features(test_x)
        sim = torch.einsum('ijk,lmk->ijlm', feats, feats)
        sim_df_2 = pd.DataFrame({
            'i1': i.flatten().numpy(),
            'i2': i_prime.flatten().numpy(),
            'j1': j.flatten().numpy(),
            'j2': j_prime.flatten().numpy(),
            'sim': sim.flatten().numpy(),
            'features': 'ntk',
            'type': 'after training'
        })
        feats = model.module.features[:(-1)](test_x).clone().detach()
        sim = torch.einsum('ijk,lmk->ijlm', feats, feats)
        i = torch.arange(7).reshape(7,1,1,1).repeat((1,7,7,7))
        i_prime = torch.arange(7).reshape(1,1,7,1).repeat((7,7,1,7))
        j = torch.arange(7).reshape(1,7,1,1).repeat((7,1,7,7))
        j_prime = torch.arange(7).reshape(1,1,1,7).repeat((7,7,7,1))
        sim_df_3 = pd.DataFrame({
            'i1': i.flatten().numpy(),
            'i2': i_prime.flatten().numpy(),
            'j1': j.flatten().numpy(),
            'j2': j_prime.flatten().numpy(),
            'sim': sim.flatten().numpy(),
            'features': 'preactivation',
            'type': 'after training'
        })
        sim_df = pd.concat([sim_df, sim_df_1, sim_df_2, sim_df_3]).reset_index(drop=True)
    if (args.mode == 'backprop') and not args.backprop_similarity:
        return df_losses, df_test_preds, model
    else:
        return df_losses, df_test_preds, sim_df, model

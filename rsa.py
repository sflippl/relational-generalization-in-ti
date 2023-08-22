import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import torch.optim as optim

def predict(x, params):
        rtn = 0
        if 'rank1' in params.keys():
            rank_diffs_1 = ((x['i1']-x['i2'])@params['rank1'])**2
            rank_diffs_2 = ((x['j1']-x['j2'])@params['rank2'])**2
            rtn = rtn + rank_diffs_1 + rank_diffs_2
        if 'add_param' in params.keys():
            rtn = rtn + x['additivity']@params['add_param']
        if 'rank_diff1' in params.keys():
            rtn = rtn + ((x['j1']@params['rank_diff1']-x['i1']@params['rank_diff2'])-(x['j2']@params['rank_diff1']-x['i2']@params['rank_diff2']))**2
        return rtn + params['intercept']

def fun(df_, type):
    y = torch.from_numpy(df_['distance'].to_numpy())
    y = y/y.max()
    i1 = F.one_hot(torch.from_numpy(df_['i1'].to_numpy()), 7).float()
    i2 = F.one_hot(torch.from_numpy(df_['i2'].to_numpy()), 7).float()
    j1 = F.one_hot(torch.from_numpy(df_['j1'].to_numpy()), 7).float()
    j2 = F.one_hot(torch.from_numpy(df_['j2'].to_numpy()), 7).float()
    additivity = F.one_hot(torch.sum(i1*i2, dim=1).long()+torch.sum(j1*j2, dim=1).long(), 3).float()
    same_label = torch.from_numpy((np.sign(df_['i1']-df_['j1'])==np.sign(df_['i2']-df_['j2'])).to_numpy()).float().reshape(-1,1)
    rank1 = nn.Parameter(Normal(0, 1).sample((7,))*0.01)
    rank2 = nn.Parameter(Normal(0, 1).sample((7,))*0.01)
    rank_diff1 = nn.Parameter(Normal(0, 1).sample((7,))*0.01)
    rank_diff2 = nn.Parameter(Normal(0, 1).sample((7,))*0.01)
    add_param = nn.Parameter(Normal(0, 1).sample((3,))*0.01)
    intercept = nn.Parameter(Normal(0,1).sample((1,))*0.01)
    params = {
        'intercept': intercept
    }
    if type in ['rank', 'all']:
        params['rank1'] = rank1
        params['rank2'] = rank2
    if type in ['add', 'all']:
        params['add_param'] = add_param
    if type in ['rank_diff', 'all']:
        params['rank_diff1'] = rank_diff1
        params['rank_diff2'] = rank_diff2
    x = {
        'i1': i1, 'i2': i2, 'j1': j1, 'j2': j2,
        'additivity': additivity,
        'same_label': same_label
    }
    losses = []
    optimizer = optim.SGD(params.values(), lr=1e-2, momentum=0.9)
    for _ in range(10000):
        optimizer.zero_grad()
        y_hat = predict(x, params)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    return pd.DataFrame({
        'step': np.arange(10000), 'loss': losses
    })

def main(args):
    df = pd.read_feather(args.path)
    df = df.reset_index(drop=True).merge(
        df[(df['i1']==df['i2'])&(df['j1']==df['j2'])][df.columns.difference(['i1', 'j1'])].rename(columns = {'sim': 'baseline_1'}).reset_index(drop=True)
    ).merge(
        df[(df['i1']==df['i2'])&(df['j1']==df['j2'])][df.columns.difference(['i2', 'j2'])].rename(columns = {'sim': 'baseline_2'}).reset_index(drop=True)
    )
    df['distance'] = df['baseline_1'] + df['baseline_2'] - 2*df['sim']
    loss = []
    for type in ['all', 'rank_diff', 'rank', 'add', 'intercept']:
        new_df = df.groupby(['features', 'type']).apply((lambda df_: fun(df_, type))).reset_index()
        new_df['params'] = type
        loss.append(new_df)
    loss = pd.concat(loss).reset_index(drop=True)
    loss.to_feather(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--save_path', required=True, type=str)
    args = parser.parse_args()
    main(args)

from copy import deepcopy
import math

import torch
import torch.nn as nn

class ABReLU(nn.Module):
    def __init__(self, rho=1):
        super().__init__()
        b = B(rho)
        self.b = b
    
    def forward(self, x):
        return self.b*torch.minimum(x,torch.tensor(0.))+torch.maximum(x,torch.tensor(0.))

def B(rho):
    b = 0 if rho==1 else (math.sqrt(1-(rho-1)**2)-1)/(rho-1)
    return b

class ModelWithNTK(nn.Module):
    """Creates a model which returns its neural tangent features.
    """
    def __init__(self):
        super().__init__()
    
    def get_gradient(self, x):
        self.zero_grad()
        y = self(x)
        y.backward()
        return torch.cat([param.grad.flatten() for param in self.parameters()])

    def ntk_features(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        feats = torch.stack([
            self.get_gradient(_x) for _x in x
        ], dim=0)
        return feats.reshape(*shape[:(-1)], -1).detach().clone()

class DenseNet(ModelWithNTK):
    """Dense neural network.
    """
    def __init__(self, inp_dim, hdims=None, rho=1, bias=False, linear_readout=False,
                 nonlinearity='piecewise_linear'):
        super().__init__()
        hdims = hdims or []
        L = []
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims[:-1], hdims)):
            linear = nn.Linear(_in, _out, bias=bias)
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(linear.weight, std=math.sqrt(2/(((B(rho)-1)**2)*_in)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(linear.weight)
            L.append(linear)
            if nonlinearity == 'piecewise_linear':
                L.append(ABReLU(rho))
            elif nonlinearity == 'tanh':
                L.append(nn.Tanh())
        self.features = nn.Sequential(*L)
        if len(hdims) > 0:
            _in = hdims[-1]
        else:
            _in = inp_dim
        self.readout = nn.Linear(_in, 1, bias=bias)
        if linear_readout:
            nn.init.zeros_(self.readout.weight)
        else:
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(self.readout.weight, std=math.sqrt(2/(((B(rho)-1)**2)*_out)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(self.readout.weight)
        self.linear_readout = linear_readout
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def forward(self, x):
        x = self.features(x)
        x = self.readout(x)
        x = torch.squeeze(x, -1)
        return x

class DenseNet2(nn.Module):
    def __init__(self, inp_dim, hdims=None, rho=1, bias=False, linear_readout=False,
                 nonlinearity='piecewise_linear', g_factor=None, symmetric_input_weights=False):
        super().__init__()
        hdims = hdims or []
        if symmetric_input_weights:
            inp_dim = inp_dim//2
        L = []
        g_factor = g_factor or [1]*(len(hdims)+1)
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims[:-1], hdims)):
            linear = nn.Linear(_in, _out, bias=bias)
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(linear.weight, std=g_factor[i]*math.sqrt(1/(((B(rho)-1)**2*_in))))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(linear.weight)
            if bias:
                nn.init.uniform_(linear.bias, -g_factor[i]*math.sqrt(1/_in), g_factor[i]*math.sqrt(1/_in))
            L.append(linear)
            if nonlinearity == 'piecewise_linear':
                L.append(ABReLU(rho))
            elif nonlinearity == 'tanh':
                L.append(nn.Tanh())
        self._features = nn.Sequential(*L)
        if len(hdims) > 0:
            _in = hdims[-1]
        else:
            _in = inp_dim
        self.readout = nn.Linear(_in, 1, bias=bias)
        if linear_readout:
            nn.init.zeros_(self.readout.weight)
        else:
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(self.readout.weight, std=g_factor[-1]*math.sqrt(1/(((B(rho)-1)**2*_out))))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(self.readout.weight)
            if bias:
                nn.init.uniform_(self.readout.bias, -g_factor[-1]*math.sqrt(1/_in), g_factor[-1]*math.sqrt(1/_in))
        self.linear_readout = linear_readout
        self.symmetric_input_weights = symmetric_input_weights
        self.inp_dim = inp_dim
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def features(self, x):
        if self.symmetric_input_weights:
            x = torch.index_select(x, -1, torch.arange(self.inp_dim))-\
                torch.index_select(x, -1, torch.arange(self.inp_dim, 2*self.inp_dim))
        x = self._features(x)
        return x
    
    def preactivation(self, x):
        if self.symmetric_input_weights:
            x = torch.index_select(x, -1, torch.arange(self.inp_dim))-\
                torch.index_select(x, -1, torch.arange(self.inp_dim, 2*self.inp_dim))
        x = self._features[:(-1)](x)
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = self.readout(x)
        x = torch.squeeze(x, -1)
        return x
        

class ZeroOutput(ModelWithNTK):
    """Dense neural network normalized by its initial network.

    Creates a network which subtracts its initialization value. This unbiases the random initialization.
    """
    def __init__(self, module, scaling=1.):
        super().__init__()
        self.module = module
        self.init_module = deepcopy(module)
        self.scaling = scaling
    
    def parameters(self):
        return self.module.parameters()
    
    def forward(self, x):
        return self.scaling*(self.module(x)-self.init_module(x))

def add_argparse_arguments(parser):
    parser.add_argument('--hdims', nargs='*', default=[], type=int)
    parser.add_argument('--rho', type=float, default=1)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--scaling', default=1., type=float)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--mode', choices=['backprop', 'linear_readout', 'ntk'], default='backprop')
    parser.add_argument('--nonlinearity', choices=['piecewise_linear', 'tanh'], default='piecewise_linear')
    return parser

def add_argparse_arguments_2(parser):
    parser.add_argument('--hdims', nargs='*', default=[], type=int)
    parser.add_argument('--rho', type=float, default=1)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--scaling', default=1., type=float)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--mode', choices=['backprop', 'linear_readout', 'ntk'], default='backprop')
    parser.add_argument('--nonlinearity', choices=['piecewise_linear', 'tanh'], default='piecewise_linear')
    parser.add_argument('--symmetric_input_weights', action='store_true')
    parser.add_argument('--g_factor', default=None, type=float, nargs='*')
    return parser

def get_network(args):
    torch.manual_seed(args.model_seed)
    model = DenseNet(
        inp_dim=2*args.n, hdims=args.hdims, rho=args.rho, bias=args.bias,
        linear_readout=(args.mode=='linear_readout'), nonlinearity=args.nonlinearity
    )
    if args.mode != 'linear_readout':
        model = ZeroOutput(model, scaling=args.scaling)
    return model

def get_network_2(args):
    torch.manual_seed(args.model_seed)
    model = DenseNet2(
        inp_dim=2*args.n, hdims=args.hdims, rho=args.rho, bias=args.bias,
        linear_readout=(args.mode=='linear_readout'), nonlinearity=args.nonlinearity,
        g_factor=args.g_factor, symmetric_input_weights=args.symmetric_input_weights
    )
    if args.mode != 'linear_readout':
        model = ZeroOutput(model, scaling=args.scaling)
    return model

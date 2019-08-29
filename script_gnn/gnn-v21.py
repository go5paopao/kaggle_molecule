FOLD_IDX = 1

# std libs
import numpy as np
import os,gc,math,json,zipfile,csv,pickle,glob,sys,time
from datetime import datetime
from tqdm import tqdm
import collections,copy,numbers,inspect,shutil,random,itertools
from timeit import default_timer as timer
from collections import OrderedDict
import multiprocessing as mp
#from pprintpp import pprint, pformat
import pandas as pd
from distutils.dir_util import copy_tree
# torch libs
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torch.nn.utils.rnn import *
# for gnn
import torch_geometric.nn as gnn
# add
from sklearn.model_selection import GroupKFold
# add
from dscribe.descriptors import ACSF
from dscribe.core.system import System
from torch_scatter import *
from torch_geometric.utils import scatter_
# original
from utils import *


# setting gpu 
COMMON_STRING =''
SEED = int(time.time()) #35202   #35202  #123  #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
COMMON_STRING += '\tset random seed\n'
COMMON_STRING += '\t\tSEED = %d\n'%SEED
torch.backends.cudnn.benchmark     = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled       = True
torch.backends.cudnn.deterministic = True
COMMON_STRING += '\tset cuda environment\n'
COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
try:
    COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
    NUM_CUDA_DEVICES = 1
COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
print(COMMON_STRING)

#global infomation
IDENTIFIER = 1
NODE_DIM=113#126#113#93 # 13 + 80(acsf)
EDGE_DIM=24
COUPLING_TYPE_STATS=[
    #type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]
NUM_TARGET = len(COUPLING_TYPE_STATS)//5
NUM_COUPLING_TYPE = NUM_TARGET
COUPLING_TYPE_MEAN = [ COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_TARGET)]
COUPLING_TYPE_STD  = [ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_TARGET)]
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_TARGET)]
AUGMENT_COUPLING_TYPE_IDX = [i for i,type_name in enumerate(COUPLING_TYPE) if type_name in ["2JHH","3JHH"]]

g_molecule_name_idx = 0
g_smile_idx = 1
g_atom_idx = 2
g_node_idx = 3
g_edge_idx = 4
g_edge_index_idx = 5
g_coupling_idx = 6
g_bond_detail_idx = 7
c_id_idx = 0
c_contribution_idx = 1
c_index_idx = 2
c_type_idx = 3
c_value_idx = 4
c_norm_scc = 5 
c_atom2_idx = 6
c_atom3_idx = 7

def make_graph_dict(molecules):
    graph_dict = {}
    for molecule_name in tqdm(molecules):
        with open('../data/graph_v8/{}.pickle'.format(molecule_name),"rb") as f:
            graph_dict[molecule_name] = list(pickle.load(f))
    return graph_dict

class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0
    def __call__(self, time):
        return self.lr
    def __str__(self):
        string = 'NullScheduler\n' + 'lr=%0.5f '%(self.lr)
        return string

class ManualScheduler():
    def __init__(self):
        super(ManualScheduler, self).__init__()
        self.lr_list    = [0.001,0.0009,0.0008,0.0006,0.0004,0.0002,0.0001,0.00009,0.00008, 0.00005, 0.00005, 0.00005]
        self.lr = self.lr_list[0]
        self.period = 3000*15
        
    def __call__(self, time):
        ix = min(time // self.period, len(self.lr_list))
        self.lr = self.lr_list[ix]
        return self.lr

    def __str__(self):
        string = 'ManualScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.1)
        self.act  = act
    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim ):
        super(GraphConv, self).__init__()
        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim),
            #nn.ReLU(inplace=True),
        )
        self.gru  = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))

    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        edge_index = edge_index.t().contiguous()
        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        edge    = self.encoder(edge).view(-1,node_dim,node_dim)
        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)
        message = x_i.view(-1,1,node_dim)@edge
        message = message.view(-1,node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)
        message = F.relu(message +self.bias)
        #2. update: n_j = f(n_j, m_j)
        update = message
        #batch_first=True
        update, hidden = self.gru(update.view(1,-1,node_dim), hidden)
        update = update.view(-1,node_dim)
        return update, hidden

class GraphConv2(nn.Module):
    def __init__(self, node_dim, edge_dim ):
        super(GraphConv2, self).__init__()
        self.encoder = nn.Sequential(
            LinearBn(node_dim*2, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, edge_dim * edge_dim),
            #nn.ReLU(inplace=True),
        )
        self.gru  = nn.GRU(edge_dim, edge_dim, batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(edge_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(edge_dim), 1.0 / math.sqrt(edge_dim))

    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        edge_index = edge_index.t().contiguous()
        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        x_j     = torch.index_select(node, 0, edge_index[1])
        x_ij    = torch.cat((x_i, x_j), 1)
        x_ij    = self.encoder(x_ij).view(-1,edge_dim,edge_dim)
        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)
        message = edge.view(-1,1,edge_dim)@x_ij
        message = message.view(-1,edge_dim)
        message = F.relu(message +self.bias)
        #2. update: n_j = f(n_j, m_j)
        update = message
        #batch_first=True
        update, hidden = self.gru(update.view(1,-1,edge_dim), hidden)
        update = update.view(-1,edge_dim)

        return update, hidden

class Set2Set(torch.nn.Module):
    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel
        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1
        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))
        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)
            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            #apply attention #shape = batch_size x ...
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)
        return q_star

#message passing
class Net(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(Net, self).__init__()        
        
        node_dim2 = 128
        edge_dim2 = 128

        self.num_target = num_target
        self.num_propagate = 6
        self.num_s2s = 6

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, node_dim2),
            nn.ReLU(inplace=True),
            LinearBn(node_dim2, node_dim2),
            nn.ReLU(inplace=True),
        )

        self.preprocess2 = nn.Sequential(
            LinearBn(edge_dim, edge_dim2),
            nn.ReLU(inplace=True),
            LinearBn(edge_dim2, edge_dim2),
            nn.ReLU(inplace=True),
        )

        for i in range(self.num_propagate):
            if i%2==0:
                setattr(self, "propagate_{}".format(i), GraphConv(node_dim2, edge_dim2))
            else:
                setattr(self, "propagate_{}".format(i), GraphConv2(node_dim2, edge_dim2))

        self.set2set = Set2Set(128, processing_step=self.num_s2s)
        
        #predict coupling constant
        for i in range(num_target):
            if i not in [1,4,6]:
                setattr(self, 
                    "type_predicts_{}".format(i),
                    nn.Sequential(
                        LinearBn(5*128+edge_dim+2, 1024),  #node_hidden_dim
                        nn.ReLU(inplace=True),
                        LinearBn( 1024, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 4),
                    )
                )
            else:
                setattr(self, 
                    "type_predicts_{}".format(i),
                    nn.Sequential(
                        LinearBn(6*128+edge_dim+2, 1024),  #node_hidden_dim
                        nn.ReLU(inplace=True),
                        LinearBn( 1024, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 4),
                    )
                )
            self.bond_detail_embedding = nn.Embedding(37, 8)

    def forward(self, node, edge, edge_index, node_index, coupling_index, bond_detail):

        #save original node
        node_origin = node
        
        #bond_detail
        bond_detail_embed = self.bond_detail_embedding(bond_detail).view(-1,8)
        #concat to edge feature
        edge = torch.cat([edge, bond_detail_embed], -1)
        edge0 = edge

        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        node   = self.preprocess(node)
        edge   = self.preprocess2(edge)
        hidden = node.view(1,num_node,-1)
        hidden2 = edge.view(1,num_edge,-1)

        for i in range(self.num_propagate):
            pred_func = getattr(self, "propagate_{}".format(i))
            if i%2==0:
                node, hidden =  pred_func(node, edge_index, edge, hidden)
            else:
                edge, hidden2 =  pred_func(node, edge_index, edge, hidden2)

        pool = self.set2set(node, node_index)
        edge = torch.cat([edge, edge0], -1)

        #---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, \
            coupling_type_index, coupling_batch_index, \
            coupling_edge_index,coupling_atom2_index = \
                    torch.split(coupling_index,1,dim=1)
        predicts = []
        for i in range(self.num_target):
            pool_i  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
            node0_i = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1))
            node1_i = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1))

            node0_i_origin = torch.index_select( node_origin[:,28], dim=0, index=coupling_atom0_index.view(-1)).view(-1,1)
            node1_i_origin = torch.index_select( node_origin[:,28], dim=0, index=coupling_atom1_index.view(-1)).view(-1,1)
            skip_x_edge = torch.index_select( edge, dim=0, index=coupling_edge_index.view(-1))
            
            if i not in [1,4,6]:
                x = torch.cat([pool_i,node0_i,node1_i, skip_x_edge, node0_i_origin,node1_i_origin],-1)
            else:
                node2_i = torch.index_select( node, dim=0, index=coupling_atom2_index.view(-1))
                x = torch.cat([pool_i,node0_i,node1_i, skip_x_edge, node0_i_origin,node1_i_origin, node2_i],-1)
            pred_func = getattr(self, "type_predicts_{}".format(i))
            type_predict = pred_func(x)
            type_predict = type_predict.view(-1, 1, 4)
            predicts.append(type_predict)

        predict = torch.cat(predicts, 1)
        coupling_type_index = coupling_type_index.view(-1,1,1)
        coupling_type_index = coupling_type_index.expand(-1,-1,4)
        predict = torch.gather(predict, 1, coupling_type_index)
        #predict = torch.index_select(predict, dim=0, index=coupling_atom1_index.view(-1))
        return predict


def compute_kaggle_metric( predict, coupling_value, coupling_type):
    predict = predict.sum(axis=-1).reshape(-1)
    mae     = [None]*NUM_TARGET
    log_mae = [None]*NUM_TARGET
    
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_TARGET):
        index = np.where(coupling_type==t)[0]
        if len(index)>0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)

            mae[t] = m
            log_mae[t] = log_m
        else:
            pass

    return mae, log_mae

def criterion(predict, coupling_value):
    #predict = predict.view(-1)
    #coupling_value = coupling_value.view(-1)
    #assert(predict.shape==coupling_value.shape)
    #loss = F.mse_loss(predict, coupling_value)
    #return loss
    predict = predict.view(-1)
    truth   = coupling_value.view(-1)
    #assert(predict.shape==truth.shape)
    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss

def get_split(fold_idx):
    """
    Validationをローカルのpair単位のCrossValidationのindexに合わせる
    """
    csv_file = '../input/train.csv'
    train_df  = pd.read_csv(csv_file)
    fold_num = 3
    random_state = 2019
    folds = GroupKFold(n_splits = fold_num)
    split_index_list = [(trn_, val_) for trn_, val_ 
                        in folds.split(train_df, train_df["scalar_coupling_constant"], groups=train_df["molecule_name"])]
    molecule_names = train_df.molecule_name.unique()
    
    use_fold_idx = split_index_list[fold_idx]
    train_split = train_df.loc[use_fold_idx[0],"molecule_name"].unique()
    valid_split = train_df.loc[use_fold_idx[1],"molecule_name"].unique()
    #molecule_names = train_df.molecule_name.unique()
    #molecule_names = np.sort(molecule_names)
    #np.random.shuffle(molecule_names)
    #num_all   = len(molecule_names)
    #num_valid = 5000
    #num_train = num_all - num_valid
    #train_split = molecule_names[num_valid:]
    #valid_split = molecule_names[:num_valid]
    return train_split, valid_split

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]
    return lr



DATA_DIR = '../input'
GRAPH = None
GRAPH_MAPPING = None
SYMBOL=['H', 'C', 'N', 'O', 'F']
ACSF_GENERATOR = ACSF(
    species = SYMBOL,
    rcut = 6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

class ChampsDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.df = pd.read_csv(DATA_DIR + '/%s.csv'%csv)
        
        self.id = split
        if split is None:
            self.id = self.df.molecule_name.unique()
        

    def __str__(self):
            string = ''\
            + '\tmode   = %s\n'%self.mode \
            + '\tsplit  = %s\n'%self.split \
            + '\tcsv    = %s\n'%self.csv \
            + '\tlen    = %d\n'%len(self)

            return string

    def __len__(self):
        return len(self.id)


    def __getitem__(self, index):

        molecule_name = self.id[index]
        #graph_file = DATA_DIR + '/atoms-graph/graph/graph/%s.pickle'%molecule_name
        #graph_file = DATA_DIR + '/graph-v4/graph_v4/graph_v4/%s.pickle'%molecule_name
        graph_file = \
        '../data/graph_v8/%s.pickle'%molecule_name
        #graph_file = DATA_DIR + '/graph-v5/graph_v5/graph_v5/%s.pickle'%molecule_name
        #graph_file = DATA_DIR + '/molecule-graph/graph_v2/graph_v2/%s.pickle'%molecule_name
        graph = list(read_pickle_from_file(graph_file))
        assert(graph[0]==molecule_name)

        # ##filter only J link
        # if 0:
        #     # 1JHC,     2JHC,     3JHC,     1JHN,     2JHN,     3JHN,     2JHH,     3JHH
        #     mask = np.zeros(len(graph.coupling.type),np.bool)
        #     for t in ['1JHC',     '2JHH']:
        #         mask += (graph.coupling.type == COUPLING_TYPE.index(t))
        #
        #     graph.coupling.id = graph.coupling.id [mask]
        #     graph.coupling.contribution = graph.coupling.contribution [mask]
        #     graph.coupling.index = graph.coupling.index [mask]
        #     graph.coupling.type = graph.coupling.type [mask]
        #     graph.coupling.value = graph.coupling.value [mask]
        
        # add ACSF
        atom = System(symbols =graph[2][0], positions=graph[2][1])
        acsf = ACSF_GENERATOR.create(atom)
        graph[3] += [acsf,]

        graph[g_node_idx][7] = graph[g_node_idx][7].reshape([-1,1])

        graph[3] = np.concatenate(graph[3],-1)
        dist = np.concatenate(graph[4],-1)[:,4].reshape(-1,1)
        graph[4].append(1/dist)
        graph[4].append(1/dist**2)
        graph[4].append(1/dist**3)
        graph[4].append(1/dist**6)
        #for i in range(len(graph[4])):
        #    print(graph[4][])
        graph[4] = np.concatenate(graph[4],-1)
        graph[3][np.isnan(graph[3])] = 0
        graph[4][np.isnan(graph[4])] = 0
        # replace coupling atom_index2 -1 => 1
        #if np.isnan(graph[3]).sum()>0 or np.isnan(graph[4]).sum() > 0:
        #    print(graph)
        return graph

def augment_coupling(coupling):
    """
    input: graph[g_coupling_idx]
    Swap HH atom index
    """
    augment_idx = np.isin(coupling[c_type_idx],AUGMENT_COUPLING_TYPE_IDX)
    coupling = list(coupling) 
    for ix in range(len(coupling)):
        if ix == c_index_idx:
            coupling[ix] = np.concatenate([coupling[ix], coupling[ix][augment_idx][:,::-1]], 0)
        else:
            coupling[ix] = np.concatenate([coupling[ix], coupling[ix][augment_idx]], 0)
    return tuple(coupling)

def null_collate(batch):
    batch_size = len(batch)
    node = []
    edge = []
    edge_index = []
    node_index = []
    coupling_value = []
    coupling_c_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []
    coupling_edge_index = []
    coupling_atom2_index = []
    infor = []
    bond_detail = []
    offset = 0
    edge_offset = 0
    for b in range(batch_size):
        graph = batch[b]
        #print(graph.molecule_name)
        num_node = len(graph[g_node_idx])
        node.append(graph[g_node_idx])
        edge.append(graph[g_edge_idx])
        edge_index.append(graph[g_edge_index_idx]+offset)
        node_index.append(np.array([b]*num_node))
        num_coupling = len(graph[g_coupling_idx][0])
        coupling_value.append(graph[g_coupling_idx][c_value_idx])
        coupling_c_value.append(graph[g_coupling_idx][c_contribution_idx])
        coupling_atom_index.append(graph[g_coupling_idx][c_index_idx]+offset)
        coupling_type_index.append (graph[g_coupling_idx][c_type_idx])
        coupling_batch_index.append(np.array([b]*num_coupling))
        coupling_edge_index.append(graph[g_coupling_idx][c_index_idx][:,0]*(num_node-1)
                                   +graph[g_coupling_idx][c_index_idx][:,1] 
                                   - (graph[g_coupling_idx][c_index_idx][:,0] 
                                      < graph[g_coupling_idx][c_index_idx][:,1]).astype(int)#.type(torch.long)
                                   + edge_offset)
        c_atom2_index = \
            np.where(graph[g_coupling_idx][c_atom2_idx] == -1, 0, graph[g_coupling_idx][c_atom2_idx])
        coupling_atom2_index.append(c_atom2_index+offset)

        infor.append((graph[g_molecule_name_idx], graph[g_smile_idx], graph[g_coupling_idx][c_id_idx]))
        offset += num_node
        edge_offset += len(graph[g_edge_idx])
        bond_detail.append(graph[g_bond_detail_idx])

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    bond_detail = torch.from_numpy(np.concatenate(bond_detail).astype(np.int32)).long()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int32)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_c_value = torch.from_numpy(np.concatenate(coupling_c_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
        np.concatenate(coupling_edge_index).reshape(-1,1),
        np.concatenate(coupling_atom2_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(coupling_index).long()
    
    ## add mulliken charge
    #mc = torch.from_numpy(np.concatenate(mc)).float()

    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor, bond_detail, coupling_c_value


def augment_collate(batch):
    augment = True
    batch_size = len(batch)
    node = []
    edge = []
    edge_index = []
    node_index = []
    coupling_value = []
    coupling_c_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []
    coupling_edge_index = []
    coupling_atom2_index = []
    infor = []
    bond_detail = []
    offset = 0
    edge_offset = 0
    for b in range(batch_size):
        graph = batch[b]
        #print(graph.molecule_name)
        num_node = len(graph[g_node_idx])
        node.append(graph[g_node_idx])
        edge.append(graph[g_edge_idx])
        edge_index.append(graph[g_edge_index_idx]+offset)
        node_index.append(np.array([b]*num_node))
        # augmentation (HH coupling)
        coupling = augment_coupling(graph[g_coupling_idx])
        num_coupling = len(coupling[0])
        coupling_value.append(coupling[c_value_idx])
        coupling_c_value.append(coupling[c_contribution_idx])
        coupling_atom_index.append(coupling[c_index_idx]+offset)
        coupling_type_index.append (coupling[c_type_idx])
        coupling_batch_index.append(np.array([b]*num_coupling))
        coupling_edge_index.append(coupling[c_index_idx][:,0]*(num_node-1)
                                   +coupling[c_index_idx][:,1] 
                                   - (coupling[c_index_idx][:,0] < coupling[c_index_idx][:,1]).astype(int)#.type(torch.long)
                                   + edge_offset)
        c_atom2_index = \
            np.where(coupling[c_atom2_idx] == -1, 0, coupling[c_atom2_idx])

        coupling_atom2_index.append(c_atom2_index+offset)

        infor.append((graph[g_molecule_name_idx], graph[g_smile_idx], coupling[c_id_idx]))
        offset += num_node
        edge_offset += len(graph[g_edge_idx])
        bond_detail.append(graph[g_bond_detail_idx])

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    bond_detail = torch.from_numpy(np.concatenate(bond_detail).astype(np.int32)).long()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int32)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_c_value = torch.from_numpy(np.concatenate(coupling_c_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
        np.concatenate(coupling_edge_index).reshape(-1,1),
        np.concatenate(coupling_atom2_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(coupling_index).long()
    ## add mulliken charge
    #mc = torch.from_numpy(np.concatenate(mc)).float()
    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor, bond_detail, coupling_c_value



def do_valid(net, valid_loader):

    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, 
            coupling_index, infor, bond_detail, coupling_c_value) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_c_value = coupling_c_value.cuda()
        coupling_index = coupling_index.cuda()
        bond_detail = bond_detail.cuda()
        
        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index, bond_detail)
            
            
            coupling_atom0_index, coupling_atom1_index, \
                coupling_type_index, coupling_batch_index, \
                coupling_edge_index,coupling_atom2_index = \
                        torch.split(coupling_index,1,dim=1)
            
            
            #loss = criterion(predict, coupling_value)
            loss = criterion(predict, coupling_c_value)
            #loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_num  += batch_size

        print('\r %8d /%8d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))
    #print('')
    valid_loss = valid_loss/valid_num

    #compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type  = np.concatenate(valid_coupling_type).astype(np.int32)
    mae, log_mae   = compute_kaggle_metric( predict, coupling_value, coupling_type,)

    num_target = 8
    for t in range(8):
        if mae[t] is None:
            mae[t] = 0
            log_mae[t]  = 0
            num_target -= 1
    mae_mean, log_mae_mean = sum(mae)/num_target, sum(log_mae)/num_target
    #list(np.stack([mae, log_mae]).T.reshape(-1))
    valid_loss = log_mae + [valid_loss,mae_mean, log_mae_mean, ]
    return valid_loss

def run_train():
    PROJECT_PATH = "./"
    out_dir = \
        './result/kaggle_predict5.1-a'
    initial_checkpoint = \
        "../data/v21_cv_1/00125000_model.pth"
    #"../input/gnn-v21-cv-2/result/kaggle_predict5.1-a/checkpoint/00044999_model.pth"
    #schduler = NullScheduler(lr=0.001)
    schduler = ManualScheduler()
    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)
    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    #log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 32 #*2 #280*2 #256*4 #128 #256 #512  #16 #32
    train_split, valid_split = get_split(FOLD_IDX)
    #train_split = train_split[:100]
    #valid_split = valid_split[:100]
    train_dataset = ChampsDataset(
                csv='train',
                mode ='train',
                split = train_split,
                #split='debug_split_by_mol.1000.npy', #
                #split='train_split_by_mol.80003.npy',
                augment=None,
    )
    train_loader  = DataLoader(
                train_dataset,
                #sampler     = SequentialSampler(train_dataset),
                sampler     = RandomSampler(train_dataset),
                batch_size  = batch_size,
                drop_last   = True,
                num_workers = 16,
                pin_memory  = True,
                collate_fn  = augment_collate #null_collate
    )
    valid_dataset = ChampsDataset(
                csv='train',
                mode='train',
                split = valid_split,
                #split='debug_split_by_mol.1000.npy', # #,None
                #split='valid_split_by_mol.5000.npy',
                augment=None,
    )
    valid_loader = DataLoader(
                valid_dataset,
                #sampler     = SequentialSampler(valid_dataset),
                sampler     = RandomSampler(valid_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = True,
                collate_fn  = null_collate
    )
    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    log.write('%s\n'%(type(net)))
    log.write('\n')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)
    iter_accum  = 1
    num_iters   = 3000  *180#00
    iter_smooth = 50
    iter_log    = 5000
    iter_valid  = 5000
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 5000))#1*1000
    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')
    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size =%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                      |--------------- VALID ----------------------------------------------------------------|-- TRAIN/BATCH ---------\n')
    log.write('                      |std %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f   %4.1f  |                    |        | \n'%tuple(COUPLING_TYPE_STD))
    log.write('rate     iter   epoch |    1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH |  loss  mae log_mae | loss   | time          \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------\n')
              #0.00100  111.0* 111.0 | 1.0 +1.2, 2.0 +1.2, 3.0 +1.2, 4.0 +1.2, 5.0 +1.2, 6.0 +1.2, 7.0 +1.2, 8.0 +1.2 | 8.01 +1.21  5.620 | 5.620 | 0 hr 04 min
               #    %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f
    train_loss   = np.zeros(20,np.float32)
    valid_loss   = np.zeros(20,np.float32)
    batch_loss   = np.zeros(20,np.float32)
    iter = 0
    i    = 0
    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = 0
        optimizer.zero_grad()
        for (node, edge, edge_index, node_index, coupling_value, 
             coupling_index, infor, bond_detail, coupling_c_value) in train_loader:
            #while 1:
                batch_size = len(infor)
                iter  = i + start_iter
                epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch
                if (iter % iter_valid==0) and iter > 0:
                    valid_loss = do_valid(net, valid_loader) #
                if (iter % iter_log==0) and iter > 0:
                    print('\r',end='',flush=True)
                    asterisk = '*' if iter in iter_save else ' '
                    log.write('%0.5f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             *valid_loss[:11],
                             train_loss[0],
                             time_to_str((timer() - start),'min'))
                    )
                    log.write('\n')
                #if 0:
                if iter in iter_save:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                    pass
                # learning rate schduler -------------
                lr = schduler(iter)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr)
                rate = get_learning_rate(optimizer)
                net.train()
                node = node.cuda()
                edge = edge.cuda()
                edge_index = edge_index.cuda()
                node_index = node_index.cuda()
                coupling_value = coupling_value.cuda()
                coupling_c_value = coupling_c_value.cuda()
                coupling_index = coupling_index.cuda()
                bond_detail = bond_detail.cuda()
                predict = net(node, edge, edge_index, node_index, coupling_index, bond_detail)
                coupling_atom0_index, coupling_atom1_index, \
                    coupling_type_index, coupling_batch_index, \
                    coupling_edge_index,coupling_atom2_index = \
                            torch.split(coupling_index,1,dim=1)
                #loss = criterion(predict, coupling_value)
                loss = criterion(predict, coupling_c_value)
                (loss/iter_accum).backward()
                if (iter % iter_accum)==0:
                    optimizer.step()
                    optimizer.zero_grad()
                # print statistics  ------------
                batch_loss[:1] = [loss.item()]
                sum_train_loss += batch_loss
                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum_train_loss = np.zeros(20,np.float32)
                    sum = 0
                print('\r',end='',flush=True)
                asterisk = ' '
                print('%0.5f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             *valid_loss[:11],
                             batch_loss[0],
                             time_to_str((timer() - start),'min'))
                , end='',flush=True)
                i=i+1
        pass  #-- end of one data loader --
    pass #-- end of all iterations --
    log.write('\n')


if __name__ == "__main__":
    molecules = pd.read_csv("../input/train.csv")["molecule_name"].unique()
    graph_dict = make_graph_dict(molecules)
    run_train()


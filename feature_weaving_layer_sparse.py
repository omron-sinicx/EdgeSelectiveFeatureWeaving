import warnings

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_min, scatter_mean

from feature_weaving_layer import EncoderMaxPool


class EncoderMaxPool4Sparse(EncoderMaxPool):
    def __init__(self, 
                 *args,
                 **kwargs):
        super(EncoderMaxPool4Sparse,self).__init__(*args,**kwargs)  
        
    
    def forward(self,x, end_points):
        z = self.conv_max(x)        
        z,_ = scatter_max(z,end_points,dim=0)
        z = torch.index_select(z,0,end_points)
        x = torch.cat([x,z],dim=1)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)        

def batched_index_select(input, dim, index):
    print(input.shape,index.shape)
    views = [1 if i != dim else -1 for i in range(len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.cat(torch.chunk(torch.gather(input, dim, index), chunks=index.shape[0], dim=dim), dim=0)


class SparseFeatureWeavingLayer(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels,                 
                 use_batch_norm=True, 
                 activation=nn.PReLU(),
                 vacant_fill=None,
                 asymmetric_encode=True,
                 symmetric_mat=False
                ):
        super(SparseFeatureWeavingLayer,self).__init__()  
        

        self.bn = None
        if use_batch_norm:
            self.bn=True
            self.bn_fw = nn.BatchNorm1d(num_features=out_channels)
            self.bn_bw = nn.BatchNorm1d(num_features=out_channels)
        self.act = activation
        
        conv_max = [nn.Linear(in_channels,mid_channels)]
        conv = [nn.Linear(in_channels+mid_channels, out_channels)]
        
        self.symmetric_mat = symmetric_mat
        if not symmetric_mat:
            if asymmetric_encode:
                conv_max += [nn.Linear(in_channels,mid_channels)]
                conv += [nn.Linear(in_channels+mid_channels, out_channels)]
            else:
                conv_max += conv_max
                conv += conv                
        self.conv_maxs = nn.ModuleList(conv_max)
        self.convs = nn.ModuleList(conv)
        
        self.vacant_fill=vacant_fill
        
    def encode(self, x, end_points, di=0):
        z = self.conv_maxs[di](x)
        z,_ = scatter_max(z,end_points,dim=2)

        z_fw = torch.cat([ torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(z[:,0], end_points[:,0]) ])
        z_bw = torch.cat([ torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(z[:,1], end_points[:,1]) ])
        z = torch.stack([z_fw,z_bw],dim=1)
        x = torch.cat([x,z],dim=3)
        x = self.convs[di](x)
        
        if self.bn is not None:
            batch_size,_,edge_num, channel = x.size()
            x_fw = self.bn_fw(x[:,0].view(batch_size, channel, edge_num)).unsqueeze(1)
            x_bw = self.bn_bw(x[:,1].view(batch_size, channel, edge_num)).unsqueeze(1)
            x = torch.stack([x_fw,x_bw],dim=1).view(batch_size,2, edge_num, channel)

        if self.act is None:
            return x
        return self.act(x)          
        
    def forward4symmetric_mat(self,x,bw_idx,edge_index):
        x = torch.cat(
            [x, torch.index_select(x, 0, bw_idx)],
            dim=1,
        )
        return self.encode(x,edge_index[0],di=0)
    
    def forward(self, xs, A_to_B_edge_idx, B_to_A_edge_idx,first_layer = False):   
        if self.symmetric_mat:
            return self.forward4symmetric_mat(xs,bw_idx,edge_index)

        if first_layer:
            _, _, batch_size, edge_num, _ = xs.size()

            x_fw = torch.cat(
                [xs[0,0], xs[0,1]]
                , dim = 2
            )
            x_bw = torch.cat(
                [xs[1,0], xs[1,1]]
                , dim = 2
            )

            z = self.encode(torch.stack([x_fw,x_bw],dim=1),torch.stack([A_to_B_edge_idx[:,0], B_to_A_edge_idx[:,0]],dim=1),0)

        else:
            _, batch_size, edge_num, _ = xs.size()

            x_fw = torch.cat(# 
                [xs[0], torch.stack([ torch.index_select(a, 0, i) for a, i in zip(scatter_mean(xs[1],B_to_A_edge_idx[:,0],dim=1), A_to_B_edge_idx[:,1]) ])] #torch.index_select(xs[1], 0, edge_index[:,1])]
            , dim = 2)
            x_bw = torch.cat(
                [xs[1], torch.stack([ torch.index_select(a, 0, i) for a, i in zip(scatter_mean(xs[0],A_to_B_edge_idx[:,0],dim=1), B_to_A_edge_idx[:,1]) ])]# torch.index_select(xs[0], 0, edge_index[:,0])],
            , dim = 2)

            z = self.encode(torch.stack([x_fw,x_bw],dim=1),torch.stack([A_to_B_edge_idx[:,0], B_to_A_edge_idx[:,0]],dim=1),0)

        return z.transpose(0,1)
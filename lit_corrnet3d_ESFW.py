from argparse import ArgumentParser
from ast import arg



import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision import transforms

import os
from knn_cuda import KNN 
import pointnet2_ops._ext as _ext
import numpy as np
import pickle
from tqdm import tqdm
import time

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import fps
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max, scatter_min,scatter_sum
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

from feature_weaving_layer_sparse import SparseFeatureWeavingLayer

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Encoder(pl.LightningModule): 
    def __init__(self, enc_emb_dim=128, enc_glb_dim=1024, k_nn=20):
        super().__init__()
        self.k = k_nn    
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.GroupNorm(32,512)
        self.bn6 = nn.GroupNorm(32,512)
        self.bn7 = nn.GroupNorm(32,256)
        self.bn8 = nn.GroupNorm(32,128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),self.bn3,nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),self.bn4,nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, enc_glb_dim//2, kernel_size=1, bias=False),self.bn5,nn.LeakyReLU(negative_slope=0.2))

        self.mlp = nn.Sequential(   
            nn.Conv1d(64+64+128+256+enc_glb_dim, 512, 1),
            self.bn6,
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            self.bn7,
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            self.bn8,
            nn.ReLU())

    def _get_graph_feature(self, x, k=20, idx=None):

        def knn(x, k): 
            inner = -2*torch.matmul(x.transpose(2, 1), x) 
            xx = torch.sum(x**2, dim=1, keepdim=True)  
            pairwise_distance = -xx - inner - xx.transpose(2, 1)  
            idx = pairwise_distance.topk(k=k, dim=-1)[1]  
            return idx

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points) 
        if idx is None:
            idx = knn(x, k=k)
        device = idx.device 
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()  
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, transpose_xyz): 
        x = transpose_xyz
        batch_size = x.size(0)
        num_points=x.size(2)
        x = self._get_graph_feature(x, self.k)  
        x = self.conv1(x) 
        x1 = x.max(dim=-1, keepdim=False)[0] 
        x = self._get_graph_feature(x1, self.k) 
        x = self.conv2(x) 
        x2 = x.max(dim=-1, keepdim=False)[0] 
        x = self._get_graph_feature(x2, self.k)
        x = self.conv3(x) 
        x3 = x.max(dim=-1, keepdim=False)[0] 
        x = self._get_graph_feature(x3, self.k) 
        x = self.conv4(x) 
        x4 = x.max(dim=-1, keepdim=False)[0] 
        x = torch.cat((x1, x2, x3, x4), dim=1) 
        local_concat = x 
        x = self.conv5(x) 
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)                      
        global_vector = x                               
        repeat_glb_feat = global_vector.unsqueeze(-1).expand(batch_size, global_vector.shape[1], num_points)
        x = torch.cat((local_concat, repeat_glb_feat), 1)  
        embedding_feat = self.mlp(x)                   
        return embedding_feat, global_vector.unsqueeze(-1)  

class norm(pl.LightningModule):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True) 
        std = torch.std(x, self.axis,keepdim=True)   
        x = (x-mean)/(std+1e-6) 
        return x

class Gradient(torch.autograd.Function):                                                                                                       
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GroupingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None
grouping_operation = GroupingOperation.apply

class Modified_softmax(pl.LightningModule):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x
def normalize(dis):
    dis_max = torch.max(dis,dim=2)[0].unsqueeze(-1).repeat(1,1,1024)
    dis_min = torch.min(dis,dim=2)[0].unsqueeze(-1).repeat(1,1,1024)
    dis = (dis - dis_min)/(dis_max-dis_min)
    return dis

class Net4VertexEst(pl.LightningModule):
    def __init__(self,in_channels,num_cats,out_channels=16,mid_channels=16,num_layers=6):
        super(Net4VertexEst,self).__init__()
        
        self.l1 = SparseFeatureWeavingLayer(in_channels*2,out_channels,mid_channels)
        self.num_layers = num_layers-2

        self.layers = [SparseFeatureWeavingLayer(out_channels*2,out_channels,mid_channels) for i in range(self.num_layers)]
        self.layers.append(SparseFeatureWeavingLayer(out_channels*2,num_cats,mid_channels,use_batch_norm=False,activation=None))
        self.layers = nn.ModuleList(self.layers)

        
        
    def forward(self, A_to_B_edge_feat, B_to_A_edge_feat, A_to_B_edge_idx, B_to_A_edge_idx, similarity_A, similarity_B):

        zs = self.l1(torch.stack([A_to_B_edge_feat, B_to_A_edge_feat]), A_to_B_edge_idx, B_to_A_edge_idx, first_layer=True)
        zs = self.layers[0](zs,A_to_B_edge_idx, B_to_A_edge_idx)
        keep_zs = zs

        for i,L in enumerate(self.layers[1:]):
            if i%2==1:
                zs = zs+keep_zs
                keep_zs = zs
            zs = L(zs,A_to_B_edge_idx, B_to_A_edge_idx)

        batch_size, _, num_edge = A_to_B_edge_idx.size()
        index_A = A_to_B_edge_idx[:,0]*1024 + A_to_B_edge_idx[:,1]
        index_B = B_to_A_edge_idx[:,0]*1024 + B_to_A_edge_idx[:,1]

        out_A = torch.zeros(batch_size,1024*1024,device=self.device)
        out_B = torch.zeros(batch_size,1024*1024,device=self.device)

        for i in range(batch_size):
            out_A[i, index_A[i]] = zs.squeeze(-1)[0,i]
            out_B[i, index_B[i]] = zs.squeeze(-1)[1,i]

        z = (out_A.view(batch_size,1024,1024) + out_B.view(batch_size,1024,1024).transpose(1,2))/2
        return z


class LitCorrNet3D(pl.LightningModule):
    pretrained_urls = {}
    def __init__(
        self,
        input_pts: int,
        k_nn: int,

       # for ESFW
        k_nn_ESFW: int,
        D_g: int,
        L: int,


        enc_emb_dim: int = 128,
        enc_glb_dim: int = 1024,
        ls_coeff: list = [10.0, 1.0, 0.1],
        dec_in_dim: int = 1024+3,
        lr: float = 1e-2,


        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.count = 0
        self.input_pts = input_pts
        self.enc_emb_dim = enc_emb_dim
        self.enc_glb_dim = enc_glb_dim
        self.dec_in_dim = enc_glb_dim + 3
        self.rec_coeff = ls_coeff[0]
        self.rank_coeff = ls_coeff[1]
        self.mfd_coeff = ls_coeff[2]
        self.k_nn = k_nn
        self.k_nn_ESFW = k_nn_ESFW
        self.D_g = D_g
        self.L = L 
        
        self.encoder  = Encoder(self.enc_emb_dim, self.enc_glb_dim, self.k_nn)
        self.ESFW = Net4VertexEst(self.enc_emb_dim+1,1,out_channels=self.D_g,mid_channels=self.D_g,num_layers=self.L)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=self.dec_in_dim, out_channels=self.dec_in_dim,      kernel_size=1),
            nn.BatchNorm1d(self.dec_in_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.dec_in_dim, out_channels=self.dec_in_dim//2, kernel_size=1),
            nn.BatchNorm1d(self.dec_in_dim//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.dec_in_dim//2, out_channels=self.dec_in_dim//4, kernel_size=1),
            nn.BatchNorm1d(self.dec_in_dim//4),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.dec_in_dim//4, out_channels=3, kernel_size=1),
            nn.Tanh(),
            ) 
        self.train_loss_last_step = []

    def _corr_to_list(self, corr_matrix):
        single_corr_list = []
        pair = torch.nonzero(corr_matrix.clone().detach())
        for each in pair.tolist():
            single_corr_list.append(tuple(each))
        return single_corr_list

    def _label_ACC_percentage_for_inference(self, label_in, label_gt):
        assert(label_in.shape==label_gt.shape)
        bsize = label_in.shape[0]
        b_acc=[]
        for i in range(bsize):
            
            element_product = torch.mul(label_in[i], label_gt[i])  
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1) #N1x1

            hit = (sum_row != 0).sum()
            acc = hit.float()/torch.tensor(N1).float()
            b_acc.append(acc*100.0)
        mean = torch.mean(torch.stack(b_acc))  
        return mean

    def _prob_to_corr_test(self, prob_matrix):
        c = torch.zeros_like(input=prob_matrix)
        idx = torch.argmax(input=prob_matrix, dim=2, keepdim=True)
        for bsize in range(c.shape[0]):
            for each_row in range(c.shape[1]):
                c[bsize][each_row][ idx[bsize][each_row] ]=1.0 

        return c


    def _KFNN(self, x, y, k=10):
        def batched_pairwise_dist(a, b):
            x, y = a.float(), b.float()
            bs, num_points_x, points_dim = x.size()
            bs, num_points_y, points_dim = y.size()
            assert(num_points_x==1024 or num_points_x==256 or num_points_x==512 or num_points_x==64)

            xx = torch.pow(x, 2).sum(2)
            yy = torch.pow(y, 2).sum(2)
            zz = torch.bmm(x, y.transpose(2, 1))
            rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) 
            ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) 
            P = rx.transpose(2, 1) + ry - 2 * zz
            return P
        
        pairwise_distance = batched_pairwise_dist(x.permute(0,2,1), y.permute(0,2,1))
        similarity = 1/(pairwise_distance + 1e-6) 
        idx = similarity.topk(k=k, dim=-1)[1]
        return similarity, idx

    @staticmethod
    def pretrained_weights_available():
        return list(LitCorrNet3D.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in LitCorrNet3D.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')
        return self.load_from_checkpoint(LitCorrNet3D.pretrained_urls[checkpoint_name], strict=False)

    def weavenet_preprocess(self, A_feat, B_feat, A_to_B_out_point_idx, B_to_A_out_point_idx,similarity_A, similarity_B):#):#(self, similarity, out_point_idx):
        batch_size, _, input_pts = A_feat.size()

        # in point idx
        in_point_idx = torch.arange(input_pts,device=self.device).view(input_pts,-1).repeat(1,self.k_nn_ESFW).view(-1)
        in_point_idx = torch.repeat_interleave(in_point_idx.unsqueeze(0),batch_size,dim=0)

        # edge idx
        A_to_B_edge_idx = torch.cat([in_point_idx,A_to_B_out_point_idx.view(batch_size,-1)],dim=1).view(batch_size,2,-1)
        B_to_A_edge_idx = torch.cat([in_point_idx,B_to_A_out_point_idx.view(batch_size,-1)],dim=1).view(batch_size,2,-1)


        # edge feature
        batch_size, input_pts, knn = A_to_B_out_point_idx.size()
        flatten_size = batch_size*input_pts*knn
        edge_num = input_pts*knn

        # A to B
        similarity_A_add = torch.zeros(batch_size,edge_num,device=self.device)
        similarity_B_add = torch.zeros(batch_size,edge_num,device=self.device)
        for i in range(batch_size):
            knn_index = A_to_B_edge_idx[i,0],A_to_B_edge_idx[i,1]
            similarity_A_add[i] = similarity_A[i].squeeze(0)[knn_index]
            knn_index = A_to_B_edge_idx[i,1],A_to_B_edge_idx[i,0]
            similarity_B_add[i] = similarity_B[i].squeeze(0)[knn_index]

        A_to_B_edge_feat_in = torch.stack([torch.index_select(a, 0, i) for a, i in zip(A_feat.transpose(1,2), in_point_idx)])
        A_to_B_edge_feat_in = torch.cat([similarity_A_add.unsqueeze(-1), A_to_B_edge_feat_in],dim=2)
        A_to_B_edge_feat_out = torch.stack([torch.index_select(a, 0, i) for a, i in zip(B_feat.transpose(1,2), A_to_B_out_point_idx.view(batch_size,-1))])
        A_to_B_edge_feat_out = torch.cat([similarity_B_add.unsqueeze(-1), A_to_B_edge_feat_out],dim=2)
        A_to_B_edge_feat = torch.stack([A_to_B_edge_feat_in,A_to_B_edge_feat_out])


        # B to A
        similarity_A_add = torch.zeros(batch_size,edge_num,device=self.device)
        similarity_B_add = torch.zeros(batch_size,edge_num,device=self.device)
        for i in range(batch_size):
            knn_index = B_to_A_edge_idx[i,0],B_to_A_edge_idx[i,1]
            similarity_B_add[i] = similarity_B[i].squeeze(0)[knn_index]
            knn_index = B_to_A_edge_idx[i,1],B_to_A_edge_idx[i,0]
            similarity_A_add[i] = similarity_A[i].squeeze(0)[knn_index]


        B_to_A_edge_feat_in = torch.stack([torch.index_select(a, 0, i) for a, i in zip(B_feat.transpose(1,2), in_point_idx)])
        B_to_A_edge_feat_in = torch.cat([similarity_B_add.unsqueeze(-1), B_to_A_edge_feat_in],dim=2)
        B_to_A_edge_feat_out = torch.stack([torch.index_select(a, 0, i) for a, i in zip(A_feat.transpose(1,2), B_to_A_out_point_idx.view(batch_size,-1))])
        B_to_A_edge_feat_out = torch.cat([similarity_A_add.unsqueeze(-1), B_to_A_edge_feat_out],dim=2)
        B_to_A_edge_feat = torch.stack([B_to_A_edge_feat_in,B_to_A_edge_feat_out])

        return A_to_B_edge_feat, B_to_A_edge_feat, A_to_B_edge_idx, B_to_A_edge_idx



    def forward(self, xyz1, xyz2): 
        HIER_feat1, pooling_feat1 = self.encoder(transpose_xyz=xyz1.transpose(1, 2)) 
        HIER_feat2, pooling_feat2 = self.encoder(transpose_xyz=xyz2.transpose(1, 2))   

        pairwise_distance, _ = self._KFNN(HIER_feat1, HIER_feat2)
        similarity = 1/(pairwise_distance + 1e-6) 
        p = self.DeSmooth(similarity.transpose(1,2).contiguous()).transpose(1,2).contiguous() 
        return p 
    
    def _run_step(self, xyz1, xyz2):
        HIER_feat1, pooling_feat1 = self.encoder(transpose_xyz=xyz1.transpose(1, 2)) 
        HIER_feat2, pooling_feat2 = self.encoder(transpose_xyz=xyz2.transpose(1, 2))

        # Preprocessing
        similarity_A, A_to_B_out_point_idx = self._KFNN(HIER_feat1, HIER_feat2, k=self.k_nn_ESFW)
        similarity_B, B_to_A_out_point_idx = self._KFNN(HIER_feat2, HIER_feat1, k=self.k_nn_ESFW)
        A_to_B_edge_feat, B_to_A_edge_feat, A_to_B_edge_idx, B_to_A_edge_idx = self.weavenet_preprocess(HIER_feat1, HIER_feat2,A_to_B_out_point_idx,B_to_A_out_point_idx, similarity_A, similarity_B)#, A_to_B_out_point_idx, B_to_A_out_point_idx)

        # Our network process
        p = self.ESFW(A_to_B_edge_feat, B_to_A_edge_feat, A_to_B_edge_idx, B_to_A_edge_idx,similarity_A,similarity_B)

        rand_grid_a=torch.bmm(xyz2.transpose(1, 2), p.transpose(2,1).contiguous()) 
        ya = pooling_feat1.expand(pooling_feat1.size(0), pooling_feat1.size(1), rand_grid_a.size(2))
        ya = torch.cat((rand_grid_a, ya), 1)
        out_a = 2*self.decoder(ya)

        rand_grid_b=torch.bmm(xyz1.transpose(1, 2), p)                             
        yb = pooling_feat2.expand(pooling_feat2.size(0), pooling_feat2.size(1), rand_grid_b.size(2)) 
        yb = torch.cat((rand_grid_b, yb), 1)     
        out_b = 2*self.decoder(yb)

        return p, out_a, out_b

    def _batch_frobenius_norm(self, matrix1, matrix2):
        loss_F = torch.norm((matrix1-matrix2),dim=(1,2))
        return loss_F

    def _knn_point(self, nsample, ref, query):
        knn = KNN(k=nsample, transpose_mode=True)
        dist, indx = knn(ref, query)
        return dist, indx

    def _get_manifold_loss(self, pred_flow, xyz1, K_FOR_KNN):
        batch_size = xyz1.shape[0]
        N = xyz1.shape[1]
        if K_FOR_KNN==None:
            K_FOR_KNN=5
        else:
            K_FOR_KNN=K_FOR_KNN
        my_k=K_FOR_KNN
        val, idx = self._knn_point(nsample=my_k, ref=xyz1, query=xyz1) 
        grouped_xyz1 = grouping_operation(xyz1.transpose(1, 2).contiguous(), idx.int()).permute(0,2,3,1) 
        grouped_flow = grouping_operation(pred_flow.transpose(1, 2).contiguous(), idx.int()).permute(0,2,3,1)  
        grouped_xyz1diff = grouped_xyz1 - torch.unsqueeze(xyz1, 2) 
        grouped_flowdiff = grouped_flow - torch.unsqueeze(pred_flow, 2)
        dist_square = torch.sum(input=grouped_xyz1diff ** 2, dim=-1)
        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-dist_square/GAUSSIAN_HEAT_KERNEL_T)
        flow_square =  torch.sum(input=grouped_flowdiff ** 2, dim=-1) 
        USE_GUASSIAN_MANIFOLD=True
        if USE_GUASSIAN_MANIFOLD==False:
            manifold_loss = torch.mul(1.0/(dist_square+1e-10), flow_square) 
        elif USE_GUASSIAN_MANIFOLD==True:
            manifold_loss = torch.mul(gaussian_heat_kernel, flow_square) 
        manifold_loss =  torch.sum(input=manifold_loss, dim=-1)/my_k 
        manifold_loss = torch.sum(input=manifold_loss, dim=-1)/N 
        manifold_loss = torch.sum(input=manifold_loss, dim=-1)/batch_size 
        return manifold_loss

    def _run_loss(self, txyz1, txyz2, p, out_a, out_b): 
        xyz1=txyz1
        xyz2=txyz2
        bmean_loss_F1 = torch.mean(self._batch_frobenius_norm(xyz1,out_a))
        bmean_loss_F2 = torch.mean(self._batch_frobenius_norm(xyz2,out_b))
        I_N1 = torch.eye(n=xyz1.shape[2], device=self.device) 
        bsize=xyz1.shape[0]
        I_N1 = I_N1.unsqueeze(0).repeat(bsize,1,1)
        bmean_P_rankLoss = torch.mean(self._batch_frobenius_norm(torch.bmm(p,p.transpose(2, 1).contiguous()), I_N1.float())) 
        pred_corr=torch.bmm(p, xyz2.transpose(2, 1).contiguous()) 
        
        manifold_loss = self._get_manifold_loss(pred_corr, xyz1.transpose(2, 1).contiguous(), K_FOR_KNN=10)
        pred_corr2=torch.bmm(p.transpose(2, 1).contiguous(), xyz1.transpose(2, 1).contiguous()) 
        manifold_loss_2 = self._get_manifold_loss(pred_corr2, xyz2.transpose(2, 1).contiguous(), K_FOR_KNN=10)  
        
        rec_term = (bmean_loss_F1+bmean_loss_F2)
        rank_term = bmean_P_rankLoss
        mfd_term = (manifold_loss+manifold_loss_2)

        return rec_term, rank_term, mfd_term
        
    def step(self, batch, batch_idx):
        label, pinput1, input2, index_ = batch

        p, out_a, out_b =self._run_step(pinput1,input2)
        rec_term, rank_term, mfd_term = self._run_loss(pinput1.transpose(2, 1), input2.transpose(2, 1), p, out_a, out_b)

        loss = self.rec_coeff*rec_term + self.rank_coeff*rank_term + self.mfd_coeff*mfd_term
        logs = {
            "rec_loss(x{})".format(str(self.rec_coeff)): rec_term,
            "rank_loss(x{})".format(str(self.rank_coeff)): rank_term,
            "mfd_loss(x{})".format(str(self.mfd_coeff)): mfd_term
        }
        self.train_loss_last_step.append(loss.item())
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True)
        self.log(name = 'train_loss',value=loss.item(),on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        
        loss, logs = self.step(val_batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        self.log(name = 'val_loss',value=loss.item(), on_step=True, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        data_dict = test_batch
        label=data_dict['label_flat']
        pinput1=data_dict['src_flat']
        input2=data_dict['tgt_flat']
        s_label=[]
        ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        for each_ratio in ratio_list:
            key_name = 'sl_'+str(each_ratio).replace('.', '')
            s_label.append(data_dict[key_name])

        p, out_a, out_b =self._run_step(pinput1,input2)

        corr_tensor = self._prob_to_corr_test(p) 
        acc_000 = self._label_ACC_percentage_for_inference(corr_tensor , label)
        logs = {
            "acc_0.00": acc_000
        }
        for k_ in range(len(ratio_list)):
            logs["acc_"+str(ratio_list[k_])] = self._label_ACC_percentage_for_inference(corr_tensor , s_label[k_])
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument(
            "--enc_emb_dim",
            type=int,
            default=128,
            help=""
        )
        parser.add_argument(
            "--enc_glb_dim",
            type=int,
            default=1024,
            help=""
        )
        parser.add_argument('--ls_coeff', type=float, nargs='+', default=[10.0, 1.0, 0.1])
        parser.add_argument('--k_nn', type=int, default=20)
        parser.add_argument("--dec_in_dim", type=int, default=1024+3)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--test_data_dir", type=str, default=None)
        parser.add_argument("--num_gpus", type=int, default=1)

        # ESFW hyperparameter
        parser.add_argument("--L",type=int, default=10)
        parser.add_argument("--D_g",type=int, default=16)
        parser.add_argument("--k_nn_ESFW",type=int, default=16)        
        return parser


def cli_main(args=None):
    from lit_dataset_clean import HumanDataModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    pl.seed_everything()
    parser = ArgumentParser()
    parser.add_argument("--ckpt_user", type=str, default=None)
    parser.add_argument("--dataset", default="human", type=str, choices=["man", "stl10", "imagenet"])
    parser.add_argument("--batch_size", type=int, default=20)
    script_args, _ = parser.parse_known_args(args)
    dm_human = HumanDataModule
    parser = LitCorrNet3D.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    dm = dm_human.from_argparse_args(args)
    
    args.input_pts = 1024
    model = LitCorrNet3D(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, gpus=args.num_gpus, benchmark=True, accelerator='dp')
    trainer.fit(model, dm)
    return dm, model, trainer

def cli_main_test_(args=None):
    from lit_dataset_clean import testset_pytable_with_soft_label
    pl.seed_everything(seed=1)
    parser = ArgumentParser()
    parser.add_argument("--ckpt_user", type=str)
    parser.add_argument("--batch_size", type=int, default=20)
    test_data = "SUN"
    parser = LitCorrNet3D.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args) 
    args.input_pts = 1024
    test_dataset_3dcoded = testset_pytable_with_soft_label(
        test_h5file_name=args.test_data_dir, 
        outname='nonrigid_surreal',
        show=False)
    print('len of test: ',len(test_dataset_3dcoded))
    print('bsize: ',args.batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset_3dcoded, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)
    model = LitCorrNet3D(**vars(args))
    hparafiledir = args.ckpt_user.split('/')[0] + '/' + args.ckpt_user.split('/')[1] + '/' + 'hparams.yaml'
    print(hparafiledir)
    model_test = model.load_from_checkpoint(
        args.ckpt_user,
    hparams_file=hparafiledir)
    trainer = pl.Trainer.from_argparse_args(args, gpus=args.num_gpus, benchmark=True) 
    trainer.test(model = model_test, test_dataloaders = testloader)

    return

if __name__ == '__main__':
    cli_main()       
    # cli_main_test_()










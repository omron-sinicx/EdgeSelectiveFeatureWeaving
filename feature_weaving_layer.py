import torch
import torch.nn as nn

class _FeatureWeavingLayer(nn.Module):
    """
    The base class for feature weaving layers
    """
    def __init__(self, encoders):
        """
        Args: 
            adjacent_edge_encoder: a layer that encodes XD feature map.
            X (int): the number of axis. default: 2
                        
        """
        assert(type(encoders)==list)
        super(_FeatureWeavingLayer,self).__init__()

        self.encoders = nn.ModuleList()
        for e in encoders:
            self.encoders.append(e)
    
    @staticmethod
    def initialize(zs):
        # set dimension order
        K = len(zs)
        idx = list(range(2,K+2))
        return [z.permute(0,1,*_FeatureWeavingLayer.rotate(idx,k,to_left=False)) for k,z in enumerate(zs)]
    
    @staticmethod
    def finalize(zs):
        return sum(zs)/len(zs)
    
    @staticmethod
    def rotate(arr, k, to_left=True):
        K = len(arr)
        k %= K
        if not to_left:
            k = K-k
        return arr[k:] + arr[:k]
            
    def forward(self, zs):
        #print("zs",zs)
        assert(len(zs)==len(zs[0].shape)-2)
        
        zcats = [torch.cat(self.rotate(zs,d),dim=1) for d in range(len(zs))]
        #print("zcats", zcats)
        zs = [enc(z) for z,enc in zip(zcats,self.encoders)]
        return zs

class EncoderMaxPool(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 dim: int = 2,
                 convXd: 'ConvXd' = nn.Conv2d,
                 BatchNormXd = nn.BatchNorm2d,
                 activation = nn.PReLU()
                ):
        super(EncoderMaxPool,self).__init__()  
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.convXd = convXd
        self.bnXd = BatchNormXd
        self.act = activation        
        self.has_built = False
        
    def build(self):
        self.conv_max = self.convXd(self.in_channels, self.mid_channels, kernel_size=1)
        self.conv = self.convXd(self.in_channels+self.mid_channels, self.out_channels, kernel_size=1)
        if self.bnXd is not None:
            self.bn = self.bnXd(self.out_channels)
        else:
            self.bn = None
        self.has_built = True
        
    def clone(self,dim):
        if not self.has_built:
            self.build()
        e = EncoderMaxPool(self.in_channels,self.out_channels,self.mid_channels,dim=dim)
        e.conv_max = self.conv_max
        e.conv = self.conv
        e.bn = self.bn
        e.act = self.act        
        e.has_built = True
        return e
                           
    def forward(self,x):
        assert(1<self.dim and self.dim<len(x.shape)) 
        z = self.conv_max(x)        
        N = x.shape[self.dim]
        z = z.max(self.dim,keepdim=True)[0]
        z = z.repeat_interleave(N,self.dim)
        x = torch.cat([x,z],dim=1)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)
        
class FeatureWeavingLayer(_FeatureWeavingLayer):
    def __init__(self,in_channels,out_channels,mid_channels,D=2,use_batch_norm=True, activation=nn.PReLU(), asymmetric=False):

        convXd = self.get_ConvXd(D)
        if use_batch_norm:
            BNXd = self.get_BNXd(D)
        else:
            BNXd = None
        
        if asymmetric:
            encoders = [
                EncoderMaxPool(in_channels,out_channels,mid_channels,dim=d+2,convXd=convXd,BatchNormXd=BNXd,activation=activation) for d in range(D)
            ]
            [e.build() for e in encoders]
        else:
            # symmetric: encoder weights are shared among dims.
            encoder = EncoderMaxPool(in_channels,out_channels,mid_channels,dim=2,convXd=convXd,BatchNormXd=BNXd,activation=activation)
            encoder.build()        
            encoders = [encoder.clone(d+2) for d in range(D)]
            
        super(FeatureWeavingLayer,self).__init__(encoders)
    
    @staticmethod
    def get_ConvXd(D):
        assert(D>1)
        if D==2:
            return nn.Conv2d
        elif D==3:
            return nn.Conv3d
        else:
            raise NotImplementedError("nn.Conv{}D has not been implemented.".format(D))
        return None
    
    @staticmethod
    def get_BNXd(D):
        assert(D>1)
        if D==2:
            return nn.BatchNorm2d
        elif D==3:
            return nn.BatchNorm3d
        else:
            raise NotImplementedError("nn.BatchNorm{}D has not been implemented.".format(D))
        return None
        
        
        

        
        
    
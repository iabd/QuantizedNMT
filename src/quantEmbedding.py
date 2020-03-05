import torch, pdb, math
import  torch.nn as  nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import  torch.nn.init as init
from quantizeWeights import *
from quantLinear import FakeQuantizationSTE
_fakeQuantize = FakeQuantizationSTE.apply
class Embedding(nn.Module):
    def __init__(self, numEmbeddings, embeddingDim, mode='none', weightBits=8, requantizeOutput=True, paddingIdx=None, maxNorm=None, normType=2, scaleGradByFreq=False, sparse=False, _weight=None, startStep=0):
        super(Embedding, self).__init__()
        
        assert mode in ['automp', 'none', 'manmp'], "mode must be either training, inference or none"
        self.embeddingDim=embeddingDim
        self.mode=mode
        self.startStep=startStep
        if self.mode=='manmp':
            assert weightBits>=2, "Weight bits must be higher than 1"
            self.weightBits=weightBits
            self.requantizeOutput=requantizeOutput

        self.paddingIdx=paddingIdx
        if self.paddingIdx is not None:
            if paddingIdx>0:
                assert self.paddingIdx<self.numEmbeddings, "padding index must be within numEmbeddings"
            elif paddingIdx<0:
                assert paddingIdx>=-self.numEmbeddings, "padding index must be within numEmbeddings"
                self.paddingIdx=self.numEmbeddings+paddingIdx
        else:
            self.paddingIdx=-1
        self.paddingIdx=paddingIdx
        self.maxNorm=maxNorm
        self.normType=normType
        self.scaleGradByFreq=scaleGradByFreq
        if _weight is None:
            self.weight=Parameter(torch.Tensor(embeddingDim, numEmbeddings))
            self.initParameters()
        else:
            assert list(_weight.shape)==[numEmbeddings, embeddingDim], "Shape of weight does not match numEmbeddings and embeddingDim"
            self.weight=Parameter(_weight)
        self.sparse=sparse
        self.register_buffer('_step', torch.zeros(1))

    def initParameters(self):
        init.normal_(self.weight)
        if self.paddingIdx is not None:
            with torch.no_grad():
                self.weight[self.paddingIdx].fill_(0)

    def extra_repr(self):
        return "embeddingDim: {}, mode: {}".format(self.embeddingDim, self.mode)

    @property
    def weightScale(self):
        return getScale(self.weight, self.weightBits)

    @property
    def fakeQuantizedWeight(self):
        return _fakeQuantize(self.weight, self.weightBits)

    def quantizedForward(self, input):
        return F.embedding(input, self.fakeQuantizedWeight, self.paddingIdx, self.maxNorm, self.normType, self.scaleGradByFreq, self.sparse)


    def forward(self, input):
        if self.mode is "automp":
            print("BROKEN WINDOW")
            import sys
            sys.exit()
        if self.mode=="none":
            return F.embedding(input, self.weight, self.paddingIdx, self.maxNorm, self.normType, self.scaleGradByFreq, self.sparse)
        if self._step>=self.startStep:
            out=self.quantizedForward(input)
        else:
            out=F.embedding(input, self.weight, self.paddingIdx, self.maxNorm, self.normType, self.scaleGradByFreq, self.sparse)
        if self.training:
            self._step+=1
        return out


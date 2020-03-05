import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from attention import MultiheadAttention
from quantEmbedding import Embedding
from quantLinear import Linear

#code heavily borrowed from The Annotated Transformer [ http://nlp.seas.harvard.edu/2018/04/03/attention.html ] 

clones=lambda module, N: nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) 

def subsequentMask(size):
    shape=(1,size, size)
    mask=np.triu(np.ones(shape), k=1).astype('uint8')
    mask=torch.from_numpy(mask)==0
    return mask

class Embeddings(nn.Module):
    def __init__(self, dModel, vocab, bitwidth=None):
        super(Embeddings, self).__init__()
        self.lut=Embedding(vocab, dModel)
        self.dModel=dModel
        self.bitwidth=bitwidth
    def forward(self, x):
        return self.lut(x)*math.sqrt(self.dModel)
            

class PositionalEncodings(nn.Module):
    def __init__(self, dModel, dropout, maxlen=1000, big=False):
        super(PositionalEncodings, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.dModel=dModel
        pe=torch.zeros(maxlen, dModel)
        position=torch.arange(0., maxlen).unsqueeze(1)
        divTerm=torch.exp(torch.arange(0., dModel, 2)*-math.log(10000.0)/dModel)
        pe[:, 0::2]=torch.sin(position * divTerm)
        pe[:,1::2]=torch.cos(position*divTerm)
        pe=pe.unsqueeze(0)
        self.big=big
        self.register_buffer('pe', pe)
    def forward(self, x):
        if self.big:
            x=x[:, :, :1024]
        else:
            x=x[:,:,:512]
        x=x+Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return self.dropout(x)



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2=nn.Parameter(torch.ones(features))
        self.b2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self, x):
        mean=x.mean(-1, keepdim=True)
        std=x.std(-1, keepdim=True)
        norm=self.a2*(x-mean)/(std+self.eps)+self.b2
        return norm

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class Feedforward(nn.Module):
    def __init__(self, dModel, dff, dropout, mode, bias, **manmpargs):
        super(Feedforward, self).__init__()
        self.w1=Linear(dModel, dff, mode, bias, **manmpargs)
        self.w2=Linear(dff, dModel, mode, bias, **manmpargs)
        self.dropout=nn.Dropout(p=dropout)
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, selfAttention, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.selfAttention=selfAttention
        self.size=size
        self.ff=ff
        self.sublayer=clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x=self.sublayer[0](x, lambda x:self.selfAttention(x,x,x,mask))
        return self.sublayer[1](x, self.ff)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers=clones(layer, N)
        self.norm=LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x=layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, selfAttention, sourceAttention, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size=size
        self.selfAttention=selfAttention
        self.sourceAttention=sourceAttention
        self.ff=ff
        self.sublayer=clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, sourceMask, tgtMask):
        x=self.sublayer[0](x, lambda x:self.selfAttention(x,x,x,tgtMask))
        x=self.sublayer[1](x, lambda x:self.sourceAttention(x, memory, memory, sourceMask))
        return self.sublayer[2](x, self.ff)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers=clones(layer, N)
        self.norm=LayerNorm(layer.size)
    def forward(self, x, memory, sourceMask, tgtMask):
        for layer in self.layers:
            x=layer(x, memory, sourceMask, tgtMask)
        return self.norm(x)

class Generator(nn.Module):
    "linear+softmax"
    def __init__(self, dModel, vocab, mode, bias, **manmpargs):
        super(Generator, self).__init__()
        self.type=type
        self.linear=Linear(dModel, vocab, mode, bias, **manmpargs)

    def forward(self,x):
        return F.log_softmax(self.linear(x), dim=-1)

class EncoderOnly(nn.Module):
    """USED for BERT"""
    def __init__(self, encoder, srcEmbed, generator):
        super(EncoderOnly, self).__init__()
        self.encoder=encoder
        self.srcEmbed=srcEmbed
        self.generator=generator

    def forward(self, src, srcMask):
        return self.encoder(self.srcEmbed(src), srcMask)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, srcEmbed, tgtEmbed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.srcEmbed=srcEmbed
        self.tgtEmbed=tgtEmbed
        self.generator=generator

    def forward(self, src, tgt, srcMask, tgtMask):
        return self.decode(self.encode(src, srcMask), srcMask, tgt, tgtMask)

    def encode(self, src, srcMask):
        return self.encoder(self.srcEmbed(src), srcMask)

    def decode(self, memory, srcMask, tgt, tgtMask):
        return self.decoder(self.tgtEmbed(tgt), memory, srcMask, tgtMask)


def createModel(srcVocab, tgtVocab, modelType='small', N=6, dModel=512, dff=2048, h=8, dropout=0.1, mode='none',
                bias=True, type='full', **manmpargs):
    assert modelType in ['small', 'large'], "Invalid modelType provided, please choose small or large"
    assert type in ['full', 'bert'], "Invalid type, please make sure it's 'full' or 'bert' in case of transformer or bert respectively"
    big=False
    if modelType=='large':
        dModel=1024
        h=16
        big=True
    c=copy.deepcopy
    attention=MultiheadAttention(h, dModel, dropout,  mode, bias, **manmpargs)
    ff=Feedforward(dModel, dff, dropout, mode, bias, **manmpargs)
    position=PositionalEncodings(dModel, dropout, big=big)
    if type=='full':
        model=EncoderDecoder(
            Encoder(EncoderLayer(dModel, c(attention), c(ff), dropout), N),
            Decoder(DecoderLayer(dModel, c(attention), c(attention), c(ff), dropout), N),
            nn.Sequential(Embedding(dModel, srcVocab, mode, manmpargs['weightBits'] if bool(manmpargs) else None, manmpargs['requantizeOutputs'] if bool(manmpargs) else None), c(position)),
            nn.Sequential(Embedding(dModel, tgtVocab, mode, manmpargs['weightBits'] if bool(manmpargs) else None, manmpargs['requantizeOutputs'] if bool(manmpargs) else None), c(position)),
            Generator(dModel, tgtVocab, mode, bias, **manmpargs)
        )
    if type=='bert':
        model=EncoderOnly(
            Encoder(EncoderLayer(dModel, c(attention), c(ff), dropout),N),
            nn.Sequential(Embedding(dModel, srcVocab, mode, manmpargs['weightBits'] if bool(manmpargs)  else None, manmpargs['requantizeOutputs'] if bool(manmpargs) else None), c(position)),\
            Generator(dModel, srcVocab, mode, bias, **manmpargs)
        )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model


import torch, copy, math, pdb
import torch.nn as nn
from quantLinear import Linear
clones=lambda module, N: nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def selfAttention(q,k,v, mask=None, dropout=None):
    dk=q.size(-1)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:
        scores=scores.masked_fill(mask==0, -1e4)
        softmaxAttention=scores.softmax(dim=-1)
    if dropout is not None:
        softmaxAttention=dropout(softmaxAttention)
    final=torch.matmul(softmaxAttention, v)
    return final, softmaxAttention

class MultiheadAttention(nn.Module):
    def __init__(self, h, dModel, dropout=0.1, mode='none', bias=True, **manmpargs):
        super(MultiheadAttention, self).__init__()
        assert dModel%h==0
        self.dk=dModel//h
        self.h=h
        self.linears=clones(Linear(dModel, dModel, mode, bias, **manmpargs), 4)
        self.attention=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, q,k,v,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
        nBatches=q.size(0)
        q,k,v=[l(x).view(nBatches, -1, self.h, self.dk).transpose(1,2) for l, x in zip(self.linears, (q,k,v))]
        x, self.attention=selfAttention(q,k,v, mask=mask, dropout=self.dropout)
        x=x.transpose(1,2).contiguous().view(nBatches, -1, self.h*self.dk)
        return self.linears[-1](x)

    
        

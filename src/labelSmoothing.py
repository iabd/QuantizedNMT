import torch.nn as nn
import torch, pdb
from torch.autograd import Variable

class LabelSmoothing(nn.Module):

    """Label smoothing actually starts to penalize the model if it gets very confident about a given choice."""
    def __init__(self, size, paddingIdx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion=nn.KLDivLoss(size_average=False)
        self.smoothing=smoothing
        self.paddingIdx=paddingIdx
        self.confidence=1-smoothing
        self.size=size
        self.trueDist=None

    def forward(self, x, target):
        assert x.size(1)==self.size
        trueDist=x.data.clone()
        trueDist.fill_(self.smoothing/(self.size-2))
        trueDist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        trueDist[:,self.paddingIdx]=0
        mask=torch.nonzero(target.data==self.paddingIdx)
        if mask.dim()>0:
            trueDist.index_fill_(0, mask.squeeze(), 0.0)
        self.trueDist=trueDist
        return self.criterion(x, Variable(trueDist, requires_grad=False))



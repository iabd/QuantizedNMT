from torch.autograd import Variable
from transformer import subsequentMask
class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src=src
        self.srcMask=(src!=pad).unsqueeze(-2)
        if trg is not None:
            self.trg=trg[:, :-1]
            self.trgY=trg[:,1:]
            self.trgMask=self.makeStdMask(self.trg, pad)
            self.nTokens=(self.trgY!=pad).data.sum()

    @staticmethod
    def makeStdMask(tgt, pad):
        "create a mask to hide padding and future words"
        tgtMask=(tgt!=pad).unsqueeze(-2)
        tgtMask=tgtMask & Variable(subsequentMask(tgt.size(-1)).type_as(tgtMask.data))
        return tgtMask

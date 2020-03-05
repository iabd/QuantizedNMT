import torch, pdb, math
import  torch.nn as  nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import  torch.nn.init as init
from quantizeWeights import *

class FakeQuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, numBits=8, scale=None):
        if scale is None:
            scale=getScale(input, numBits)
        quant = quantize(input, numBits, scale)
        return deQuantize(quant, scale)

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None

_fakeQuantize=FakeQuantizationSTE.apply

class Linear(nn.Module):
    def __init__(self, inFeatures, outFeatures, mode='none', bias=True, startStep=0, inference=False, **manmpargs):
        super(Linear, self).__init__()
        assert mode in ['none', 'automp', 'manmp'], "mode must be either training, inference or none"
        self.inFeatures=inFeatures
        self.outFeatures=outFeatures
        self.mode=mode
        if self.mode=='manmp':
            globals().update(manmpargs)
            assert activationBits>=2, "Activation bits must be higher than 1"
            assert weightBits>=2, "Weight bits must be higher than 1"
            self.activationBits=activationBits
            self.weightBits=weightBits
            self.biasBits=32
            self.requantizeOutput=requantizeOutputs
        self.startStep=startStep
        self.weight=Parameter(torch.zeros((outFeatures, inFeatures)))
        if bias:
            self.bias=Parameter(torch.zeros(outFeatures))
        else:
            self.bias=self.regiter_parameter('bias', None)
        self.register_buffer('_step', torch.zeros(1))
        self.initParams()

    def initParams(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fanIn, _=init._calculate_fan_in_and_fan_out(self.weight)
            bound=1/math.sqrt(fanIn)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return "inFeatures: {}, outFeatures: {}, bias: {}, mode: {}".format(self.inFeatures, self.outFeatures, self.bias is not None, self.mode)

    @property
    def weightScale(self):
        return getScale(self.weight, self.weightBits)

    @property
    def fakeQuantizedWeight(self):
        return _fakeQuantize(self.weight, self.weightBits, self.weightScale)

    @property
    def quantizedWeight(self):
        quantWeight=quantize(self.weight, self.weightBits, self.weightScale)
        return quantWeight

#    @property
#    def weightOffset(self):
#        _, weightOffset=quantize(self.weight, self.weightBits, self.weightScale)
#        return weightOffset



    def quantizedBias(self, scale):
        try:
            bias=quantize(self.bias, self.biasBits, scale)
        except AttributeError:
            bias=None
        return bias

    def quantizedTrainForward(self, input):
        inputScale=getScale(input, self.activationBits)
        out=F.linear(_fakeQuantize(input, self.activationBits, inputScale), self.fakeQuantizedWeight, self.bias)
        if self.requantizeOutput:
            return _fakeQuantize(out, self.activationBits)
        return out

    def quantizedInferenceForward(self, input):
        inputScale=getScale(input, self.activationBits)
        inputQuant =quantize(input, self.activationBits, inputScale)
        dequantScale=self.weightScale*inputScale
        #dequantOffset=inputOffset+self.weightOffset
        out=F.linear(inputQuant, self.quantizedWeight, self.quantizedBias(scale=dequantScale))
        out=deQuantize(out, dequantScale)
        if self.requantizeOutput:
            return _fakeQuantize(out, self.activationBits)
        return out

    def autoMPForward(self, input):
        print("BROKEN WINDOW, PLEASE PRESS 'n' or 'c' TO TERMINATE PROGRAM")
        pdb.set_trace()
        import sys
        return sys.exit()

    def forward(self, input):
        if self.mode=='automp':
            out=self.autoMPForward(input)
            return out
        if self.mode=="none":
            return F.linear(input, self.weight, self.bias)
        if self.training:
            try:
                if self._step>=self.startStep:
                    out=self.quantizedTrainForward(input)
                else:
                    out=F.linear(input, self.weight, self.bias)
                self._step+=1
            except:
                pdb.set_trace()
            return out
        else:
            return  self.quantizedInferenceForward(input)



































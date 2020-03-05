import torch, pdb

def getScale(matrix, numBits, quantMax=None):
    with torch.set_grad_enabled(False):
        xMax=matrix.abs().max()
    if quantMax:
        return quantMax/xMax
    return (2**(numBits-1)-1)/xMax


def quantize(x, numBits=8, scale=None):
    quantMax=2**(numBits-1)-1
    quantMin=-quantMax
    if scale is None:
        scale=getScale(x, numBits, quantMax)

    x=x.mul(scale)
    quantized=torch.clamp(torch.round(x), quantMin, quantMax)
    return quantized



def deQuantize(x_, scale):
    return x_.div(scale)


def getSScale(matrix, quantMax=None):
    if quantMax is None:
        quantMax=2**(numBits-1)-1
    return matrix.abs().max()/quantMax

def uInt(x, numBits=8, quantMax=None):
    return torch.clamp(torch.round(x), -quantMax, quantMax)


def qquantize(xFloat, numBits=8, scale=None):
    quantMin = 0
    quantMax = 2 ** numBits - 1
    with torch.set_grad_enabled(False):
        xMax=xFloat.abs().max()
        xMin=xFloat.abs().min()

    if scale is None:
        scale=getScale(xFloat, quantMax)
    xOffset = uInt(quantMax - xMax / scale, numBits, quantMax)
    xQuant = xFloat / scale + xOffset
    return uInt(xQuant, numBits, quantMax), xOffset

def deQQuantize(xQuant, xScale, xOffset):
    xQuant=xQuant.type(torch.float32)
    xNormal=(xQuant-xOffset)*xScale
    return xNormal



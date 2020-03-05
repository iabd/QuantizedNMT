import functools
from quantizeWeights import *
x=torch.randn((9,9))
def nestedGetattr(object, attr, *args):
    def _getattr(object, attr):
        return getattr(object, attr, *args)
    return functools.reduce(_getattr, [object]+attr.split('.'))

def nestedSetattr(object, attr, value):
    pre, _, post=attr.rpartition('.')
    return setattr(nestedGetattr(object, pre) if pre else object, post, value)

def quantify(model, keywords=["weights"]):
    objList=[]
    for name, _ in model.named_parameters():
        if all(x in name for x in keywords):
            objList.append(name)

    for item in objList:
        nestedSetattr(model, item, torch.nn.Parameter(quantize(nestedGetattr(model, item)), requires_grad=False))


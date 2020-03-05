from torch.autograd import Variable
import torch, os
from collections import Counter
from batch import Batch
from transformer import subsequentMask

def floorToBase(num, base=8):
    ret = base * round(num / base)
    if ret > num:
        return ret - base
    return ret

def clipCounter(counter, n,leastCommon=True):
    if leastCommon:
        countr=Counter(dict(counter.most_common()[:-n-1:-1]))
    else:
        countr=Counter(dict(counter.most_common()[:n]))
    for items in countr:
        del counter[items]
    return counter

def customDict(argList, mainDict):
    retDict={}
    for items in argList:
        retDict[items]=mainDict[items]
    return retDict

def greedyDecode(model, src, srcMask, maxlen, startSymbol):
    ys=torch.ones(1,1,).fill_(startSymbol).type_as(src.data)
    for i in range(maxlen-1):
        out=model.decode(model.encode(src,srcMask), srcMask, Variable(ys), Variable(subsequentMask(ys.size(1)).type_as(src.data)))
        prob=model.generator(out[:,-1])
        _, nextWord=torch.max(prob, dim=1)
        nextWord=nextWord.data[0]
        ys=torch.cat([ys, torch.ones(1,1).type_as(src.data).fill_(nextWord)], dim=1)
    return ys


def getUniqueFolder(root, identifier):
    idx=1
    if not (os.path.exists(root)):
        os.mkdir(root)
    for folder in os.listdir(root):
        if folder.startswith(identifier):
            idx+=1

    return os.path.join(root, identifier+str(idx))


def rebatch(padIdx, batch):
    "fix order in torchtext to match ours"
    src, trg=batch.src.transpose(0,1), batch.trg.transpose(0,1)
    return Batch(src=src, trg=trg, pad=padIdx)


def log(mes, f="trainLog.txt"):
    try:
        with open(f, "a+") as file_:
            file_.write(mes)
    except:
        f.write(mes)

def modelSize(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def batchSizeFn(new, count, sofar):
    global maxSrcInBatch, maxTgtInBatch
    if count==1:
        maxSrcInBatch=0
        maxTgtInBatch=0
    maxSrcInBatch=max(maxSrcInBatch, len(new.src))
    maxTgtInBatch=max(maxTgtInBatch, len(new.trg)+2)
    srcElements=count*maxSrcInBatch
    tgtElements=count*maxTgtInBatch
    return max(srcElements, tgtElements)

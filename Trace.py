import time, os, argparse, pdb, pprint,torch

import torch.cuda.nvtx as nvtx

from tqdm import tqdm
from src.util import customDict, greedyDecode, log, getUniqueFolder, rebatch
from src.transformer import *
from src.noamOpt import NoamOpt
from src.dataIterator import BatchIterator
from src.batch import Batch
from src.labelSmoothing import LabelSmoothing
from src.lossCompute import MultiGPULossCompute
from src.dataLoader import generateDataloaders

def runEpoch(dataIter, model, lossCompute, logfile):
    startInit=time.time()
    start=time.time()
    totalTokens=0
    totalLoss=0
    tokens=0
    log("[epoch] \n", logfile)
    for i, batch in enumerate(dataIter):

        nvtx.range_push("Batch: {}".format(i))

        nvtx.range_push("Forward pass")
        
        out=model.forward(batch.src, batch.trg, batch.srcMask, batch.trgMask)

        nvtx.range_push("Loss compute")
        loss=lossCompute(out, batch.trgY, batch.nTokens)
        totalLoss+=loss
        totalTokens+=batch.nTokens
        tokens+=batch.nTokens


        nvtx.range_pop()
        nvtx.range_pop()

        
        if i%50==1:
            elapsed=time.time()-start
            elapsedInit=time.time()-startInit
            elapsed=torch.LongTensor([elapsed])
            if torch.LongTensor([elapsed])==0:
                continue
            message="Epoch: %d Loss: %f TPS %f Batch time elapsed: %d total time elapsed %d" %(i, loss/batch.nTokens, tokens/elapsed, elapsed, elapsedInit)
            log(message+"\n", logfile)
            print(message)
            tokens=0
            start=time.time()
    return totalLoss/totalTokens

global maxSrcInBatch, maxTgtInBatch
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



def train(**kwargs):
    globals().update(kwargs)
    print('loading data...')
    SRC, TGT, train, val, test=generateDataloaders(**dataArgs)
    padIdx=TGT.vocab.stoi["<blank>"]
    model=createModel(len(SRC.vocab), len(TGT.vocab),modelType,  N=6, dModel=512, mode=trainMode, bias=True, **manmpArgs)

    model.cuda()
    criterion=LabelSmoothing(size=len(TGT.vocab), paddingIdx=padIdx, smoothing=0.1)
    criterion.cuda()
    trainIterator=BatchIterator(train, batch_size=batchSize, device=torch.device(devices[0]), repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batchSizeFn, train=True)
    validIterator=BatchIterator(val, batch_size=batchSize, device=torch.device(devices[0]), repeat=False, sort_key=lambda x:(len(x.src), len(x.trg)), batch_size_fn=batchSizeFn, train=False)
    modelParallel=nn.DataParallel(model, device_ids=devices)
    modelOptimizer=NoamOpt(model.srcEmbed[0].embeddingDim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98),eps=1e-9))
    folder=getUniqueFolder('./models/', 'model')
    if not (os.path.exists(folder)):
        os.mkdir(folder)
    logfile=os.path.join(folder, 'logfile')
    print("Training model")
    validLosses=[]
    for epoch in tqdm(range(epochs)):

        nvtx.range_push("EPOCH: {}".format(epoch))

        modelParallel.train()
        log("Train epoch: " +str(epoch) + "\n", logfile)
        runEpoch((rebatch(padIdx, b) for b in trainIterator), modelParallel, MultiGPULossCompute(model.generator, criterion, devices=devices, opt=modelOptimizer), logfile)


        nvtx.range_push("VAL EPOCH: {}".format(epoch))


        modelParallel.eval()
        log("Validation Epoch: "+str(epoch) + "\n", logfile)
        loss=runEpoch((rebatch(padIdx, b) for b in validIterator), modelParallel, MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None), logfile)
        validLosses.append(loss)

        nvtx.range_pop()

        checkpoint={
            'stateDict': model.state_dict(),
            'setting':kwargs,
            'validationLoss':loss
        }
        if saveMode =='all':
            modelName=saveMode + '_loss_{loss:3.3f}.chkpt'.format(loss)
            torch.save(checkpoint, os.path.join(folder, modelName))
        elif saveMode=='best':
            modelName=saveMode + '.chkpt'
            if loss <= max(validLosses):
                torch.save(checkpoint, os.path.join(folder, modelName))
                print('*'*8, "CHECKPOINT UPDATED", '*'*8)
        print(loss)

        nvtx.range_pop()


    for i, batch in enumerate(validIterator):
        src=batch.src.transpose(0,1)[:1]
        srcMask=(src!=SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out=greedyDecode(model, src, srcMask, maxlen=60, startSymbol=TGT.vocab.stoi["<s>"])
        print("Translation: ", end="\t")
        for i in range(0, out.size(0)):
            for j in range(0, out.size(1)):
                sym=TGT.vocab.itos[out[i,j]]
                if sym=="</s>":
                    break
                print(sym, end=" ")
            print()
        print("Target: ", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym=TGT.vocab.itos[batch.trg.data[i,0]]
            if sym=="</s>":
                break
            print(sym, end=" ")
        print()
        break


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.set_defaults(method=train)
    deviceList=[x for x in range(torch.cuda.device_count())]
    parser.add_argument('-devices', type=int, default=deviceList, nargs='+', help="A list of GPUs to use")
    parser.add_argument('-datapath', type=str, default="data/", help="path where data is kept")
    parser.add_argument('-batchSize', type=int, default=10000, help="batch size")
    parser.add_argument('-epochs', type=int, default=5, help="number of training epochs")
    parser.add_argument('-saveMode', type=str, default='best', help='save state dicts of model. [all for all epochs, best for latest epoch]')
    parser.add_argument('-modelType', type=str, default='small', help="small for transformer with 8 heads and 512 dimensions, large for 16heads and 1024 dimensions")
    parser.add_argument('-sourceLang', type=str, default='en', help="source language")
    parser.add_argument('-targetLang', type=str, default='fr', help='target language')
    parser.add_argument('-trainMode', type=str, default='none', help='mode')
    parser.add_argument('-activationBits', type=int, default=8, help='activation bits if in case of quantization training')
    parser.add_argument('-weightBits', type=int, default=16, help='weight bits if in case of quantization training/inference')
    parser.add_argument('-requantizeOutputs', type=bool, default=False, help='requantize outputs?')
    arguments = parser.parse_args()
    assert arguments.trainMode in ['manmp', 'automp', 'none'], "The trainMode must be one of 'manmp', 'automp', or 'none'"
    assert arguments.saveMode in ['all', 'best'], "the save mode should be either 'all' or 'best'"
    manmpArgsList=['activationBits', 'weightBits', 'requantizeOutputs']
    dataArgsList=['datapath', 'sourceLang', 'targetLang']


    allParams=vars(arguments)
    cleanedParams={}
    cleanedParams['dataArgs']=customDict(dataArgsList, allParams)
    cleanedParams['manmpArgs']=customDict(manmpArgsList, allParams)

    for keys in allParams:
        if keys not in manmpArgsList+dataArgsList:
            cleanedParams[keys]=allParams[keys]

    pprint.pprint(cleanedParams, width=3)
    arguments.method(**cleanedParams)

import sys
sys.path.append('src/')

from transformer import *
import argparse, time, torch, pdb, os, pprint
import torch.quantization
from tqdm import tqdm
from src.util import greedyDecode, log, batchSizeFn, customDict, log, getUniqueFolder, rebatch, modelSize
from dataIterator import BatchIterator
from dataLoader import generateDataloaders


def test(**params):
    globals().update(params)
    SRC, TGT, train, val, test=generateDataloaders(**dataArgs)
    modeMap={
        'half':'automp',
        'full':'none',
        'custom':'manmp'
    }
    testIter=BatchIterator(val, batch_size=batchSize, device=torch.device(devices[0]), repeat=False, sort_key=lambda x:(len(x.src), len(x.trg)), batch_size_fn=batchSizeFn, train=False)
    print('loading model...')
    model=createModel(len(SRC.vocab), len(TGT.vocab),modelType='small',  N=6, dModel=512, mode=modeMap[mode], activationBits=activationBits, weightBits=weightBits, requantizeOutputs=False, bias=True)
    model.load_state_dict(torch.load(trainedModel))
    model.cuda(device=torch.device(devices[0]))
    model.eval()
    pdb.set_trace()
    #logfile = 'log' + "_" + str(activationBits) +"  " +str(weightBits)+ " "  + sourceLang + "_" + targetLang + mode + ".txt"
    logfile='temp'
    logfile=open(logfile, 'wb+')
    log("testing model {} ...\n".format(trainedModel).encode(), logfile)
    print(('\t'+'x'*30)*3)
    for idx, batch in tqdm(enumerate(testIter)):
        src_ = batch.src.transpose(0, 1)
        trg_ = batch.trg.transpose(0, 1)
        pdb.set_trace()
        for idx2 in tqdm(range(0, len(src_), 1)):
            then = time.time()
            src = src_[idx2:idx2 + 1]
            trg = trg_[idx2:idx2 + 1]
            srcMask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            maxlen = max(src.size()) + 3
            output = greedyDecode(model, src, srcMask, maxlen=maxlen, startSymbol=TGT.vocab.stoi["<s>"])

            for i in range(0, output.size(0)):
                for j in range(0, output.size(1)):
                    sym = TGT.vocab.itos[output[i, j]]
                    if sym == "</s>":
                        break
                    if sym == "<s>":
                        continue
                    logfile.write(sym.encode('utf8', 'replace'))
                    logfile.write(b" ")

            logfile.write(b"\t")
            for i in range(trg.size(0)):
                for j in range(trg.size(1)):
                    sym = TGT.vocab.itos[trg[i, j]]
                    if sym == "<unk>":
                        sym = "<nid>"
                    if sym == "</s>":
                        break
                    if sym == "<s>":
                        continue
                    logfile.write(sym.encode('utf8', 'replace'))
                    logfile.write(b" ")
            iterTime = time.time() - then
            then = 0
            logfile.write(b"\t" + str(iterTime).encode())
            logfile.write(b"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.set_defaults(method=test)
    parser.add_argument('-trainedModel', type=str, help="path to trained model")
    parser.add_argument('-datapath', type=str, help="path to test data (test file should be named 'test.tsv' inside ideally)", default='data/')
    parser.add_argument('-sourceLang', type=str, default='en', help="source langauge for translation")
    parser.add_argument('-targetLang', type=str, default='fr', help='target language for translation')
    parser.add_argument('-logfile', type=str, help='name of the logfile', default='logfile')
    parser.add_argument('-batchSize', type=int, help='batch size', default=2000)
    parser.add_argument('-maxlen', type=int, help='maxlen for each sentence', default=100)
    parser.add_argument('-devices', type=int, nargs='+', default=[0], help="A list of GPUs to use")
    parser.add_argument('-bitwidth', type=int, default=16, help='bitwidth for quantizing weights')
    parser.add_argument('-mode', type=str, default='full', help='inference mode')
    parser.add_argument('-activationBits', type=int, default=16, help='activation bits')
    parser.add_argument('-weightBits', type=int, default=16, help='weightBits')
    parseArgs=parser.parse_args()
    dataArgsList = ['datapath', 'sourceLang', 'targetLang']
    assert parseArgs.mode in ['custom', 'full', 'half']
    allParams = vars(parseArgs)
    cleanedParams = {}
    cleanedParams['dataArgs'] = customDict(dataArgsList, allParams)

    for keys in allParams:
        if keys not in dataArgsList:
            cleanedParams[keys] = allParams[keys]

    pprint.pprint(cleanedParams, width=3)
    parseArgs.method(**cleanedParams)

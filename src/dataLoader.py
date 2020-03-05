from torchtext import data
import spacy
from dataField import DataField
spacyEn=spacy.load('en')
spacyIt=spacy.load('it')
spacyFr=spacy.load('fr')
def tokenizeEn(text):
    #spacyEn = spacy.load('en')
    return [tok.text for tok in spacyEn.tokenizer(text)]

def tokenizeIt(text):
    #spacyIt = spacy.load('it')
    return [tok.text for tok in spacyIt.tokenizer(text)]

def tokenizeFr(text):
    #spacyFr = spacy.load('fr')
    return [tok.text for tok in spacyFr.tokenizer(text)]

currPrint=0
def myFilter(x, maxlen=100):
    return 'src' in vars(x) and 'trg' in vars(x) and len(vars(x)['src'])<=maxlen and len(vars(x)['trg'])<=maxlen

def generateDataloaders(datapath='data/', sourceLang='en', targetLang='fr'):
    START='<s>'
    END='</s>'
    BLANK='<blank>'
    tokenizeLang={
        'it':tokenizeIt,
        'fr':tokenizeFr,
        'en':tokenizeEn
    }

    SRC=DataField(tokenize=tokenizeLang[sourceLang], pad_token=BLANK)
    TGT=DataField(tokenize=tokenizeLang[targetLang], init_token=START, eos_token=END, pad_token=BLANK)
    train, val, test=data.TabularDataset.splits(path=datapath, train='train.tsv', test='test.tsv', validation='valid.tsv', fields=[('src', SRC), ('trg', TGT)], format='tsv', filter_pred=myFilter)
    MINFREQ=2
    SRC.build_vocab(train.src, min_freq=MINFREQ)
    TGT.build_vocab(train.trg, min_freq=MINFREQ)
    return (SRC, TGT, train, val, test)

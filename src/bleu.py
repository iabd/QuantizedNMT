import nltk, argparse
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize as tokenize

def bleu(translation, **kwargs):
    cc=SmoothingFunction()
    with open(translation, 'r') as f:
        translations=f.readlines()
    scores=[]
    for line in translations:
        try:
            trans=line.split('\t')
            reference, hypothesis=trans[0], trans[1]
        except Exception as e:
            print(e)
            continue
        score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=cc.method4)
        scores.append(score)
    
    print( sum(scores)/len(scores))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.set_defaults(method=bleu)
    parser.add_argument('-translation', type=str, help="location of the translated file")
    parseArgs=parser.parse_args()
    parseArgs.method(**vars(parseArgs))

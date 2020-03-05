import torchtext, pdb
from torch.utils.data import Dataset
from collections import Counter, OrderedDict
from itertools import chain
from pytorchVocab import Vocab



class DataField(torchtext.data.Field):
    def __init__(self, tokenize, pad_token="<pad>", init_token=None, eos_token=None):
        super().__init__()
        self.tokenize=tokenize
        self.pad_token=pad_token
        self.init_token=init_token
        self.eos_token=eos_token
        self.vocab_cls=Vocab


    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))

        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


from torchtext import data
from batch import Batch
class BatchIterator(data.Iterator):
    def createBatches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size*100):
                    pBatch=data.batch(sorted(p, key=self.sort_key),self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(pBatch)):
                        yield b

            self.batches=pool(self.data(), self.random_shuffler)

        else:
            self.batches=[]
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

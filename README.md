# Quantized Transformer for Machine Translation

#### quantization scheme: https://arxiv.org/pdf/1910.06188.pdf

### Normal Training
    
    python3 train.py -batchSize 3200 -datapath ./data/ -devices 0 1 2 3 -epochs 10 -modelType large -sourceLang it -targetLang en

### Quantization Aware Training
    
    python3 train.py -batchSize 3200 -datapath ./data/ -devices 0 1 2 3 -epochs 10 -modelType large -sourceLang it -targetLang en -trainMode manmp -activationBits 8 -weightBits 16 -requantizeOutputs True

### Translation

    python3 test.py -trainedModel ./checkpoints/model1/best.chkpt 
            

#### Note
File `Trace.py` is identical to `train.py` but with NVTX traces. The file is meant to be run to trace the GPU time of each code segment.

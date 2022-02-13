from jax import random, ops
import jax.numpy as np
import matplotlib.pyplot as plt

"""
Sequence to sequence dataset. 
Input should be repeated in output.
"""

class CopyDataSet:

    def __init__(self, batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue, bias=False):
        self.batch_size = batch_size
        self.maxSeqLength = maxSeqLength
        self.minSeqLength = minSeqLength
        self.bits = bits
        self.padding = padding
        self.lowValue = lowValue
        self.highValue = highValue 
        self.bias = bias

        self.input = self.bits+2 # start and stop
        self.output = self.bits

    def calcSample(self, key):
        seqLength = random.randint(key, (1,), minval=self.minSeqLength, maxval=self.maxSeqLength+1)[0]
        pattern = random.choice(key, np.array([self.lowValue, self.highValue]), shape=(seqLength, self.output))

        x = np.ones(((self.maxSeqLength*2)+2,self.input)) * self.padding
        y = np.ones(((self.maxSeqLength*2)+2,self.output)) * self.padding

        startSeq = np.ones((self.input)) * self.padding
        startSeq = ops.index_update(startSeq,0,1.0)

        endSeq = np.ones((self.input)) * self.padding
        endSeq = ops.index_update(endSeq,1,1.0)


        x = ops.index_update(x, ops.index[0], startSeq)
        x = ops.index_update(x, ops.index[1:(1+seqLength),2:], pattern)
        x = ops.index_update(x, ops.index[(1+seqLength)], endSeq)

        y = ops.index_update(y, ops.index[seqLength+2:(2*seqLength)+2,:], pattern)

        return x, y

    def getSample(self, key):
        
        inputs = []
        outputs = []

        for i, k in zip(range(self.batch_size), random.split(key, self.batch_size)):

            x, y = self.calcSample(k)

            if (self.bias):
                x = np.append(x, np.ones((x.shape[0],1)), axis=1)

            inputs.append(x)
            outputs.append(y)

        return np.array(inputs), np.array(outputs)

if __name__ == "__main__":

    key = random.PRNGKey(1)
    
    maxSeqLength = 10
    minSeqLength = 7
    bits = 1
    padding = 0
    lowValue = 0
    highValue = 1
    batch_size= 1

    data = CopyDataSet(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)
    x, y = data.getSample(key)

    x = x[0]
    y = y[0]

    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.subplots_adjust(top=0.85,bottom=0.15,left=0.05,right=0.95)

    cmap = plt.get_cmap('jet')
    t=ax1.matshow(x.T,aspect='auto',cmap=cmap)
    ax1.set_ylabel("Input")
    p=ax2.matshow(y.T,aspect='auto',cmap=cmap)
    ax2.set_ylabel("Traget")

    fig.suptitle('Copy Task')
    fig.colorbar(t,ax=(ax1,ax2),orientation="vertical",fraction=0.1)

    plt.show()



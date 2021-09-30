import numpy as np
from dynarray import DynamicArray


class pos():
    """
    Implemantation of POS method as proposed by:
    
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic principles of remote-PPG. 
    IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. [7565547]. 
    https://doi.org/10.1109/TBME.2016.2609282
    
    """
    
    P : np.ndarray = np.array([[0, 1, -1], [-2, 1, 1]]).T     # projection matrix
    
    def __init__(self):
        self.n : int = 0     # counter
        self.l : int = 60 #48    # segment size
        self.m : int = self.n-self.l+1
        self.inputSignal : DynamicArray = DynamicArray(array_or_shape=(None,3), capacity=200)
        self.signal : DynamicArray = DynamicArray(np.zeros(self.l-1), capacity=200)
    
    def append(self, x):
        self.n += 1
        self.m += 1
        self.inputSignal.append(x)
        
        if self.m > 0:
            c = self.inputSignal[-self.l:,:] / np.mean(self.inputSignal[-self.l:, :], axis=0)
            s = c @ self.P
            h = s[:, 0] + s[:, 1] * (np.std(s[:, 0]) / (np.std(s[:, 1]) + 1e-5))
            self.signal.append(0)
            self.signal[-self.l:] += h - np.mean(h)
            
    @property
    def shape(self):
        return self.signal.shape
                                     
           
        
    
if __name__ == '__main__':
    input = np.sin(2*np.pi/100 * np.array([np.arange(48), np.arange(48), np.arange(48)]).T)
    print(input)
    print(input.shape)
    signal = pos()
    for i in range(48):
        signal.append(input[i])
        
    print(signal.shape)
    print('n', signal.n, 'm', signal.m)
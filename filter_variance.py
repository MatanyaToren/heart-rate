import numpy as np

class VarianceFilter:
    """
    This class takes samples consecutive samples and leaves out outlying samples
    returns the last inlier sample and if its the newst sample
    """
    def __init__(self, n_history=100, n_minimal=20, num_sigmas=2):
        self.past_samples = []
        self.n_history = n_history
        self.n_minimal = n_minimal
        self.n = 0
        self.num_sigmas = num_sigmas
        self.valid = True
        
    def update(self, sample):
        if self.n < self.n_minimal:
            self.past_samples.append(sample)
            self.n += 1
            self.valid = True
            
        elif self.in_bounderies(sample):
            self.past_samples.append(sample)
            self.n += 1
            self.valid = True
            
        else:    
            self.valid = False
            
        return self.past_samples[-1], self.valid
        
        
        
    def in_bounderies(self, sample):
        self.mean = np.mean(self.past_samples[-self.n_history:])
        self.std = np.std(self.past_samples[-self.n_history:])
        
        return (sample <= self.mean + self.num_sigmas*self.std) and (sample >= self.mean - self.num_sigmas*self.std)
    
    
if __name__ == '__main__':
    filter = VarianceFilter()
    samples = np.random.randn(1000) + (np.random.randn(1000) > 2) * 4
    # print([filter.update(sample) for sample in samples])
    filtered_samples = [filter.update(sample) for sample in samples]
    valids = np.array([valid for (_, valid) in filtered_samples])
    sample_num = np.arange(1000)
    
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.scatter(sample_num[valids], samples[valids], c='g')
    ax.scatter(sample_num[np.logical_not(valids)], samples[np.logical_not(valids)], c='r')
    
    plt.show()
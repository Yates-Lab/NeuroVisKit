from scipy.ndimage import gaussian_filter
import numpy as np
import torch

def get_gaus_kernel(shape):
    kernel = np.ones(shape)
    for i in range(len(shape)):
        assert shape[i] % 2 == 1
        k = np.zeros(shape[i])
        k[shape[i]//2] = 1
        k = gaussian_filter(k, shape[i]/5).reshape(-1, *([1]*(len(shape)-1-i)))
        kernel = kernel * k
    return kernel
def split_batched_op(input, op, groups=2, device="cpu"):
    b = input.shape[0]
    gsize = int(np.ceil(b / groups))
    i = 0
    while i < b:
        iend = min(i+gsize, b)
        input[i:iend] = op(input[i:iend].to(device)).cpu()
        i = iend
    return input

class IndexableDict(dict):
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return IndexableDict({k: v[key] for k, v in self.items()})
        return super().__getitem__(key)
def hann_window(shape):
    window = 1
    for i in range(len(shape)):
        window = window * torch.hann_window(shape[i]).reshape(-1, *([1]*(len(shape)-1-i)))
    return window
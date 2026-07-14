
import cupy as cp

def hanning(x,width):
    return 0.5 * (1 - cp.cos(x * 2 * cp.pi / (width - 1)))

def hamming(x,width):
    return 0.54 - 0.46 * cp.cos(x * 2 * cp.pi / (width - 1))

def blackman(x,width):
    return 0.42 - 0.5 * cp.cos(x * 2 * cp.pi / (width - 1)) + 0.08 * cp.cos(x * 4 * cp.pi / (width - 1))

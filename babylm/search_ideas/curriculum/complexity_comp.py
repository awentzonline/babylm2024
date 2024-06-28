import gzip

import numpy as np


def get_compressed_size(tokens, offset=None):
    """
    The offset is to ensure all numbers have the same string length. I think it's quicker than zero-padding a string.
    """
    if offset is None:
        log_max_val = np.log10(tokens.max())
        offset = (10 ** np.floor(log_max_val + 1)).astype(np.int32)
    tokens = tokens + offset
    tokens = bytes(np.array2string(tokens, max_line_width=np.inf), 'utf-8')
    compressed = gzip.compress(tokens)
    return len(compressed)


if __name__ == '__main__':
    size = 1000
    x = np.arange(size)
    print(get_compressed_size(x, offset=10000))
    x = np.arange(1000, 1000 + size)
    print(get_compressed_size(x))
    x = np.random.randint(1000, 1000 + size, size)
    print(get_compressed_size(x))
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    """
    Parameters
    ----------
    input_data : (dataset size, channel, height, width) of four-dimensional array
    filter_h : Height of the filter
    filter_w : Width of the filter
    stride : Stride
    padding : Padding
    Returns
    -------
    col : 2D matrix
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(
        input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], "constant"
    )
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
    """
    Parameters
    ----------
    col :
    input_shape : Original input shape, (N, C, H, W)
    filter_h :
    filter_w
    stride
    padding
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding : H + padding, padding : W + padding]


def im2col_indices(input_data, filter_height, filter_width, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    input_data_padded = np.pad(
        input_data, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant"
    )

    # Compute the output dimensions
    batch_size, channels, height, width = input_data_padded.shape
    out_height = int((height - filter_height) / stride) + 1
    out_width = int((width - filter_width) / stride) + 1

    # Get the indices for the input data and the filters
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    # Get the corresponding columns from the input data
    cols = input_data_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)

    return cols

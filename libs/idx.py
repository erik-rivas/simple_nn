from struct import unpack

import numpy as np


def display(img, width=28, threshold=200):
    render = ""
    for i in range(len(img)):
        if i % width == 0:
            render += "\n"
        if img[i] > threshold:
            render += "@"
        else:
            render += "."
    return render


def get_data_type(data_type):
    """
    Returns the description and data type from a idx encoded integer
    """

    dt_dict = {
        0x08: ("0x08: unsigned byte", "B"),
        0x09: ("0x09: signed byte", "b"),
        0x0B: ("0x0B: short (2 bytes)", "h"),
        0x0C: ("0x0C: int (4 bytes)", "H"),
        0x0D: ("0x0D: float (4 bytes)", "f"),
        0x0E: ("0x0E: double (8 bytes) ", "d"),
    }

    desc, dtype = dt_dict[data_type]

    return desc, dtype


def read_idx(path: str, items_to_read: int = None):
    arr = []

    with open(path, "rb") as file:
        content = file.read(4)
        magic = unpack(">i", content)[0]
        # Data type
        data_type = (magic >> 8) & 0xFF
        desc, dtype = get_data_type(data_type)
        print(desc)
        n_dim = magic & 0xFF

        dims = []
        for _ in range(n_dim):
            content = file.read(4)
            dim = unpack(">i", content)[0]
            dims.append(dim)

        print(f"{magic:X}")
        print(f"data_type: {data_type:X}")
        print(f"n_dims: {n_dim}")
        # print(f"dims: {dims}")

        shape = tuple(dims)

        if n_dim == 3:
            n_imgs, rows, cols = dims

            print(f"n_imgs: {n_imgs}, rows: {rows}, cols: {cols}")

            items_to_read = items_to_read or n_imgs
            for i in range(items_to_read):
                n_flat = rows * cols
                img = np.fromfile(file, dtype=dtype, count=n_flat)
                arr.append(img)

        if n_dim == 1:
            n_flat = shape[0]
            items_to_read = items_to_read or n_flat
            arr = np.fromfile(file, dtype=dtype, count=items_to_read)

    return np.array(arr), shape


if __name__ == "__main__":
    imgs, shape = read_idx("data/mnist/train-images-idx3-ubyte", items_to_read=10)
    img = imgs[0]
    no_imgs, rows, cols = shape
    # render = display(img=img, width=cols, threshold=200)
    # print(render)

    labels, shape = read_idx("data/mnist/train-labels-idx1-ubyte", items_to_read=10)

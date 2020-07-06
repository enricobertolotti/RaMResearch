import numpy as np


def multiply_w_offset(a, b, b_center):
    b_dim = b.shape
    a_dim = [[int(dim/2), dim-int(dim/2)] for dim in b_dim]
    for i in range(len(b_center)):
        a_dim[i] = [b_center[i]-a_dim[i][0], b_center[i]+a_dim[i][1]]
    cropped_a = a[a_dim[0][0]:a_dim[0][1], a_dim[1][0]:a_dim[1][1], a_dim[2][0]:a_dim[2][1]]

    # multiplied = np.multiply(cropped_a.astype(np.int16, casting="safe"), b.astype(np.int16, casting="safe"))
    multiplied = np.multiply(cropped_a, b)
    # normalized = np.divide(multiplied, 255).astype(np.int8)
    print("Multiplied Min:\t" + str(np.min(multiplied)) + "\t Max:" + str(np.max(multiplied)))
    return multiplied


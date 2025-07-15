def naive_matrix_vector_product(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i] += x[i, j] * y[j]
    return z
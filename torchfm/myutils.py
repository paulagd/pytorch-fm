import numpy as np
import torch
import scipy.sparse as sp


def pairwise(array):
    # [x * y for i, x in enumerate(array) for j, y in enumerate(array) if j > i]
    result = []
    for i, x in enumerate(array):
        for j, y in enumerate(array):
            if j > i:
                result.append(x * y)
    return np.sum(result)


def compute_pairwise(user, item, embeddings, context_list, gcn_flag=False):
    array = [user, item]
    if gcn_flag:
        for c in context_list:
            array.append(embeddings[c])
    else:
        for c in context_list:
            array.append(embeddings(c))

    return pairwise(array)


def multiply_context_embedding(ui, embeddings, c):
    result = ui * embeddings(c[0])
    for c_i in c[1:]:
        result = result * embeddings(c_i)

    return result


def multiply_context_gcn(ui, embeddings, c):

    result = ui * embeddings[c[0]]
    for c_i in c[1:]:
        result = result * embeddings[c_i]

    return result


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(H, norm):
    """
        Row-normalize sparse matrix the way S-GCN does  (message passing)
        Message passing ( D * H ), WHERE H is the representation of the last layer (I for the first layer).
    """
    # D is the degree of each row
    D = np.array(H.sum(1))

    # norm_mx = np.diag(D in diagonal) * H

    if norm == 'sym':
        # S = D^-1/2 * (H) * D^-1/2
        print("Symmetric normalization")
        D_inv_half = np.power(D, -0.5).flatten()
        D_inv_half[np.isinf(D_inv_half)] = 0.
        D_mat = sp.diags(D_inv_half)
        aux = D_mat.dot(H)
        norm_mx = aux.dot(D_mat)
    elif norm == 'left':
        print("Left normalization")
        # D^-1 * (H)
        r_inv = np.power(D, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        D_mat = sp.diags(r_inv)
        norm_mx = D_mat.dot(H)
    else:
        print("Other normalization")
        D_mat = sp.diags(D.flatten())
        # norm_mx = D_mat * H
        norm_mx = D_mat.dot(H)

    return norm_mx


def list_to_device(data, device, non_valid_indexes=[]):
    for idx, c in enumerate(data):
        data[idx] = np.delete(c, non_valid_indexes).to(device)
    return data


def build_DAD_mx(mx, normalization):
    # -----JUST TEST EXAMPLE ------
    # t = [[0, 1, 2], [0, 1, 3]]
    # a = np.zeros((4, 4))
    #
    # for x in t:
    # 	a[x[0], x[1]] = 1.0
    # 	for idx in range(len(x[2:])):
    # 		a[x[0], x[2 + idx]] = 1.0
    # 		a[x[1], x[2 + idx]] = 1.0
    # adj_train.todense()
    # ------------------------------
    # MAKE MATRIX SPARSE
    adj_train = sp.coo_matrix(mx, dtype=np.float32)
    # MAKE MATRIX SYMMETRIC U-I  --> I-U
    adj_train = adj_train + adj_train.T.multiply(adj_train.T > adj_train) - adj_train.multiply(adj_train.T > adj_train)
    assert (True in np.unique(adj_train.todense().diagonal()) == 1) == False, "DIAGONAL HAS NON ZERO VALUES"
    # SUM THE SELF.CONNECTIONS AND NORMALIZE IT (SYM OR LEFT NORM)
    adj_train = normalize(adj_train + sp.eye(adj_train.shape[0]), normalization)
    adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)

    return adj_train
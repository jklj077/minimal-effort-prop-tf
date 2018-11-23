'''
Define meProp operations

MatMulMeProp:
    usage: y = MatMulMeProp(w, x, k)
    when backprop, use dense matmul
    but dy in each matmul of the two is declared to be sparse

MatMulMePropUnifiedCompacted:
    usage: y = MatMulMePropUnifiedCompacted(w, x, k)
    when backprop, dy is compacted into a smaller matrix (columns that are zeros are removed)
    the version implemented in PyTorch
'''

import tensorflow as tf
from tensorflow.python.framework import function


def get_top_k(values, k):
    def convert(indices):
        # (num_row) = [0, 1,..., num_row]
        new_indices = tf.range(0, tf.shape(indices)[0])
        # (num_row, 1) = [[0], [1],..., [num_row]]
        new_indices = tf.reshape(new_indices, [-1, 1])
        # (num_row, k) = [[0, 0,..., 0], [1, 1,..., 1],..., [num_row, num_row,..., num_row]]
        new_indices = tf.tile(new_indices, [1, tf.shape(indices)[1]])
        # (num_elem = num_row*num_elem_per_row, 2) = [[0, x], [0, x],..., [0, x],..., [...]]
        new_indices = tf.stack(
            [tf.reshape(new_indices, [-1]),
             tf.reshape(indices, [-1])], 1)
        return new_indices

    _, top_ind = tf.nn.top_k(tf.abs(values), k)
    top_ind = convert(top_ind)
    top_val = tf.gather_nd(values, top_ind)
    sp = tf.SparseTensor(
        tf.cast(top_ind, tf.int64), top_val,
        tf.cast(tf.shape(values), tf.int64))
    #sp = tf.sparse_reorder(sp)
    return sp


def get_top_k_unified(values, k):
    def convert(nrow, indices):
        # indices is col index
        # (num_row) = [0, 1,..., num_row]
        new_indices = tf.range(0, nrow)
        # (num_row, 1) = [[0], [1],..., [num_row]]
        new_indices = tf.reshape(new_indices, [-1, 1])
        # (num_row, k) = [[0, 0,..., 0], [1, 1,..., 1],..., [num_row, num_row,..., num_row]]
        new_indices = tf.tile(new_indices, [1, tf.shape(indices)[0]])
        # (num_elem = num_row*num_elem_per_row, 2) = [[0, x], [0, x],..., [0, x],..., [...]]
        new_indices = tf.stack(
            [tf.reshape(new_indices, [-1]),
             tf.tile(indices, [nrow])], 1)
        return new_indices

    nrow = tf.shape(values)[0]
    sumcol = tf.reduce_sum(tf.abs(values), 0)
    _, top_ind = tf.nn.top_k(sumcol, k)
    top_ind = convert(nrow, top_ind)
    top_val = tf.gather_nd(values, top_ind)
    sp = tf.SparseTensor(
        tf.cast(top_ind, tf.int64), top_val,
        tf.cast(tf.shape(values), tf.int64))
    #sp = tf.sparse_reorder(sp)
    return sp


def gather_col(params, indices):
    shape = tf.shape(params)
    p_flat = tf.reshape(params, [-1])
    # (num_row, 1) = [[row_0_ind], [row_1_ind],..., [row_n_ind]]

    i_flat = tf.reshape(tf.range(0, shape[0]) * shape[1], [-1, 1])
    # (num_row*nind) = [[row_0_ind+ind_0, row_0_ind+ind_1, ...],...]

    i_flat = tf.reshape(i_flat + indices, [-1])
    return tf.reshape(tf.gather(p_flat, i_flat), [shape[0], -1])


def scatter_col(params, indices, shape):
    p_flat = tf.reshape(params, [-1])
    i_flat = tf.reshape(tf.range(0, shape[0]) * shape[1], [-1, 1])
    i_flat = tf.reshape(i_flat + indices, [-1])
    return tf.reshape(
        tf.scatter_nd(
            tf.reshape(i_flat, [-1, 1]), p_flat, [shape[0] * shape[1]]),
        [shape[0], -1])


def MatMulMePropGrad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    grad = tf.sparse_tensor_to_dense(get_top_k(grad, op.inputs[2]), validate_indices=False)
    grad_a = tf.matmul(grad, b, transpose_b=True, a_is_sparse=True)
    grad_b = tf.matmul(a, grad, transpose_a=True, b_is_sparse=True)
    return grad_a, grad_b, None


@function.Defun(
    tf.float32, tf.float32, tf.int32, python_grad_func=MatMulMePropGrad)
def MatMulMeProp(a, b, k):
    return tf.matmul(a, b)

def MatMulMePropUnifiedCompactedGrad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    k = op.inputs[2]
    _, indices = tf.nn.top_k(tf.reduce_sum(tf.abs(grad), axis=0), k)
    d_grad = gather_col(grad, indices)
    b_t = tf.transpose(b)
    d_b_t = tf.gather(b_t, indices)
    da = tf.matmul(d_grad, d_b_t)
    d_db = tf.matmul(a, d_grad, transpose_a=True)
    db = scatter_col(d_db, indices, tf.shape(b))
    return da, db, None

@function.Defun(
    tf.float32,
    tf.float32,
    tf.int32,
    python_grad_func=MatMulMePropUnifiedCompactedGrad)
def MatMulMePropUnifiedCompacted(a, b, k):
    return tf.matmul(a, b)


#pylint: disable=not-context-manager

import tensorflow as tf
from tensorflow.python.framework import function


def get_top_k(values, k):
    def convert(indices):
        with tf.name_scope('dense_indices_to_sparse_indices'):
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

    with tf.name_scope('top_k_selection_in_abs'):
        _, top_ind = tf.nn.top_k(tf.abs(values), k, sorted=False)
        top_ind = convert(top_ind)
        top_val = tf.gather_nd(values, top_ind, name='gather_original_value')
        sp = tf.SparseTensor(
            tf.cast(top_ind, tf.int64), top_val,
            tf.cast(tf.shape(values), tf.int64))
        #sp = tf.sparse_reorder(sp)
    return sp


def get_top_k_unified(values, k):
    def convert(nrow, indices):
        with tf.name_scope('dense_indices_to_sparse_indices'):
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

    with tf.name_scope('unified_top_k_selection_in_abs'):
        nrow = tf.shape(values)[0]
        sumcol = tf.reduce_sum(tf.abs(values), 0)
        _, top_ind = tf.nn.top_k(sumcol, k)
        top_ind = convert(nrow, top_ind)
        top_val = tf.gather_nd(values, top_ind, name='gather_original_value')
        sp = tf.SparseTensor(
            tf.cast(top_ind, tf.int64), top_val,
            tf.cast(tf.shape(values), tf.int64))
        sp = tf.sparse_reorder(sp)
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


def MePropGrad(op, grad):
    return tf.sparse_tensor_to_dense(
        get_top_k(grad, op.inputs[1]), validate_indices=False), None


@function.Defun(tf.float32, tf.int32, python_grad_func=MePropGrad)
def MeProp(values, k):  #pylint: disable=unused-argument
    return values


def MePropRecordGrad(op, grad):
    top = get_top_k(grad, op.inputs[1])
    ones = tf.ones_like(top.values, dtype=tf.int32)
    agg_top_inds = tf.sparse_reduce_sum(
        tf.SparseTensor(top.indices, ones, top.dense_shape), 0)
    return tf.sparse_tensor_to_dense(
        top,
        validate_indices=False), None, agg_top_inds, tf.shape(op.inputs[0])[0]


@function.Defun(
    tf.float32,
    tf.int32,
    tf.int32,
    tf.int32,
    python_grad_func=MePropRecordGrad)
def MePropRecord(values, k, record, ref):  #pylint: disable=unused-argument
    return values


def MePropUnifiedGrad(op, grad):
    return tf.sparse_tensor_to_dense(
        get_top_k_unified(grad, op.inputs[1])), None


@function.Defun(tf.float32, tf.int32, python_grad_func=MePropUnifiedGrad)
def MePropUnified(values, k):  #pylint: disable=unused-argument
    return values


def MatMulMePropGrad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    grad = tf.sparse_tensor_to_dense(get_top_k(grad, op.inputs[2]))
    grad_a = tf.matmul(grad, b, transpose_b=True, a_is_sparse=True)
    grad_b = tf.matmul(a, grad, transpose_a=True, b_is_sparse=True)
    return grad_a, grad_b, None


@function.Defun(
    tf.float32, tf.float32, tf.int32, python_grad_func=MatMulMePropGrad)
def MatMulMeProp(a, b, k):  #pylint: disable=unused-argument
    return tf.matmul(a, b)


def MatMulMePropSparseGrad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    grad_k = get_top_k(grad, op.inputs[2])
    #grad = tf.sparse_tensor_to_dense(get_top_k(grad, op.inputs[2]))
    grad_a = tf.sparse_tensor_dense_matmul(grad_k, b, adjoint_b=True)
    #grad_a = tf.matmul(grad, b, transpose_b=True, a_is_sparse=True)
    grad_b = tf.transpose(
        tf.sparse_tensor_dense_matmul(grad_k, a, adjoint_a=True))
    #grad_b = tf.matmul(a, grad, transpose_a=True, b_is_sparse=True)
    return grad_a, grad_b, None


@function.Defun(
    tf.float32, tf.float32, tf.int32, python_grad_func=MatMulMePropSparseGrad)
def MatMulMePropSparse(a, b, k):  #pylint: disable=unused-argument
    return tf.matmul(a, b)


def MatMulMePropSparseOneGrad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    grad_k = get_top_k(grad, op.inputs[2])
    #grad = tf.sparse_tensor_to_dense(get_top_k(grad, op.inputs[2]))
    grad_a = tf.sparse_tensor_dense_matmul(grad_k, b, adjoint_b=True)
    #grad_a = tf.matmul(grad, b, transpose_b=True, a_is_sparse=True)
    grad_b = tf.matmul(
        a,
        tf.sparse_tensor_to_dense(grad_k),
        transpose_a=True,
        b_is_sparse=True)
    return grad_a, grad_b, None


@function.Defun(
    tf.float32,
    tf.float32,
    tf.int32,
    python_grad_func=MatMulMePropSparseOneGrad)
def MatMulMePropSparseOne(a, b, k):  #pylint: disable=unused-argument
    return tf.matmul(a, b)


def MatMulMePropUnifiedGrad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    grad = tf.sparse_tensor_to_dense(get_top_k_unified(grad, op.inputs[2]))
    grad_a = tf.matmul(grad, b, transpose_b=True, a_is_sparse=True)
    grad_b = tf.matmul(a, grad, transpose_a=True, b_is_sparse=True)
    return grad_a, grad_b, None


@function.Defun(
    tf.float32, tf.float32, tf.int32, python_grad_func=MatMulMePropUnifiedGrad)
def MatMulMePropUnified(a, b, k):  #pylint: disable=unused-argument
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

    # db: version 1
    d_db = tf.matmul(a, d_grad, transpose_a=True)
    db = scatter_col(d_db, indices, tf.shape(b))
    return da, db, None
    # db: version 2
    # d_db_t = tf.matmul(d_grad, a, transpose_a=True)
    # db_t = tf.scatter_nd(indices, d_db_t, tf.shape(b_t))
    # return da, tf.transpose(db_t), None


@function.Defun(
    tf.float32,
    tf.float32,
    tf.int32,
    python_grad_func=MatMulMePropUnifiedCompactedGrad)
def MatMulMePropUnifiedCompacted(a, b, k):  #pylint: disable=unused-argument
    return tf.matmul(a, b)

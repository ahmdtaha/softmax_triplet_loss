import tensorflow as tf
import numpy as np

def all_pairs_tf(a, b):
    """
    Return a tensor of all pairs
    a -- [batch_size1, dim]
    b -- [batch_size2, dim]
    """
    dim = tf.shape(a)[1]
    temp_a = tf.expand_dims(a, axis=1) + tf.zeros(tf.shape(tf.expand_dims(b,axis=0)), dtype=b.dtype)
    temp_b = tf.zeros(tf.expand_dims(a, axis=1), dtype=a.dtype) + tf.expand_dims(b,axis=0)
    return tf.concat((tf.reshape(temp_a, [-1,1,dim]), tf.reshape(temp_b, [-1,1,dim])), axis=1)


def all_diffs_tf(a, b):
    """
    Return a tensor of all combinations of a - b

    a -- [batch_size1, dim]
    b -- [batch_size2, dim]

    reference: https://github.com/VisualComputingInstitute/triplet-reid
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def all_diffs(a, b):
    """
    Return a tensor of all combinations of a - b

    a -- [batch_size1, dim]
    b -- [batch_size2, dim]

    reference: https://github.com/VisualComputingInstitute/triplet-reid
    """
    return np.expand_dims(a, axis=1) - np.expand_dims(b, axis=0)

def cdist(diff, metric='squaredeuclidean'):
    """
    Return the distance according to metric

    diff -- [..., dim], difference matrix
    metric  --   "squaredeuclidean": squared euclidean
                 "euclidean": euclidean (without squared)
                 "l1": manhattan distance
    """

    if metric == "squaredeuclidean":
        return np.sum(np.square(diff), axis=-1)
    elif metric == "euclidean":
        return np.sqrt(np.sum(np.square(diff), axis=-1) + 1e-12)
    elif metric == "l1":
        return np.sum(np.abs(diff), axis=-1)
    else:
        raise NotImplementedError

def cdist_tf(diff, metric='squaredeuclidean'):
    """
    Return the distance according to metric

    diff -- [..., dim], difference matrix
    metric  --   "squaredeuclidean": squared euclidean
                 "euclidean": euclidean (without squared)
                 "l1": manhattan distance
    """

    if metric == "squaredeuclidean":
        return tf.reduce_sum(tf.square(diff), axis=-1)
    elif metric == "euclidean":
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-12)
    elif metric == "l1":
        return tf.reduce_sum(tf.abs(diff), axis=-1)
    else:
        raise NotImplementedError
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

def pairwise_distance(feature, squared=False):
  """Computes the pairwise distance matrix with numerical stability.

  output[i, j] = || feature[i, :] - feature[j, :] ||_2

  Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.

  Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
  """
  pairwise_distances_squared = math_ops.add(
      math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
      math_ops.reduce_sum(
          math_ops.square(array_ops.transpose(feature)),
          axis=[0],
          keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                  array_ops.transpose(feature))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
  # Get the mask where the zero distances are at.
  error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

  # Optionally take the sqrt.
  if squared:
    pairwise_distances = pairwise_distances_squared
  else:
    pairwise_distances = math_ops.sqrt(
        pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

  # Undo conditionally adding 1e-16.
  pairwise_distances = math_ops.multiply(
      pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

  num_data = array_ops.shape(feature)[0]
  # Explicitly set diagonals to zero.
  mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
      array_ops.ones([num_data]))
  pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
  return pairwise_distances



def masked_minimum_idx(data, mask, dim=1):
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)

    masked_minimums_idx = math_ops.argmin(
        math_ops.multiply(data - axis_maximums, mask), dim)
    return masked_minimums_idx
def masked_maximum_idx(data, mask, dim=1):
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)

    masked_maximums_idx = math_ops.argmax(
        math_ops.multiply(data - axis_minimums, mask), dim)
    return masked_maximums_idx

def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
  masked_maximums = math_ops.reduce_max(
      math_ops.multiply(data - axis_minimums, mask), dim,
      keepdims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
  masked_minimums = math_ops.reduce_min(
      math_ops.multiply(data - axis_maximums, mask), dim,
      keepdims=True) + axis_maximums
  return masked_minimums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
      The loss encourages the positive distances (between a pair of embeddings with
      the same labels) to be smaller than the minimum negative distance among
      which are at least greater than the positive distance plus the margin constant
      (called semi-hard negative) in the mini-batch. If no such negative exists,
      uses the largest negative distance instead.
      See: https://arxiv.org/abs/1503.03832.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        triplet_loss: tf.float32 scalar.
      """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    triplet_loss = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')


    return triplet_loss
def triplet_semihard_loss_mine(labels, embeddings, margin=1.0, background=0,log_var=None):
  """Computes the triplet loss with semi-hard negative mining.

  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.
  See: https://arxiv.org/abs/1503.03832.

  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.
    background: int, indicate the background event (default to be label 0), mask out the background event if indicated. set to -1 if not masking out the background event.

  Returns:
    triplet_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_distance(embeddings, squared=True)
  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  # Compute the mask.
  pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
  mask = math_ops.logical_and(
      array_ops.tile(adjacency_not, [batch_size, 1]),
      math_ops.greater(
          pdist_matrix_tile, array_ops.reshape(
              array_ops.transpose(pdist_matrix), [-1, 1])))
  mask_final = array_ops.reshape(
      math_ops.greater(
          math_ops.reduce_sum(
              math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
          0.0), [batch_size, batch_size])
  mask_final = array_ops.transpose(mask_final)

  adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  mask = math_ops.cast(mask, dtype=dtypes.float32)

  # negatives_outside: smallest D_an where D_an > D_ap.
  negatives_outside = array_ops.reshape(
      masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = array_ops.transpose(negatives_outside)

  # negatives_inside: largest D_an.
  negatives_inside = array_ops.tile(
      masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])

  negatives_outside_idx = array_ops.reshape(masked_minimum_idx(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside_idx = array_ops.transpose(negatives_outside_idx)
  negatives_inside_idx = array_ops.tile(masked_maximum_idx(pdist_matrix, adjacency_not)[:, tf.newaxis], [1, batch_size])

  semi_hard_negatives = array_ops.where(
      mask_final, negatives_outside, negatives_inside)
  semi_hard_negatives_idx = array_ops.where(
      mask_final, negatives_outside_idx, negatives_inside_idx)

  loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

  # mask for foreground events
  mask_foreground = array_ops.tile(
          math_ops.not_equal(
              labels, background), [1, batch_size])
  mask_positives = math_ops.multiply(
          mask_positives, math_ops.cast(
              mask_foreground, dtype=dtypes.float32))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = math_ops.reduce_sum(mask_positives)

  prefix = 1
  suffix = 0
  verbose = False
  if log_var is not None:

      # positive_idx = tf.argmax(furthest_dist, axis=1)
      # negative_idx = tf.argmin(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)

      if verbose:
          log_var = tf.Print(log_var, [log_var], 'log_var ', summarize=100)

      anchor_log_var = array_ops.tile(log_var, [1, batch_size])
      positive_log_var = array_ops.tile(log_var, [batch_size, 1])

      if verbose:
          anchor_log_var = tf.Print(anchor_log_var, [anchor_log_var], 'anchor_log_var ', summarize=100)
          positive_log_var = tf.Print(positive_log_var, [positive_log_var], 'positive_log_var ', summarize=100)

      s_anchor = tf.boolean_mask(anchor_log_var, tf.cast(mask_positives, tf.bool))
      s_positive = tf.boolean_mask(positive_log_var, tf.cast(mask_positives, tf.bool))
      negative_idx = tf.boolean_mask(semi_hard_negatives_idx, tf.cast(mask_positives, tf.bool))

      if verbose:
          negative_idx = tf.Print(negative_idx, [negative_idx], 'negative_idx ', summarize=100)

      s_negative = tf.gather_nd(log_var, negative_idx[:, tf.newaxis])

      if verbose:
          s_anchor = tf.Print(s_anchor, [s_anchor], 's_anchor ', summarize=100)
          s_positive = tf.Print(s_positive, [s_positive], 's_positive ', summarize=100)
          s_negative = tf.Print(s_negative, [s_negative], 's_negative ', summarize=100)
      print('Using Log Var')

      prefix = 0.5 * (tf.exp(-s_anchor) + tf.exp(-s_positive) + tf.exp(-s_negative))
      suffix = 0.5 * (s_anchor + s_positive + s_negative)

      # log_var = tf.nn.relu(log_var)

      # s_anchor = log_var
      # s_positive = tf.gather_nd(log_var, positive_idx[:, tf.newaxis])
      # s_negative = tf.gather_nd(log_var, negative_idx[:, tf.newaxis])

      # furthest_positive = furthest_positive  # * (tf.exp(-s_anchor) + tf.exp(-s_positive) + tf.exp(-s_negative)) #+ 0.5 * (s_anchor + s_positive + s_negative)
      # closest_negative = closest_negative  # * (tf.exp(-s_anchor) + tf.exp(-s_positive) + tf.exp(-s_negative))

      # prefix = 0.5 * (tf.exp(-s_anchor) + tf.exp(-s_positive) + tf.exp(-s_negative))
      # suffix = 0.5 * (s_anchor + s_positive + s_negative)

      print('prefix add log ****')
  else:
      print('Not ** Using Log Var')

  #triplet_loss = prefix * tf.nn.softplus(tf.boolean_mask(loss_mat, tf.cast(mask_positives, tf.bool))) + suffix
  triplet_loss =  prefix * math_ops.maximum(tf.boolean_mask(loss_mat, mask_positives), 0.0) + suffix
  # triplet_loss = math_ops.truediv(
  #     math_ops.reduce_sum(
  #         math_ops.maximum(
  #             math_ops.multiply(loss_mat, mask_positives), 0.0)),
  #     num_positives,
  #     name='triplet_semihard_loss')

  # keep track of active count for analysis
  active_count = math_ops.truediv(
          math_ops.reduce_sum(
              math_ops.multiply(math_ops.cast(
                  mask_final,dtype=dtypes.float32), mask_positives)),
          num_positives,
          name='active_count')

  return triplet_loss, active_count

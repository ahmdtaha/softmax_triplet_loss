import tensorflow as tf

def get_center_loss(features, labels, alpha, num_classes):


    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(), trainable=False)

    labels = tf.reshape(labels, [-1])


    centers_batch = tf.gather(centers, labels)

    loss = tf.nn.l2_loss(features - centers_batch)


    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels,diff,name='centers_update_op')

    return loss, centers, centers_update_op,appear_times,diff

# triplet loss
def center_loss(centeriods,anchor):
    pos_dist1 = tf.reduce_sum(tf.square(tf.subtract(centeriods, anchor)), 1)
    return pos_dist1
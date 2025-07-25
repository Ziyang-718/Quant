import tensorflow as tf


def sparsemax(logits, axis=-1):
    """Fully TF-graph-friendly sparsemax."""
    z = logits
    z -= tf.reduce_max(z, axis=axis, keepdims=True)        # 数值稳定

    z_sorted = tf.sort(z, direction='DESCENDING', axis=axis)
    z_cumsum = tf.cumsum(z_sorted, axis=axis)

    shape   = tf.shape(z)          # 动态 shape
    dim     = shape[axis]          # 本轴长度

    # 支持负 axis
    rank = tf.rank(z)
    axis = tf.where(axis < 0, axis + rank, axis)

    # k = [1, 2, ..., dim]  (float 同 logits dtype)
    k = tf.cast(tf.range(1, dim + 1), z.dtype)

    # ---------- 关键：动态生成 broadcast_shape ----------
    # ones_pre  = shape[:axis]  -> 全 1
    # ones_post = shape[axis+1:] -> 全 1
    ones_pre  = tf.ones_like(shape[:axis])
    ones_post = tf.ones_like(shape[axis + 1:])

    broadcast_shape = tf.concat([ones_pre, [dim], ones_post], axis=0)
    k = tf.reshape(k, broadcast_shape)                     # 广播成和 z_sorted 同 rank
    # -----------------------------------------------------

    rhs      = 1 + k * z_sorted
    support  = tf.cast(z_cumsum < rhs, z.dtype)
    k_z      = tf.reduce_sum(support, axis=axis, keepdims=True)

    z_support    = z_sorted * support
    z_cumsum_k   = tf.reduce_sum(z_support, axis=axis, keepdims=True)
    tau          = (z_cumsum_k - 1) / k_z

    return tf.maximum(z - tau, 0.0)
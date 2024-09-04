import tensorflow as tf
import keras.backend as K


def weighted_crossentropy(y_true, y_pred, background_weight=1.,
                          object_weight=1.,
                          boundary_weight=5.):
    class_weights = tf.constant([[[[background_weight,
                                    object_weight,
                                    boundary_weight]]]])

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true, logits=y_pred
    )

    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_losses = weights * unweighted_losses
    loss = tf.reduce_mean(weighted_losses)

    return loss


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

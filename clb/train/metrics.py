import keras.backend as K
import tensorflow as tf


def iou(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) -
                                      intersection + smooth)


def channel_iou(channel, name):
    def iou_func(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true[..., channel])
        y_pred_f = K.flatten(y_pred[..., channel])
        intersection = K.sum(y_true_f * y_pred_f)
        return ((intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

    iou_func.__name__ = name
    return iou_func


def channel_precision(channel, name):
    def precision_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal(K.argmax(y_pred, axis=-1), channel),
                            "float32")

        true_positives = K.sum(K.round(K.clip(
            y_true[:, :, :, channel] * y_pred_tmp, 0, 1
        )))

        predicted_positives = K.sum(K.round(K.clip(y_pred_tmp, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision_func.__name__ = name
    return precision_func


def channel_recall(channel, name):
    def recall_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal(K.argmax(y_pred, axis=-1), channel),
                            "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:, :, :, channel] *
                                              y_pred_tmp, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:, :, :, channel],
                                                  0,
                                                  1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall

    recall_func.__name__ = name
    return recall_func


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

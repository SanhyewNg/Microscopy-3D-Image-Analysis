import io
import os
from copy import copy

import tensorflow as tf
import numpy as np
from keras import callbacks
from keras.preprocessing.image import array_to_img
from PIL import Image, ImageOps

from clb.dataprep.utils import ensure_3d_rgb, ensure_2d_rgb
from clb.utils import chunks


class ValidationCallback(callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        new_logs = {}
        for k, v in logs.items():
            if k.startswith('val_'):
                new_logs[k[len('val_'):]] = v
        return super(ValidationCallback, self).on_epoch_end(epoch, new_logs)


class TrainingCallback(callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        new_logs = {}
        for k, v in logs.items():
            if k.startswith('val_'):
                continue
            new_logs[k] = v
        return super(TrainingCallback, self).on_epoch_end(epoch, new_logs)


class TensorBoardImage(callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        super(TensorBoardImage, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)

    def _convert_image(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """

        if isinstance(tensor, Image.Image):
            height, width, channel = tensor.height, tensor.width, 3
            image = tensor
        else:
            height, width, channel = tensor.shape
            image = Image.fromarray(tensor)

        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

    def get_image_data(self):
        """yield pairs (tag, image_np) that should be included in thedashboard.
        """
        raise NotImplementedError

    def on_epoch_end(self, epoch, logs=None):
        for tag, img in self.get_image_data():
            image = self._convert_image(img)
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                         image=image)])

            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()


class Classification(TensorBoardImage):
    def __init__(self, Xs, Ys, name, only_errors=False, tiles_height=4, log_dir='./logs'):
        super(Classification, self).__init__(log_dir=log_dir)
        self.Xs = Xs
        self.Ys = Ys
        self.only_errors = only_errors
        self.tiles_height = tiles_height
        self.name = name

    def get_image_data(self):
        pred = self.model.predict(self.Xs, batch_size=1)

        xs = copy(self.Xs)
        ys_true = np.array(copy(self.Ys)) > 0.5
        ys = pred > 0.5

        results_to_show = [(x, y, y_true) for x, y, y_true in zip(xs, ys, ys_true) if
                           not self.only_errors or y_true != y]
        concat = visualize_classificator_output(results_to_show, self.tiles_height)
        yield self.name, concat


class Segmentation(TensorBoardImage):
    def __init__(self, Xs, Ys, architecture, log_dir='./logs'):
        self.Xs = Xs
        self.Ys = Ys
        self.architecture = architecture

        super(Segmentation, self).__init__(log_dir=log_dir)

    def get_image_data(self):
        if self.architecture == 'dcan':
            objects, boundaries = self.model.predict(self.Xs, batch_size=1)
        elif self.architecture == 'unet':
            pred = self.model.predict(self.Xs, batch_size=1)

        for i in range(len(self.Xs)):

            x = copy(self.Xs[i,])

            if self.architecture == 'dcan':
                concat = visualize_dcan_output(x=x,
                                               pred_obj=objects[i],
                                               pred_bnd=boundaries[i],
                                               y_obj=self.Ys[0][i],
                                               y_bnd=self.Ys[1][i])
            elif self.architecture == 'unet':
                concat = visualize_unet_output(x=x,
                                               pred=pred[i],
                                               y=self.Ys[i])

            yield 'image_{:03}'.format(i + 1), concat


def TensorBoard(log_dir='./logs', **kwargs):
    training = TrainingCallback(log_dir=os.path.join(log_dir, 'training'),
                                **kwargs)
    validation = ValidationCallback(log_dir=os.path.join(log_dir,
                                                         'validation'),
                                    **kwargs)

    return callbacks.CallbackList([validation, training])


def tile_vertically(imgs):
    concat = Image.new('RGB', (imgs[0].width, len(imgs) * imgs[0].height))
    y_offset = 0
    for im in imgs:
        concat.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return concat


def tile_horizontally(imgs):
    concat = Image.new('RGB', (len(imgs) * imgs[0].width, imgs[0].height))
    x_offset = 0
    for im in imgs:
        concat.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return concat


def visualize_dcan_output(x, pred_obj, pred_bnd, y_obj, y_bnd):
    diff_obj = y_obj - pred_obj
    diff_obj += 1
    diff_obj /= 2

    diff_bnd = y_bnd - pred_bnd
    diff_bnd += 1
    diff_bnd /= 2

    imgs = [x, y_obj, pred_obj, diff_obj, y_bnd, pred_bnd, diff_bnd]
    imgs = [array_to_img(x) for x in imgs]
    imgs = [ImageOps.expand(x, border=4, fill='black') for x in imgs]

    return tile_horizontally(imgs)


def visualize_unet_output(x, pred, y):
    diff = y - pred
    diff += 1
    diff /= 2

    imgs = [x, y, pred, diff]
    imgs = [array_to_img(x) for x in imgs]
    imgs = [ImageOps.expand(x, border=2, fill='black') for x in imgs]

    return tile_horizontally(imgs)


def visualize_classificator_output(results_to_show, tiles_height):
    """
    Present the results as tiles with border color corresponding to the evaluation.
    Yellow is false negative and red is false positive.
    Args:
        results_to_show: list of (input_data, prediction, ground truth)
            Input can be either 2D single/multi channel or 3D with channels
                in case of 3D the middle slice is chosen to show.
        tiles_height: number of tiles to show vertically in one column

    Returns:
        PIL Image showing all tiles.
    """
    imgs = []
    for input_data, prediction, gt in results_to_show:
        if prediction == gt:
            colour = 'green'
        elif prediction > gt:
            colour = 'red'
        else:
            colour = 'yellow'

        if input_data.ndim > 3:  # if 4d then assume Z, Y, X ,C and pick middle slice
            input_data = input_data[len(input_data) // 2]
        input_data = ensure_2d_rgb(input_data)
        img = array_to_img(input_data)
        img = ImageOps.expand(img, border=4, fill=colour)
        imgs.append(img)

    vertical_stripes = [tile_vertically(chunk) for chunk in chunks(imgs, tiles_height)]
    tiles = tile_horizontally(vertical_stripes)

    return tiles

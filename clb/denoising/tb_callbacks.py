import io

import PIL.Image as PImage
import numpy as np
import skimage
import tensorflow as tf
import keras.callbacks as callbacks


class TensorBoardImage(callbacks.Callback):
    def __init__(self, log_dir, dataset):
        super().__init__()

        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(log_dir)
        self.dataset = dataset

    @staticmethod
    def _convert_image(image):
        height, width = image.shape[:2]
        image = PImage.fromarray(np.squeeze(image)).convert('L')

        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        input_image, target_image = next(self.dataset)
        output_image = self.model.predict(input_image[0:1])
        image = np.hstack([np.squeeze(skimage.img_as_ubyte(input_image[0])),
                           np.squeeze(skimage.img_as_ubyte(target_image[0])),
                           np.squeeze(skimage.img_as_ubyte(output_image[0]))])
        summary = self._convert_image(image)
        summary = tf.Summary(value=[tf.Summary.Value(tag='image',
                                                     image=summary)])
        self.writer.add_summary(summary, global_step=epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()

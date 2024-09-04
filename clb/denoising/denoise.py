"""Module defines tools for denoising images using ready model."""
import os

import fire
import imageio
import keras.models as models
import numpy as np
import skimage
import tqdm

import clb.dataprep.readers as readers
import clb.dataprep.utils as utils
import clb.denoising.io as denoiseio
import clb.denoising.postprocess as postproc
import clb.denoising.preprocess as preproc


def denoise_image(image, model, batch_size, patches_shape=(128, 128),
                  patches_stride=(32, 32)):
    """Denoise single image.

    Image is first cut to patches with stride `patches_stride`. Then patches
    are denoised using `denoise_patches`. Denoised patches are merged by taking
    average on overlapping areas.

    Args:
        image (np.ndarray): 2d input image.
        model (keras.Model): Model used for inference.
        batch_size (int): Maximum number of patches going into network at once.
        patches_shape (tuple|None): Shape of patches going into model in form
                                    of (y, x).
        patches_stride (tuple|None): Stride for patches extraction in form of
                                     (s_y, s_x).
    Returns:
        np.ndarray: Denoised image, shape (Y, X), dtype float.
    """
    image = skimage.img_as_float32(image)
    if patches_shape is not None:
        patches = preproc.extract_patches(np.squeeze(image),
                                          shape=patches_shape,
                                          stride=patches_stride)
        denoised_patches = model.predict(utils.ensure_4d(patches), batch_size=batch_size)
        denoised_image = postproc.merge_patches(patches=np.squeeze(denoised_patches),
                                                image_shape=np.squeeze(image).shape,
                                                stride=patches_stride)
    else:
        image = utils.ensure_4d(image)
        image = utils.rescale_to_float(image, 'float32')
        denoised_image = model.predict(image)

    return np.squeeze(denoised_image)


def denoise_stack(images, model, batch_size, patches_shape=(128, 128),
                  patches_stride=(32, 32)):
    """Denoise stack of images.

    Args:
        images (Iterable): Input images.
        model (keras.Model): Model used for inference.
        batch_size (int): Maximum number of patches going into network at once.
        patches_shape (tuple|None): Shape of one patch. If None network will
                                  take whole image.
        patches_stride (tuple): Stride for patches.

    Returns:
        Iterator: Denoised images.
    """
    denoised_stack = (denoise_image(image=image,
                                    model=model,
                                    batch_size=batch_size,
                                    patches_shape=patches_shape,
                                    patches_stride=patches_stride)
                      for image in images)

    return denoised_stack


class FileExtensionError(Exception):
    """Raised when file has wrong exception."""


def denoise(input,
            output,
            model,
            series=0,
            channel=0,
            start=0,
            stop=None,
            batch_size=1,
            patches_shape=None,
            patches_stride=None):
    """Denoise image at `input` and save result at `output`.

    For now only .png images are supported, `out` is also expected to be
    .png. Directory part of the `out` should also exist.

    Args:
        input (str): Path to input image(s). Should be .tif, .lif or .uff
                     (unrecognized extension will be treated as .uff).
        output (str): Path to output image(s). Should be .tif.
        model (str): Path to model.
        series (int): Series to denoise (applies only for .lif files).
        channel (int): Channel to denoise.
        start (int): Starting slice.
        stop (int|None): Stopping slice.
        batch_size (int): Size of batch of patches going into the network at
                          once.
        patches_shape (tuple): Shape of patches used when denoising bigger
                               image. Should be in form y,x,1.
        patches_stride (tuple): Stride for extracting patches when
                                denoising bigger image, should be in form
                                s_y,s_x,1.
    """
    # Checking output path.
    dirname, basename = os.path.split(os.path.abspath(output))
    if not os.path.isdir(dirname):
        raise FileNotFoundError('{} - directory does not exist.'.format(dirname))

    model = models.load_model(model)

    if input.endswith('.png'):
        image = denoiseio.read_png(input)

        denoised_image = denoise_image(image=image,
                                       model=model,
                                       batch_size=batch_size,
                                       patches_shape=patches_shape,
                                       patches_stride=patches_stride)
        imageio.mimwrite(output, [skimage.img_as_ubyte(denoised_image)])
    else:
        with readers.get_volume_reader(path=input, series=series) as reader:
            images = reader[start:stop, channel]
            denoised_images = denoise_stack(model=model,
                                            images=images,
                                            batch_size=batch_size,
                                            patches_shape=patches_shape,
                                            patches_stride=patches_stride)
            denoised_images = (skimage.img_as_ubyte(img) for img in denoised_images)
            imageio.mimwrite(output, tqdm.tqdm(denoised_images, total=len(images)))


if __name__ == '__main__':
    fire.Fire(denoise)

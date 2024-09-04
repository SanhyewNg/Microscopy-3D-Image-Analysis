import os
import pathlib
import clb.denoising.denoise as denoise
import cv2
import fire
import keras.models as models
import pandas as pd
import skimage
import skimage.measure as skimeasure


def get_sorted_paths(dir_path):
    paths = os.listdir(dir_path)
    paths.sort()
    paths = [os.path.join(dir_path, path) for path in paths]
    return paths


def calculate_metric(pred_paths, gt_paths, metric_fun):
    predictions = (skimage.img_as_float32(cv2.imread(path, cv2.IMREAD_UNCHANGED))
                   for path in pred_paths)
    gts = (skimage.img_as_float32(cv2.imread(path, cv2.IMREAD_UNCHANGED))
           for path in gt_paths)
    paths_to_metric = {
        path: metric_fun(gt, pred)
        for path, gt, pred in zip(pred_paths, gts, predictions)
    }
    return paths_to_metric


def create_metric_df(metrics):
    return pd.DataFrame.from_dict(metrics)


def make_predictions(model_path, input_dir, output_dir):
    model = models.load_model(str(model_path))
    output_dir.mkdir(parents=True)
    for path in pathlib.Path(input_dir).glob('*'):
        input_image = skimage.img_as_float32(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        output_image = denoise.denoise_image(input_image, model, batch_size=1)
        cv2.imwrite(str(output_dir / path.name),
                    skimage.img_as_ubyte(output_image))


def calculate_metrics(predictions_dir, gt_dir):
    pred_paths, gt_paths = get_sorted_paths(predictions_dir), get_sorted_paths(gt_dir)

    metrics = {
        'psnr': calculate_metric(pred_paths, gt_paths, skimeasure.compare_psnr),
        'ssim': calculate_metric(pred_paths, gt_paths, skimeasure.compare_ssim)
    }

    return metrics


def main(output, model_paths, input_dir, gt_dir):
    for model in model_paths:
        model_path = pathlib.Path(model)
        output_path = pathlib.Path(output)
        predictions_dir = output_path / model_path.name
        make_predictions(model, input_dir, predictions_dir)
        metrics = calculate_metrics(predictions_dir, gt_dir)

        df = create_metric_df(metrics)
        df.loc['mean'] = df.mean(axis='rows')
        df.to_csv(output / (str(model_path.name) + '.csv'))


if __name__ == '__main__':
    fire.Fire(main)

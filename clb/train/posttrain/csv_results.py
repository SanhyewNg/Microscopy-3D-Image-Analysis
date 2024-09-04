import os

import fire
import pandas as pd
from tqdm import tqdm


def gather_results(logdir, id_column, out_file=None, logfile_name='log.csv',
                   multiple_trainings=True, monitored_metric='val_loss',
                   minimize_metric=True, beautify=None):
    """Gather best experiments' results into single CSV file.

    This functionality can work both for directory with multiple results
    from various experiments and for single experiment's results.

    Args:
        logdir (str): path to directory with different models' logs (can be
                      a path to single model's training as well).
        id_column (str): name of the column that will serve as identifier
        out_file (str): path to output CSV file.
        logfile_name (str): what is the name of CSV log file (it should be
                            located in each model's output directory).
        multiple_trainings (bool): does `logdir` contain directories with
                                   results from multiple trainings
                                   (otherwise it will be treated as a single
                                   training)
        monitored_metric (str): what is the metric that we want to use to
                                choose best epoch in given training. By
                                default it's validation loss.
        minimize_metric (bool): should the metric be minimized (lowest value
                                is the best one) or not (highest value is
                                the best one).
        beautify (str): should values from resulting
                         dataframe be beautified (converted
                         to desired types, rounded etc.)
                         according to the type of results:
                         - DCAN
                         - VGG

    Returns:
        if `out_file == None`, then dataframe with best results is returned;
        if `out_file` provided, CSV file is created.
    """
    if multiple_trainings:
        models_dirs = os.listdir(logdir)
    else:
        models_dirs = [os.path.basename(logdir)]

    results = pd.DataFrame()

    for directory in tqdm(models_dirs):
        try:
            if multiple_trainings:
                csv_path = os.path.join(logdir, directory, logfile_name)
            else:
                csv_path = os.path.join(logdir, logfile_name)

            df = pd.read_csv(csv_path)
            df = df.sort_values(by=[monitored_metric],
                                ascending=minimize_metric)

            best_row = df.iloc[0]
            # Append model name to easily match values with the model.
            best_row.loc[id_column] = directory

            results = results.append(best_row)
        except FileNotFoundError:
            # Old trainings do not have CSV log files, skip that fact.
            pass

    if beautify:
        results = beautify_gathered_results(results, beautify)

    if out_file:
        results.to_csv(out_file, index=False)
    else:
        return results


def beautify_gathered_results(dataframe, method):
    """Convert dataframe to be more readable.

    Args:
        dataframe: specific dataframe with results that should be beautified.
        method: what method of conversion to use:
                    - DCAN - assume that dataframe comes with DCAN metrics
                    - VGG - assume that dataframe comes with VGG metrics

    Returns:
        beautified dataframe
    """
    dataframe['epoch'] = pd.to_numeric(dataframe['epoch'], downcast='integer')

    invalid_frame_message = "Dataframe given doesn't match required dataframe format: {0}"
    if method == "DCAN":
        if not {'epoch', 'model', 'val_loss', 'val_boundaries_iou',
                'val_objects_iou'}.issubset(dataframe.columns):
            raise ValueError(invalid_frame_message.format("DCAN"))

        dataframe = dataframe.round({'val_loss': 4, 'val_objects_iou': 4,
                                     'val_boundaries_iou': 4})

        dataframe = dataframe[['model', 'val_loss', 'val_objects_iou',
                               'val_boundaries_iou', 'epoch']]

        dataframe = dataframe.sort_values(by=['val_objects_iou',
                                              'val_boundaries_iou'],
                                          ascending=False)
    elif method == "VGG":
        if not {'epoch', 'model', 'val_loss', 'val_f1'}.issubset(dataframe.columns):
            raise ValueError(invalid_frame_message.format("VGG"))

        dataframe = dataframe.round({'val_loss': 3, 'val_f1': 3})

        dataframe = dataframe[['model', 'val_loss', 'val_f1', 'epoch']]

        dataframe = dataframe.sort_values(by=['val_f1'],
                                          ascending=False)
    else:
        raise ValueError("Unsupported beautify method: {0}".format(method))
    return dataframe


if __name__ == '__main__':
    fire.Fire(gather_results)

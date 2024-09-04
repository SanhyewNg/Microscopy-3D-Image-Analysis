import fire
import gspread
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from tqdm import tqdm

from clb.train.posttrain.csv_results import gather_results


class GSheetsManager:

    scopes = ['https://spreadsheets.google.com/feeds']

    def connect(self, service_account_file):
        """Connect using provided credentials.

        Args:
            service_accout_file (str): path to JSON file with credentials
                                       required to manage Google Sheets API.

        Creates:
            self.client object that can be used to open sheets from Google.
        """
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=self.scopes)

        self.client = gspread.Client(auth=credentials)
        self.client.session = AuthorizedSession(credentials)

    def __init__(self, service_account_file='sheets_api_credentials.json'):
        self.connect(service_account_file)

    def upload_dataframe(self, dataframe, spreadsheet_id, id_column,
                         columns_to_upload, worksheet=None, overwrite=False):
        """Upload specified data from dataframe into Google Sheets document.

        Args:
            dataframe: pandas dataframe to be uploaded
            spreadsheet_id (str): unique identifier of the spreadsheet
            id_column (str): name of the column that serves as ID column
            columns_to_upload (iterable of strings): list or tuple of
                                                     columns that will be
                                                     uploaded from given
                                                     `dataframe`.
            worksheet (str): name of the worksheet to work on. If no value
                             provided, by default first sheet will be used.
            overwrite (bool): should existing values in specific cells be
                              overwritten
        """

        sheet = self.client.open_by_key(spreadsheet_id)

        if worksheet:
            worksheet = sheet.worksheet(worksheet)
        else:
            worksheet = sheet.sheet1

        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe.index)):
            # Find ID in ID column (to know specific experiment's row).
            experiment_row = worksheet.find(row[id_column]).row

            for column in columns_to_upload:
                col_idx = worksheet.find(column).col

                if worksheet.cell(experiment_row, col_idx).value == '' or overwrite:
                    worksheet.update_cell(experiment_row, col_idx, row[column])
                else:
                    # If any of the columns are already present, let's not try
                    # to fill others since it may almost for sure mean given
                    # experiment has already been uploaded. It will save some
                    # read operations (that are limited by Google Cloud
                    # quota).
                    break

    def get_and_upload_best_results(self, logdir, spreadsheet_id, id_column,
                                    columns_to_upload,
                                    logfile_name='log.csv',
                                    multiple_trainings=True,
                                    monitored_metric='val_loss',
                                    minimize_metric=True, beautify=True,
                                    worksheet=None, overwrite=False):
        """Gather results and upload them to Google Sheets document.

        This functionality can work both for directory with multiple results
        from various experiments and for single experiment's results.

        Args:

            logdir (str): path to directory with different models' logs (can be
                        a path to single model's training as well).
            spreadsheet_id (str): unique identifier of the spreadsheet
            id_column (str): name of the column that will serve as identifier
            columns_to_upload (iterable of strings): list or tuple of
                                                     columns that will be
                                                     uploaded from given
                                                     `dataframe`.
            logfile_name (str): what is the name of CSV log file (it should be
                                located in each model's output directory).
            multiple_trainings (bool): does `logdir` contain directories with
                                       results from multiple trainings
                                       (otherwise it will be treated as a
                                       single training)
            monitored_metric (str): what is the metric that we want to use to
                                    choose best epoch in given training. By
                                    default it's validation loss.
            minimize_metric (bool): should the metric be minimized (lowest
                                    value is the best one) or not (highest
                                    value is the best one).
            worksheet (str): name of the worksheet to work on. If no value
                             provided, by default first sheet will be used.
            overwrite (bool): should existing values in specific cells be
                              overwritten
        """
        dataframe = gather_results(logdir=logdir, id_column=id_column,
                                   logfile_name=logfile_name,
                                   multiple_trainings=multiple_trainings,
                                   monitored_metric=monitored_metric,
                                   minimize_metric=minimize_metric,
                                   beautify=beautify)

        self.upload_dataframe(dataframe=dataframe,
                              spreadsheet_id=spreadsheet_id,
                              id_column=id_column,
                              columns_to_upload=columns_to_upload,
                              worksheet=worksheet, overwrite=overwrite)


if __name__ == '__main__':
    fire.Fire(GSheetsManager)

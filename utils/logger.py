"""CSV logger"""

import csv

from tensorflow.keras.callbacks import Callback


class MyCSVLogger(Callback):

    def __init__(self, filename):
        self.filename = filename

    def on_test_begin(self, logs=None):
        self.csv_file = open(self.filename, "a")

        class CustomDialect(csv.excel):
            delimiter = ','

        self.fieldnames = ['error [%]']
        self.writer = csv.DictWriter(self.csv_file, self.fieldnames, dialect=CustomDialect)
        self.writer.writeheader()

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        logs = {'error [%]': 100.0 - logs['accuracy'] * 100.0}
        self.writer.writerow(logs)
        self.csv_file.flush()

    def on_test_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

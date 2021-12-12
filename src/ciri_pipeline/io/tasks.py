import logging
import os
import pathlib
import random
import shutil
from statistics import mean

import luigi
from ciri_pipeline.settings import GCS_DATA_BUCKET, GCS_PROJECT_NAME, PIPELINE_DATA_DIR
from google.cloud import storage


class DownloadTrainingFilesTask(luigi.Task):
    """Downloads files from CIRC GCS Bucket."""

    _data_dir = PIPELINE_DATA_DIR + "/raw"
    _gcs_project_name = GCS_PROJECT_NAME
    _gcs_bucket_name = GCS_DATA_BUCKET

    def output(self):
        return luigi.LocalTarget(self._data_dir)

    def run(self):
        # Anonymous client is used as bucket is public-read:
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(self._gcs_bucket_name)

        # Find all content in bucket:
        blobs = bucket.list_blobs()

        # Create data-folder:
        os.makedirs(PIPELINE_DATA_DIR, exist_ok=True)
        os.makedirs(self._data_dir, exist_ok=True)

        # Download blob files:
        for blob in blobs:
            destination = os.path.join(self._data_dir, blob.name)
            directory = os.path.dirname(os.path.abspath(destination))

            # Create sub-directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            if not blob.name.endswith("/"):
                if not os.path.exists(destination):
                    blob.download_to_filename(destination)
                    logging.info(f"Downloaded: {blob.name}")
                else:
                    logging.info(f"Already exists: {blob.name}")


class LimitTrainingFiles(luigi.Task):
    """Limits the number of files that are used to some max.

    Used to prevent over-balanced sampling if uneven category sets.
    """

    _data_dir = PIPELINE_DATA_DIR + "/limited"
    _upper_threshold = 1.15

    def requires(self):
        return DownloadTrainingFilesTask()

    def output(self):
        return luigi.LocalTarget(self._data_dir)

    def run(self):
        input_folder = self.input().path

        min_file_cnt = (
            mean([len(files) for r, d, files in os.walk(input_folder)])
            * self._upper_threshold
        )

        # Iterate through all files and seperate:
        for root, dirs, files in os.walk(input_folder):
            sub_dir = pathlib.PurePath(root).name
            for filename in random.sample(files, min(len(files), min_file_cnt)):

                destination_dir = os.path.join(self._data_dir, sub_dir)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir, exist_ok=True)

                # Copy file to assigned folder:
                shutil.copy(
                    os.path.join(root, filename),
                    os.path.join(destination_dir, filename),
                )

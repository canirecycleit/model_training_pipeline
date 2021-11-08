import logging
import os

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
        os.makedirs(self._data_dir, exist_ok=True)

        # Download blob files:
        for blob in blobs:
            destination = os.path.join(self._data_dir, blob.name)
            directory = os.path.dirname(os.path.abspath(destination))

            # Create sub-directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            if not blob.name.endswith("/"):
                blob.download_to_filename(destination)
                logging.info(f"Downloaded: {blob.name}")

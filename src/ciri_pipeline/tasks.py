import os

import luigi
from google.cloud import storage


class DownloadTrainingFiles(luigi.Task):

    _data_dir = "./data/training"
    _gcs_project_name = "CIRI"
    _gcs_bucket_name = "canirecycleit-data"

    def output(self):
        return luigi.LocalTarget(self._data_dir)

    def run(self):
        # storage_client = storage.Client(project=self._gcs_project_name)
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(self._gcs_bucket_name)

        # Find all content in bucket:
        blobs = bucket.list_blobs()

        # Create data-folder:
        os.makedirs(self._data_dir)

        # Download
        for blob in blobs:
            destination = os.path.join(self._data_dir, blob.name)
            directory = os.path.dirname(os.path.abspath(destination))

            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            if not blob.name.endswith("/"):
                blob.download_to_filename(destination)
                print(f"Downloaded: {blob.name}")

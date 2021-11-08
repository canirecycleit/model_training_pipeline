import os
import pathlib
import shutil

import luigi
import tensorflow as tf
from ciri_pipeline.io.tasks import DownloadTrainingFilesTask
from numpy.random import choice


class SplitTrainingValidationTask(luigi.Task):

    _data_dir = "./data"
    _validation_percent = 0.2

    _validation_output = "raw_validation"
    _training_output = "raw_training"

    def requires(self):
        return DownloadTrainingFilesTask()

    def output(self):
        train_output = os.path.join(self._data_dir, self._training_output)
        validation_output = os.path.join(self._data_dir, self._validation_output)

        return [luigi.LocalTarget(train_output), luigi.LocalTarget(validation_output)]

    def run(self):

        input_folder = self.input().path

        label_names = os.listdir(input_folder)
        label2index = dict((name, index) for index, name in enumerate(label_names))

        os.makedirs(os.path.join(self._data_dir, self._training_output))
        os.makedirs(os.path.join(self._data_dir, self._validation_output))

        for root, dirs, files in os.walk(input_folder):
            for filename in files:

                sub_dir = pathlib.PurePath(root).name
                label_dir = str(label2index[sub_dir])

                draw = choice(
                    [self._training_output, self._validation_output],
                    1,
                    p=[1 - self._validation_percent, self._validation_percent],
                )[0]

                destination_dir = os.path.join(self._data_dir, draw, label_dir)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir, exist_ok=True)

                shutil.copy(
                    os.path.join(root, filename),
                    os.path.join(destination_dir, filename),
                )


class TrainTFRecordTask(luigi.Task):

    _batch_size = 128
    _image_width = 256
    _image_height = 192
    _num_channels = 3

    _num_shards = 10

    _data_dir = "./data"
    _training_output = "tfrecord_training"

    def requires(self):
        return SplitTrainingValidationTask()

    def output(self):
        return luigi.LocalTarget(os.path.join(self._data_dir, self._training_output))

    def create_tf_example(self, item):

        # Read image
        image = tf.io.read_file(item[1])
        image = tf.image.decode_jpeg(image, channels=self._num_channels)
        image = tf.image.resize(image, [self._image_height, self._image_width])
        image = tf.cast(image, tf.uint8)

        # Label
        label = int(item[0])

        # Build feature dict
        feature_dict = {
            "image": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()])
            ),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "width": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[self._image_width])
            ),
            "height": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[self._image_height])
            ),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def create_tf_records(self, data, num_shards=10, prefix="", folder="data"):
        num_records = len(data)
        step_size = num_records // num_shards + 1

        for i in range(0, num_records, step_size):
            print(
                "Creating shard:",
                (i // step_size),
                " from records:",
                i,
                "to",
                (i + step_size),
            )
            path = "{}/{}_000{}.tfrecords".format(folder, prefix, i // step_size)
            print(path)

            # Write the file
            with tf.io.TFRecordWriter(path) as writer:
                # Filter the subset of data to write to tfrecord file
                for item in data[i : i + step_size]:
                    tf_example = self.create_tf_example(item)
                    writer.write(tf_example.SerializeToString())

    def run(self):

        input_folder = self.input()[0].path
        labels = os.listdir(input_folder)

        data_list = []
        for label in labels:
            image_files = os.listdir(os.path.join(input_folder, label))
            data_list.extend(
                [(label, os.path.join(input_folder, label, f)) for f in image_files]
            )

        os.makedirs(os.path.join(self._data_dir, self._training_output))

        # Create TF Records for train
        self.create_tf_records(
            data_list,
            num_shards=self._num_shards,
            folder=os.path.join(self._data_dir, self._training_output),
        )

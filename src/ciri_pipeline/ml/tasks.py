import os
import pathlib
import shutil

import luigi
import mlflow
import tensorflow as tf
import tensorflow_hub as hub
from ciri_pipeline import ml
from ciri_pipeline.io.tasks import DownloadTrainingFilesTask
from keras.callbacks import EarlyStopping
from numpy.random import choice
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


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


class BuildTFRecordTask(luigi.Task):

    _batch_size = 128
    _image_width = 256
    _image_height = 192
    _num_channels = 3

    _num_shards = 10

    _data_dir = "./data"
    _folder_output = "tfrecord_output"

    def requires(self):
        return SplitTrainingValidationTask()

    def output(self):
        return luigi.LocalTarget(os.path.join(self._data_dir, self._folder_output))

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

        os.makedirs(os.path.join(self._data_dir, self._folder_output))

        # Create TF Records for train
        self.create_tf_records(
            data_list,
            num_shards=self._num_shards,
            folder=os.path.join(self._data_dir, self._folder_output),
        )


class TrainTFRecordTask(BuildTFRecordTask):

    _folder_output = "tfrecord_training"


class ValidationTFRecordTask(BuildTFRecordTask):

    _folder_output = "tfrecord_validation"


class TrainModel(luigi.Task):

    _batch_size = 128

    _image_width = 256
    _image_height = 192
    _num_channels = 3

    # TODO: This shouldn't be hard-coded
    _num_classes = 3

    _feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
    }

    def requires(self):
        return TrainTFRecordTask()

    def parse_tfrecord(self, proto):
        parsed_record = tf.io.parse_single_example(proto, self._feature_description)

        image = tf.io.decode_raw(parsed_record["image"], tf.uint8)
        image.set_shape([self._num_channels * self._image_height * self._image_width])
        image = tf.reshape(
            image, [self._image_height, self._image_width, self._num_channels]
        )
        # Label
        label = tf.cast(parsed_record["label"], tf.int32)
        # label = tf.one_hot(label, num_classes)

        return image, label

    # Normalize pixels
    def normalize(self, image, label):
        image = image / 255
        return image, label

    def build_transfer_model(self):
        """Build a transfer model based on mobile-net pre-trained architecture."""
        input_shape = [
            self._image_height,
            self._image_width,
            self._num_channels,
        ]  # height, width, channels
        handle = (
            "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5"
        )

        # Regularize using L1
        kernel_weight = 0.02
        bias_weight = 0.02

        model = Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                hub.KerasLayer(handle, trainable=False),
                keras.layers.Dense(
                    units=124,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l1(kernel_weight),
                    bias_regularizer=keras.regularizers.l1(bias_weight),
                ),
                keras.layers.Dense(
                    units=64,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l1(kernel_weight),
                    bias_regularizer=keras.regularizers.l1(bias_weight),
                ),
                keras.layers.Dense(
                    units=self._num_classes,
                    activation=None,
                    kernel_regularizer=keras.regularizers.l1(kernel_weight),
                    bias_regularizer=keras.regularizers.l1(bias_weight),
                ),
            ],
            name="transfer_model",
        )

        return model

    def run(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        input_folder = self.input().path
        train_tfrecord_files = tf.data.Dataset.list_files(input_folder + "/*")

        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.2),
                layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
            ]
        )

        # Train pipeline:
        train_data = train_tfrecord_files.flat_map(tf.data.TFRecordDataset)
        train_data = train_data.map(self.parse_tfrecord, num_parallel_calls=AUTOTUNE)
        train_data = train_data.map(self.normalize, num_parallel_calls=AUTOTUNE)
        train_data = train_data.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        train_data = train_data.batch(self._batch_size)
        train_data = train_data.prefetch(buffer_size=AUTOTUNE)

        # Model Training Parameters
        learning_rate = 0.01
        decay_rate = 0.5
        epochs = 2

        # Model parameters
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        es = EarlyStopping(monitor="val_accuracy", verbose=1, patience=3)
        lr = keras.callbacks.LearningRateScheduler(
            lambda epoch: learning_rate / (1 + decay_rate * epoch)
        )

        with mlflow.start_run():

            # Execute different model approaches and save experiment results:
            model = self.build_transfer_model()

            print(model.summary())
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=["accuracy", "sparse_categorical_accuracy"],
            )

            # Train model
            training_results = model.fit(
                train_data,
                validation_data=train_data,
                epochs=epochs,
                callbacks=[es, lr],
                verbose=1,
            )

            mlflow.log_param("decay_rate", decay_rate)
            mlflow.log_param("learning_rate", learning_rate)

            mlflow.keras.log_model(
                model, "model", registered_model_name="ciri_trashnet_model"
            )

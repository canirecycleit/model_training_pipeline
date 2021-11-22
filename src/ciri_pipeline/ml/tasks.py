import json
import logging
import os
import pathlib
import shutil

import ciri_pipeline.settings as settings
import luigi
import mlflow
import tensorflow as tf
import tensorflow_hub as hub
from ciri_pipeline.io.tasks import DownloadTrainingFilesTask
from keras.callbacks import EarlyStopping
from numpy.random import choice
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Set log-level to INFO:
logging.basicConfig(level=logging.INFO)


class SplitTrainingValidationTask(luigi.Task):
    """Splits files into training and validation folders."""

    _data_dir = settings.PIPELINE_DATA_DIR
    _validation_percent = settings.VALIDATION_PCNT

    _validation_output = "raw_validation"
    _training_output = "raw_training"
    _label_map_output_folder = "label_map"
    _label_map_output_file = "mapping.json"

    def requires(self):
        return DownloadTrainingFilesTask()

    def output(self):
        train_output = os.path.join(self._data_dir, self._training_output)
        validation_output = os.path.join(self._data_dir, self._validation_output)
        label_map_output = os.path.join(
            self._data_dir, self._label_map_output_folder, self._label_map_output_file
        )

        # 3 Outputs including 1) training data, 2) validation data
        # & a 3) map of label values:
        return [
            luigi.LocalTarget(train_output),
            luigi.LocalTarget(validation_output),
            luigi.LocalTarget(label_map_output),
        ]

    def serialize_label_mapping(self, label_dict):
        """Serializes the label mapping for later Task usage."""
        # Create label-mapping folder:
        label_mapping_output_folder = os.path.join(
            self._data_dir, self._label_map_output_folder
        )
        os.makedirs(label_mapping_output_folder, exist_ok=True)

        # Serialize mapping:
        label_mapping_file = os.path.join(
            label_mapping_output_folder, self._label_map_output_file
        )
        with open(label_mapping_file, "w") as f:
            f.write(json.dumps(label_dict))
            logging.info("Serialized mapping.json file.")

    def run(self):
        logging.info(f"Running: {self.__class__.__name__}")

        # Get downloaded raw data:
        input_folder = self.input().path

        # Get existing labels (via sub-folders):
        label_names = os.listdir(input_folder)
        label2index = dict((name, index) for index, name in enumerate(label_names))

        # Create a label look-up for later Tasks:
        index2label = dict((index, name) for index, name in enumerate(label_names))
        self.serialize_label_mapping(index2label)

        # Create training/validation output folders:
        os.makedirs(os.path.join(self._data_dir, self._training_output))
        os.makedirs(os.path.join(self._data_dir, self._validation_output))

        # Iterate through all files and seperate:
        for root, dirs, files in os.walk(input_folder):
            for filename in files:

                # Convert to index:
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

                # Copy file to assigned folder:
                shutil.copy(
                    os.path.join(root, filename),
                    os.path.join(destination_dir, filename),
                )


class BuildTFRecordTask(luigi.Task):
    """Serialize Image Files as TFRecords."""

    _batch_size = settings.BATCH_SIZE
    _image_width = settings.IMAGE_WIDTH
    _image_height = settings.IMAGE_HEIGHT
    _num_channels = 3

    _num_shards = settings.NUM_SHARDS

    _data_dir = settings.PIPELINE_DATA_DIR
    _folder_output = "tfrecord_output"

    # Selects which index from SplitTrainingValidationTask
    # to use (0=Training, 1=Validation)
    _input_index = 0

    def requires(self):
        return SplitTrainingValidationTask()

    def output(self):
        tfrecord_output_folder = os.path.join(self._data_dir, self._folder_output)
        return luigi.LocalTarget(tfrecord_output_folder)

    def create_tf_example(self, item):
        """Create a TFRecord Example for serialization.

        param: item - tuple (label, file_path)
        """

        file_path = item[1]
        label = int(item[0])

        # Read image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=self._num_channels)
        image = tf.image.resize(image, [self._image_height, self._image_width])
        image = tf.cast(image, tf.uint8)

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
        """Create TFRecord shards."""

        num_records = len(data)
        step_size = num_records // num_shards + 1

        for i in range(0, num_records, step_size):
            logging.info(
                "Creating shard:" +
                str(i // step_size) +
                " from records:" +
                str(i) +
                "to" +
                str(i + step_size)
            )
            path = "{}/{}_000{}.tfrecords".format(folder, prefix, i // step_size)
            logging.info(path)

            # Write the file
            with tf.io.TFRecordWriter(path) as writer:
                # Filter the subset of data to write to tfrecord file
                for item in data[i : i + step_size]:
                    tf_example = self.create_tf_example(item)
                    writer.write(tf_example.SerializeToString())

    def run(self):
        logging.info(f"Running: {self.__class__.__name__}")

        # Guard-statement:
        if self._input_index > 1:
            raise ValueError("_input_index must be less than 2")

        input_folder = self.input()[self._input_index].path

        labels = os.listdir(input_folder)
        data_list = []
        for label in labels:
            image_files = os.listdir(os.path.join(input_folder, label))
            data_list.extend(
                [(label, os.path.join(input_folder, label, f)) for f in image_files]
            )

        # Create output folder:
        os.makedirs(os.path.join(self._data_dir, self._folder_output))

        # Create TF Records:
        self.create_tf_records(
            data_list,
            num_shards=self._num_shards,
            folder=os.path.join(self._data_dir, self._folder_output),
        )


class TrainTFRecordTask(BuildTFRecordTask):

    _input_index = 0
    _folder_output = "tfrecord_training"


class ValidationTFRecordTask(BuildTFRecordTask):

    _input_index = 1
    _folder_output = "tfrecord_validation"


class TrainModel(luigi.Task):

    _batch_size = settings.BATCH_SIZE

    _image_width = settings.IMAGE_WIDTH
    _image_height = settings.IMAGE_HEIGHT
    _num_channels = 3

    _data_dir = settings.PIPELINE_DATA_DIR
    _output_folder = "trained_model"

    _feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
    }

    def requires(self):
        return [
            TrainTFRecordTask(),
            ValidationTFRecordTask(),
            SplitTrainingValidationTask(),
        ]

    def output(self):
        output_folder = os.path.join(self._data_dir, self._output_folder)
        return [luigi.LocalTarget(output_folder)]

    def parse_tfrecord(self, proto):
        """Parses a serialized TF Record."""
        parsed_record = tf.io.parse_single_example(proto, self._feature_description)

        image = tf.io.decode_raw(parsed_record["image"], tf.uint8)
        image.set_shape([self._num_channels * self._image_height * self._image_width])
        image = tf.reshape(
            image, [self._image_height, self._image_width, self._num_channels]
        )

        label = tf.cast(parsed_record["label"], tf.int32)

        return image, label

    def normalize(self, image, label):
        """Normalize pixels between 0 and 1."""
        image = image / 255
        return image, label

    def build_transfer_model(self, num_classes):
        """Build a transfer model based on pre-trained architecture."""
        input_shape = [
            self._image_height,
            self._image_width,
            self._num_channels,
        ]

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
                    units=num_classes,
                    activation=None,
                    kernel_regularizer=keras.regularizers.l1(kernel_weight),
                    bias_regularizer=keras.regularizers.l1(bias_weight),
                ),
            ],
            name="transfer_model",
        )

        return model

    def build_pipeline(self, tfrecordfiles, augment=False):
        """Builds pipeline from TFRecord"""

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # Optional image augmentation layer:
        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.2),
                layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
            ]
        )

        data = tfrecordfiles.flat_map(tf.data.TFRecordDataset)
        data = data.map(self.parse_tfrecord, num_parallel_calls=AUTOTUNE)
        data = data.map(self.normalize, num_parallel_calls=AUTOTUNE)

        if augment:
            data = data.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )

        data = data.batch(self._batch_size)
        data = data.prefetch(buffer_size=AUTOTUNE)
        return data

    def run(self):
        """Build data pipeline and train resulting model."""

        logging.info(f"Running: {self.__class__.__name__}")

        training_input_folder = self.input()[0].path
        validation_input_folder = self.input()[1].path

        index2label = {}
        label_mapping = self.input()[-1][2].path
        with open(label_mapping, "r") as f:
            index2label = json.loads(f.read())

        num_classes = len(index2label)
        logging.info(f"Classifying {num_classes} classes")

        train_tfrecord_files = tf.data.Dataset.list_files(training_input_folder + "/*")
        validation_tfrecord_files = tf.data.Dataset.list_files(
            validation_input_folder + "/*"
        )

        train_data = self.build_pipeline(train_tfrecord_files)
        validation_data = self.build_pipeline(validation_tfrecord_files)

        # Model Training Parameters
        learning_rate = settings.LEARNING_RATE
        decay_rate = settings.DECAY_RATE
        epochs = settings.EPOCHS

        logging.info(f"Training for {epochs} epochs")

        # Model parameters
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        es = EarlyStopping(monitor="val_accuracy", verbose=1, patience=3)
        lr = keras.callbacks.LearningRateScheduler(
            lambda epoch: learning_rate / (1 + decay_rate * epoch)
        )

        logging.info(f"Starting MLflow tracking to: {mlflow.get_tracking_uri()}")
        
        with mlflow.start_run():

            # Execute different model approaches and save experiment results:
            model = None

            # In case connection errors that cause resets:
            while not model:
                try:
                    model = self.build_transfer_model(num_classes=num_classes)
                except ConnectionResetError as err:
                    logging.error(str(err))

            print(model.summary())
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=["accuracy", "sparse_categorical_accuracy"],
            )

            # Train model
            training_results = model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=[es, lr],
                verbose=1,
            )

            mlflow.log_param("decay_rate", decay_rate)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("num_classes", num_classes)

            history = training_results.history
            mlflow.log_metric("accuracy", history["accuracy"][-1])
            mlflow.log_metric("val_loss", history["val_loss"][-1])
            mlflow.log_param("epochs", len(history["accuracy"]))

            # Log label mapping for retrieval with model:
            mlflow.log_artifact(label_mapping)

            mlflow.keras.log_model(
                model, "model", registered_model_name="ciri_trashnet_model"
            )

        logging.info("Finished model pipeline")
        os.makedirs(os.path.join(self._data_dir, self._output_folder))

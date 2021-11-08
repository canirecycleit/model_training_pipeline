import luigi

from ciri_pipeline.ml.tasks import TrainModel, TrainTFRecordTask, ValidationTFRecordTask


def main():

    task1 = TrainModel()

    luigi.build([task1], local_scheduler=True)


if __name__ == "__main__":
    main()

import luigi

from ciri_pipeline.ml.tasks import TrainTFRecordTask, ValidationTFRecordTask


def main():

    task1 = TrainTFRecordTask()
    task2 = ValidationTFRecordTask()
    luigi.build([task1, task2], local_scheduler=True)


if __name__ == "__main__":
    main()

import luigi

from ciri_pipeline.ml.tasks import TrainTFRecordTask


def main():

    task = TrainTFRecordTask()
    luigi.build([task], local_scheduler=True)


if __name__ == "__main__":
    main()

import luigi

from ciri_pipeline import tasks


def main():

    task = tasks.DownloadTrainingFiles()
    luigi.build([task], local_scheduler=True)


if __name__ == "__main__":
    main()

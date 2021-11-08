"""Main entry point for Model Training Pipeline."""
import argparse
import logging
import time

import luigi

from ciri_pipeline.ml.tasks import TrainModel


def main(args=None):

    while True:
        luigi.build([TrainModel()], local_scheduler=True)

        # Guard-statment:
        if not args.time or args.time == 0:
            logging.info("Exiting model pipeline.")
            exit(0)

        logging.info(f"Sleeping for {args.time} seconds.")
        time.sleep(args.time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CIRI Model Training Pipeline")

    parser.add_argument(
        "--time",
        default=0,
        type=int,
        help="How long to sleep between pipeline runs.  Zero indicates only run once.",
    )

    main(parser.parse_args())

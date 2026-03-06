import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Runtime script that will rebuild a PYTHON Flavored Job \
                     object, call its function and store the result in the flux \
                     KVS"
    )
    parser.add_argument(
        "--job-id",
        "-id",
        type=str,
        required=True,
        help="The identification of the Job object",
    )
    parser.add_argument(
        "--job-dir",
        "-d",
        type=str,
        required=True,
        help="The directory where the <job_id>.json lives",
    )

    # Parse the arguments
    args = parser.parse_args()
    spec_file = Path(args.)

    # TODO: get the job_spec object by loading the json spec file
    #       then import the function, call it with the args and
    #       store the results in the flux KVS


if __name__ == "__main__":
    main()

import argparse
import pickle
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
        "--jobid",
        "-id",
        type=str,
        required=True,
        help="The identification of the Job object",
    )
    parser.add_argument(
        "--jobdir",
        "-dir",
        type=str,
        required=True,
        help="The directory where the <job_id>.json lives",
    )

    args = parser.parse_args()
    job_id = args.jobid
    spec_file = Path(args.jobdir)

    # TODO: get the job_spec object by loading the json spec file
    #       then import the function, call it with the args and
    #       store the results in the flux KVS
    try:
        with open(spec_file, "r") as file:
            job = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{spec_file}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. Check the file format.")
        exit(1)

    module_name = importlib.import_module(job.func_module)
    func_name = job.func_qualname

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"Error: could not import the module {module_name}")
        print(e)
        exit(1)

    try:
        func = getattr(module, func_name)
    except Exception as e:
        print(f"Error: could not find function '{func_name}'")
        print(e)
        exit(1)

    if job.deps:
        deps_results = []
        for dep in job.deps:
            try:
                with open(f"../{dep}/result.pkl", "rb") as file:
                    result = pickle.load(file)
                    deps_results.append(result)
            except FileNotFoundError:
                print(f"Error: The file '{spec_file}' was not found.")
                exit(1)
            except pickle.UnpicklingError as e:
                print(f"Error: couldn't load pickled object")
                print(e)
                exit(1)

    try:
        result = func(*job.args, **job.kwargs)
    except Exception as e:
        print(f"Error: calling funcition '{func}' failed")
        print(e)
        exit(1)

    result_file = Path(f"{spec_file.parent}/result.pkl")
    with open(result_file, "wb") as file:
        # Use pickle.dump() to serialize the object and write it to the file
        pickle.dump(result, file)

    print(f"result successfully dumped to {result_file}")


if __name__ == "__main__":
    main()

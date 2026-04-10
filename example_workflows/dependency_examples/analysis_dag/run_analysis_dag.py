# run_analysis_dag.py
from analysis_dag_workflow import pipe, build_workflow


def main() -> None:
    build_workflow(num_cases=8)
    pipe.submit(
        buffer_time=0.0,
        write_restart_freq=None,
    )


if __name__ == "__main__":
    main()

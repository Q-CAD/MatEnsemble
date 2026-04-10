# run_python_chore_workflow.py
from python_chore_workflow import pipe, build_workflow


def main() -> None:
    build_workflow()
    pipe.submit(
        buffer_time=0.0,
        write_restart_freq=None,
    )


if __name__ == "__main__":
    main()

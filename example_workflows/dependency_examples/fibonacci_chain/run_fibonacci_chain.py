# run_fibonacci_chain.py
from fibonacci_chain_workflow import pipe, build_workflow


def main() -> None:
    build_workflow(n=12)
    pipe.submit(
        buffer_time=0.0,
        write_restart_freq=None,
    )


if __name__ == "__main__":
    main()

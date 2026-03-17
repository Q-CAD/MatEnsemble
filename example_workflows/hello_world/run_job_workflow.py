from job_workflow import pipe, run_mpi_hello


def main():
    for i in range(1, 11):
        run_mpi_hello(i)

    pipe.submit()


if __name__ == "__main__":
    main()

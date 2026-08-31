from matensemble.pipeline import Pipeline

pipe = Pipeline()

for _ in range(10):
    pipe.exec(command=["echo", "Hello, World!"], num_tasks=10)

pipe.submit()

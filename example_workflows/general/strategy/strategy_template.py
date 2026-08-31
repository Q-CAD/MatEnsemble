from matensemble.pipeline import Pipeline
from matensemble.model import Resources
from matensemble.chore import ChoreSpec

pipe = Pipeline()

md_resources = dict(num_tasks=128, cores_per_task=1, gpus_per_task=4, mpi=True)
analysis_resources = dict(num_tasks=1, cores_per_task=8)


@pipe.chore(name="simulate", **md_resources)
def simulate(candidate):
    # Run LAMMPS, DFT, phase-field, or another science application here.
    return {"trajectory": "traj.dump", "candidate": candidate}


@pipe.chore(name="score", **analysis_resources)
def score(simulation):
    # Analyze the completed simulation and propose the next high-value sample.
    return {
        "uncertainty": 0.18,
        "next_candidate": {"temperature": 1750, "composition": "SiO2"},
    }


@pipe.strategy(bolo_list=["score"], **analysis_resources)
def adapt(report):
    if report["uncertainty"] < 0.05:
        return None

    return ChoreSpec(
        args=(report["next_candidate"],),
        kwargs={},
        resources=Resources(**md_resources),
        qualname="simulate",
    )


seed = {"temperature": 1600, "composition": "SiO2"}
trajectory = simulate(seed)

# OutputReference creates the simulate -> score DAG edge.
score(trajectory)

future = pipe.submit(log_delay=10)
results = future.result()

$SCRATCH/matensemble_campaigns/
|- campaign_1
|  |- workflow_1
|  |  |- status.json
|  |- ...
|  |- workflow_N
|  |  |- status.json
|- . . .
|  |- workflow_1
|  |  |- status.json
|  |- ...
|  |- workflow_N
|  |  |- status.json
|- campaign_N
|  |- workflow_1
|  |  |- status.json
|  |- ...
|  |- workflow_N
|  |  |- status.json


## Status JSON file

```json
{
    "nodes": 19,
    "cores_per_node": 56,
    "gpus_per_node": 8,
    "pending": 0,
    "running": 0,
    "completed": 0,
    "failed": 5,
    "free_cores": 896,
    "free_gpus": 128,
    "state": "running"
}
```

We want a dashboard where users will be able to visually monitor their scientific workflows on HPC systems.
The dashboard needs a python server that will give users the ability to see all of the campaigns.
There should be a side panel on the left that lists out all of the campaigns.
If you click on a campaign it will be like entering a folder on an explorer and list all of the workflows.
If a workflow is running there should be an indicator where there is a color for running, color for successful completion and a color for failure.
When the user clicks on a specific workflow it will pull up a line chart showing the change in


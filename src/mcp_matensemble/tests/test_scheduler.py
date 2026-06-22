from __future__ import annotations

from pathlib import Path

import pytest

from mcp_matensemble.scheduler import (
    inspect_outputs,
    plan_job_submission,
    submit_job,
    tail_log,
)


def test_plan_submission_validates_slurm_inside_workspace(tmp_path):
    script = tmp_path / "run_batch.slurm"
    script.write_text(
        "#!/usr/bin/env bash\n#SBATCH -N 1\nmatensemble run workflow.py\n",
        encoding="utf-8",
    )

    plan = plan_job_submission("run_batch.slurm", cwd=tmp_path)

    assert plan["command"] == ["sbatch", str(script)]
    assert plan["execute_default"] is False


def test_submit_job_defaults_to_dry_run(tmp_path):
    script = tmp_path / "run_batch.slurm"
    script.write_text(
        "#!/usr/bin/env bash\n#SBATCH -N 1\nmatensemble run workflow.py\n",
        encoding="utf-8",
    )

    result = submit_job("run_batch.slurm", cwd=tmp_path)

    assert result["executed"] is False
    assert result["command"][0] == "sbatch"


def test_submission_rejects_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside.slurm"
    outside.write_text("#SBATCH -N 1\nmatensemble run workflow.py\n", encoding="utf-8")

    with pytest.raises(ValueError, match="workspace"):
        plan_job_submission(str(outside), cwd=tmp_path)


def test_inspect_outputs_and_tail_log(tmp_path):
    workflow = tmp_path / "matensemble_workflow-20260101_000000"
    chore = workflow / "out" / "chore-test-0001"
    chore.mkdir(parents=True)
    (workflow / "status.json").write_text('{"completed": 1, "failed": 0}', encoding="utf-8")
    (workflow / "matensemble_workflow.log").write_text("a\nb\nc\n", encoding="utf-8")
    (chore / "stdout").write_text("ok\n", encoding="utf-8")

    inspected = inspect_outputs(str(workflow), cwd=tmp_path)
    tailed = tail_log(str(workflow), lines=2, cwd=tmp_path)

    assert inspected["status"]["completed"] == 1
    assert inspected["chore_count"] == 1
    assert tailed["text"] == "b\nc"

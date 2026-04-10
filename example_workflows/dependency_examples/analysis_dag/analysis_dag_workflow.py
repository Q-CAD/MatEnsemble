# analysis_dag_workflow.py
from dataclasses import dataclass

from matensemble.pipeline import Pipeline

N_CASES = 8

pipe = Pipeline()


@dataclass
class MetricsBundle:
    total: int
    maximum: int
    even_count: int


@pipe.chore(name="make-case")
def make_case(case_id: int) -> list[int]:
    # deterministic toy data
    data = [case_id * 10 + i for i in range(1, 9)]
    print(f"case {case_id}: generated {data}")
    return data


@pipe.chore(name="sum")
def compute_sum(data: list[int]) -> int:
    return sum(data)


@pipe.chore(name="max")
def compute_max(data: list[int]) -> int:
    return max(data)


@pipe.chore(name="even-count")
def compute_even_count(data: list[int]) -> int:
    return sum(1 for x in data if x % 2 == 0)


@pipe.chore(name="bundle")
def bundle_metrics(total: int, maximum: int, even_count: int) -> MetricsBundle:
    return MetricsBundle(
        total=total,
        maximum=maximum,
        even_count=even_count,
    )


@pipe.chore(name="classify")
def classify_case(data: list[int], metrics: MetricsBundle) -> dict:
    label = "large" if metrics.total > 100 else "small"
    return {
        "length": len(data),
        "label": label,
        "total": metrics.total,
        "maximum": metrics.maximum,
        "even_count": metrics.even_count,
    }


@pipe.chore(name="case-summary")
def summarize_case(case_id: int, payload: dict) -> dict:
    summary = {"case_id": case_id, **payload}
    print(f"summary: {summary}")
    return summary


@pipe.chore(name="global-report")
def global_report(case_summaries: list[dict]) -> dict:
    ordered = sorted(case_summaries, key=lambda x: x["case_id"])
    large_cases = [c["case_id"] for c in ordered if c["label"] == "large"]

    report = {
        "num_cases": len(ordered),
        "large_cases": large_cases,
        "grand_total": sum(c["total"] for c in ordered),
        "max_of_maxes": max(c["maximum"] for c in ordered),
    }
    print(report)
    return report


def build_workflow(num_cases: int = N_CASES):
    summaries = []

    for case_id in range(num_cases):
        raw = make_case(case_id)

        total = compute_sum(raw)
        maximum = compute_max(raw)
        even_count = compute_even_count(raw)

        metrics = bundle_metrics(total, maximum, even_count)
        classified = classify_case(raw, metrics)

        summaries.append(summarize_case(case_id, classified))

    global_report(summaries)

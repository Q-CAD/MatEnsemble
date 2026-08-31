I am on the Frontier super computer and want to run some calculations.

Create an workflow with <WORKFLOW_MANAGER> that reproduces this adaptive scientific campaign.

The workflow should model an asynchronous autonomous optimization experiment over the 2D Rastrigin
function on [-5.12, 5.12]^2. Each task evaluates one candidate point (x, y), sleeps for a
heterogeneous simulated wall time, computes a noisy Rastrigin objective, and returns candidate_id,
x, y, grid arm, expected_wall_s, actual_wall_s, objective, reward = -objective, and source.

Divide the domain into an ARM_GRID x ARM_GRID grid of arms. Start with an initial candidate batch
that gives broad coverage, then use Thompson sampling over per-arm reward estimates to decide
future candidates. As task results complete, update each arm’s running mean/count, track the best
observed candidate, and spawn one new evaluation when autonomous mode is enabled and the campaign
has not exceeded N_CANDIDATES or TARGET_WALL_S.

Candidate generation should mix:
- global random exploration,
- local sampling around the current best candidate,
- Thompson-sampled arm exploration.

The workflow should emphasize asynchronous scheduling behavior: tasks have deliberately varied
wall times, and the strategy should prioritize ready/spawned work using a reward-per-cost score
when the workflow manager supports priorities or queue reordering. Produce JSON/CSV summaries
containing all completed evaluations, best result, wall-time statistics, generation events, and
strategy trace events.

Can you produce the code for the workflow and give me instructions on:
- how many resources to allocate
- how to setup the environment
- how to run the jobs
- how to interpret the results

I also want information about how <WORKFLOW_MANAGER> ran the workflow
- how many tasks were scheduled
- how long it took to run
- how well resources were utilized

if that is possible.

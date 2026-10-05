# P-I pre-flight: SYNTHETIC outputs

These are not quadrotor results. They come from `experiments/p1_quadrotor/preflight.py`
at commit 6de7d31, on a synthetic linear plant: ẋ = A(x − x0) + B(u − u0), with A and B
drawn at random from seed 0.

- **Configuration:** the real configuration, with the pre-flight's budget cuts
  (`preflight_override.yaml`).
- **What they show:** that every stage runs and that every quantity `prereg/p1.md`
  promises is produced. Their verdicts say nothing about the quadrotor.
- **Committed here:** the completeness check (`preflight_check.json`), the rules applied
  to the synthetic outputs (`predictions.json`) and the stopping-rule exercise
  (`preflight_stops.json`).
- **Not committed:** the full tables and checkpoints, which are reproducible from the
  script.

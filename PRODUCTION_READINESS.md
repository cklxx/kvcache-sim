# Production Readiness Guide

This simulator is suitable for production-adjacent offline evaluation only when
each run is tied to a workload trace, calibration profile, and machine-readable
report. Treat synthetic runs as directional tests, not capacity guarantees.

## Supported Use

- Compare KV-cache policies, EIC capacity, PD ratios, and transfer strategies.
- Reproduce fixed-seed experiments in CI and review metric drift.
- Screen architecture choices before running expensive serving benchmarks.

## Not Supported

- Online scheduling or control-plane decisions.
- Hardware purchase decisions from synthetic traces alone.
- Absolute TTFT, TPOT, or tail-latency claims without external calibration.

## Required Inputs For Production Evaluation

- A workload trace representative of production traffic.
- Prefix identifiers such as `hash_ids`; token-count-only traces are lower
  confidence because prefix reuse is synthesized.
- A calibration profile derived from serving or microbenchmark data.
- A saved JSON report with the config snapshot and run metadata.

## Minimum Acceptance Gates

- Trace validation has no high-risk warnings.
- Active rack/GPU diagnostics match the intended scale of the experiment.
- Golden tests pass for fixed-seed smoke runs.
- The calibration profile documents source benchmark dates and model/runtime
  versions.
- Result changes are reviewed as relative deltas unless calibrated absolute
  error has been measured.

## Recommended Workflow

1. Run a smoke preset after code changes.
2. Validate the production trace and inspect warning fields.
3. Run the target mode with a calibration profile and JSON report enabled.
4. Compare report metrics against the last accepted baseline.
5. Promote only the conclusions that are supported by calibrated inputs.

## Confidence Levels

- **Low:** synthetic trace, no calibration, no diagnostics review.
- **Medium:** real token-count trace, calibration profile, fixed-seed report.
- **High:** real prefix-backed trace, calibrated compute/network parameters,
  diagnostics reviewed, and benchmark cross-checks within known error bounds.

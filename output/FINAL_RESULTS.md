# CAAI complete simulation audit

Generated: 2026-11-17 14:41:42
Quick mode: `True`
All steps OK: `True`

## Generated artifacts

- `theory_tests`: **ok** (3.1s)
- `algorithm4_tests`: **ok** (4.6s)
- `hardcode_scan`: **ok** (0.0s)

## Tables

The authoritative tables are in `output/tables/`; JSON manifests and
stdout logs are in `output/logs/`.  Key files:

- `od_e1_main_output_feedback.md`
- `od_e2_output_feedback_shift.md`
- `od_e3_output_feedback_ablation.md`
- `od_e4_algorithm4_end_to_end.md`
- `od_e5_algorithm4_rates.md`
- `od_e5b_adversary_block_rates.md`
- `od_e9_sdpi_diagnostics.md`
- `od_e10_identifiability_negative.md`
- `od_e6_hard_wasserstein_own_adversary.md`
- `od_e7_g_w2_membership.md`
- `od_e8_nonlinear_contact_saturation.md`
- `od_e9_tail_only.md`
- `e18_covariance_only.md`, `e19_heavy_tail.md`, `e20_g_sample_size.md`, `e21_saturation.md`

## Scope guards

- The output-feedback rows using a Kalman filter are unknown-disturbance-history
  simulations.  They are not claimed to reproduce full-state performance.
- Algorithm 4 requires an independent disturbance-channel probe for separate
  identification of E and N; without it the code returns an explicit
  non-identifiability result.
- Hard-Wasserstein results are finite-dimensional, sampled trajectory
  certificates.  They are not a proof of infinite-horizon global Bellman
  optimality.
- Nonlinear/contact/tail experiments are reduced stress models, not
  high-fidelity hardware validations.

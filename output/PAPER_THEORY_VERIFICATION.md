# Paper theorem verification

All numerical checks below use the two application models in the manuscript.  They verify algebraic identities, stability certificates, and finite-sample rates; they do not turn a finite sample into a global infinite-horizon proof.

## humanoid

| Check | Result | Value |
|---|---:|---:|
| assumptions: stabilizability rank | PASS | 6.0 |
| assumptions: observability-window rank | PASS | 6.0 |
| wasserstein_bellman: affine Bellman residual | PASS | 4.3655745685100555e-11 |
| wasserstein_bellman: input FOC residual | PASS | 1.4033812996793565e-14 |
| wasserstein_bellman: adversary FOC residual | PASS | 1.5862718200929604e-14 |
| wasserstein_bellman: closed-loop G Schur | PASS | 0.9853625286681004 |
| positive_P: P symmetric | PASS | 3.014577520672848e-14 |
| positive_P: P Lyapunov residual | PASS | 9.135190304375618e-13 |
| positive_P: Phi_P positive | PASS | 26.75861961733888 |
| positive_P: P positive | PASS | 31.78477877234651 |
| positive_P: Hww negative | PASS | -4.922811353520771 |
| positive_P: fixed lambda above Hww threshold | PASS | 4.922811353520771 |
| reconstruction_bound: exact model-based reconstruction | PASS | 2.928818591588429e-14 |
| reconstruction_bound: data-driven N^-1/2 trends | PASS | -0.5025961862400629 |
| reconstruction_bound: data-driven slope subspace | PASS | -0.613141313460259 |
| reconstruction_bound: data-driven slope reconstruction | PASS | -0.46799443752412223 |
| reconstruction_bound: data-driven slope input_identity | PASS | -0.46815725207788844 |
| reconstruction_bound: data-driven slope disturbance_identity | PASS | -0.5025961862400629 |
| scaling_lmi_sdpi: X LMI first block positive | PASS | 1.180893760101435 |
| scaling_lmi_sdpi: X LMI second block positive | PASS | 0.9999999999999956 |
| scaling_lmi_sdpi: direct X-SDP feasible | PASS | 0.01849240769038964 |
| scaling_lmi_sdpi: pure Lyapunov SDP feasible | PASS | 0.0 |
| scaling_lmi_sdpi: initial destabilizing policy | PASS | 2.0305079576971075 |
| scaling_lmi_sdpi: scaled initializer returns stable original policy | PASS | 0.9998404033329197 |
| scaling_lmi_sdpi: SDPI every logged iterate stable | PASS | 0.9998404033329197 |
| scaling_lmi_sdpi: SDPI final fixed-point residual small | PASS | 7.822265550419951e-13 |
| scaling_lmi_sdpi: scaling initializer source | PASS | scaled_candidate_verified_on_original |
| data_driven_consistency: data-driven slope diagnostics finite | PASS | 0.6858891082708881 |
| data_driven_consistency: diagnostic slope eA | PASS | -0.6858891082708881 |
| data_driven_consistency: diagnostic slope eB | PASS | -0.43389831817363733 |
| data_driven_consistency: diagnostic slope eE | PASS | -0.40872231100363743 |
| data_driven_consistency: diagnostic slope eP | PASS | -0.5219275668274423 |
| data_driven_consistency: diagnostic slope eK | PASS | -0.6539251344319754 |
| data_driven_consistency: diagnostic slope ep | PASS | -0.3348566326959886 |
| data_driven_consistency: scope | PASS | consistency theorem; rates reported as diagnostics only |
| rigorous_rate_validation: local-rate diagnostics finite | PASS | 0.5422371253424946 |
| rigorous_rate_validation: local-rate diagnostics consistent with N^-1/2 regime | PASS | -0.5422371253424946 |
| rigorous_rate_validation: local-rate slope eP | PASS | -0.5037378994395614 |
| rigorous_rate_validation: local-rate slope eK | PASS | -0.5422371253424946 |
| rigorous_rate_validation: local-rate slope ep | PASS | -0.5342402022730101 |

## microgrid

| Check | Result | Value |
|---|---:|---:|
| assumptions: stabilizability rank | PASS | 4.0 |
| assumptions: observability-window rank | PASS | 4.0 |
| wasserstein_bellman: affine Bellman residual | PASS | 8.731149137020111e-11 |
| wasserstein_bellman: input FOC residual | PASS | 1.3756308425898965e-13 |
| wasserstein_bellman: adversary FOC residual | PASS | 4.275768827980918e-13 |
| wasserstein_bellman: closed-loop G Schur | PASS | 0.802116533354402 |
| positive_P: P symmetric | PASS | 1.1546020556458904e-13 |
| positive_P: P Lyapunov residual | PASS | 1.0191534130100254e-12 |
| positive_P: Phi_P positive | PASS | 1.5666237117858566 |
| positive_P: P positive | PASS | 1.756027151076012 |
| positive_P: Hww negative | PASS | -2.9794103370198046 |
| positive_P: fixed lambda above Hww threshold | PASS | 2.9794103370198046 |
| reconstruction_bound: exact model-based reconstruction | PASS | 6.650226650194185e-15 |
| reconstruction_bound: data-driven N^-1/2 trends | PASS | -0.5063615761401129 |
| reconstruction_bound: data-driven slope subspace | PASS | -0.895218653058855 |
| reconstruction_bound: data-driven slope reconstruction | PASS | -0.4676192028072243 |
| reconstruction_bound: data-driven slope input_identity | PASS | -0.48434402594839343 |
| reconstruction_bound: data-driven slope disturbance_identity | PASS | -0.5063615761401129 |
| scaling_lmi_sdpi: X LMI first block positive | PASS | 1.1269950974570746 |
| scaling_lmi_sdpi: X LMI second block positive | PASS | 0.9999999999999614 |
| scaling_lmi_sdpi: direct X-SDP feasible | PASS | 4.902738229900583e-05 |
| scaling_lmi_sdpi: pure Lyapunov SDP feasible | PASS | 0.0 |
| scaling_lmi_sdpi: initial destabilizing policy | PASS | 2.0412317606242083 |
| scaling_lmi_sdpi: scaled initializer returns stable original policy | PASS | 0.9861184921041231 |
| scaling_lmi_sdpi: SDPI every logged iterate stable | PASS | 0.9861184921041231 |
| scaling_lmi_sdpi: SDPI final fixed-point residual small | PASS | 3.312265435883943e-12 |
| scaling_lmi_sdpi: scaling initializer source | PASS | scaled_candidate_verified_on_original |
| data_driven_consistency: data-driven slope diagnostics finite | PASS | 0.6568647927742804 |
| data_driven_consistency: diagnostic slope eA | PASS | -0.41126880374784397 |
| data_driven_consistency: diagnostic slope eB | PASS | -0.4053696805525941 |
| data_driven_consistency: diagnostic slope eE | PASS | -0.5093402863929982 |
| data_driven_consistency: diagnostic slope eP | PASS | -0.6309681342909081 |
| data_driven_consistency: diagnostic slope eK | PASS | -0.6568647927742804 |
| data_driven_consistency: diagnostic slope ep | PASS | -0.5888990547576818 |
| data_driven_consistency: scope | PASS | consistency theorem; rates reported as diagnostics only |
| rigorous_rate_validation: local-rate diagnostics finite | PASS | 0.5069916231801176 |
| rigorous_rate_validation: local-rate diagnostics consistent with N^-1/2 regime | PASS | -0.5069916231801176 |
| rigorous_rate_validation: local-rate slope eP | PASS | -0.5026909967480852 |
| rigorous_rate_validation: local-rate slope eK | PASS | -0.5069916231801176 |
| rigorous_rate_validation: local-rate slope ep | PASS | -0.4733325748874211 |


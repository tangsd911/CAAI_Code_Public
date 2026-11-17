| Scenario | Method | Seeds | E non-identifiable | rank full | E first-stage F | Conclusion |
|---|---|---|---|---|---|---|
| humanoid | OLS | 5 | 5/5 | 1.00 | — | confounded |
| humanoid | IV/2SLS, lagged only | 5 | 5/5 | 1.00 | 0.773 | not identifiable without exogenous disturbance excitation |
| microgrid | OLS | 5 | 5/5 | 1.00 | — | confounded |
| microgrid | IV/2SLS, lagged only | 5 | 5/5 | 1.00 | 0.736 | not identifiable without exogenous disturbance excitation |

Negative control with q=0. The matrix rank can be full while E is still not identified: d=Nx+z+xi is feedback-correlated with x, and xi enters the transition error. This result is retained to prevent a false claim that generic closed-loop data identify the pair (E,N). Exogenous disturbance-channel excitation (or prior knowledge) is required.

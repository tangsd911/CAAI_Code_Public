| Scenario | Distribution | Controller | Seeds | Valid seeds | Diverged seeds | MAE | MAE std | Discounted cost | Cost std | Tail event fraction | Saturation fraction | Contact fraction | Frequency violation fraction | Max abs state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| humanoid | gaussian | DRC | 20 | 20 | 0 | 0.316901 | 0.005808 | 191.357729 | 10.450693 | 0.02930 | 0.8898 | 0.0476 | — | 0.962 |
| humanoid | gaussian | LQR | 20 | 20 | 0 | 0.416696 | 0.006686 | 436.462854 | 18.258149 | 0.02930 | 0.8859 | 0.0551 | — | 0.929 |
| humanoid | student_t_3 | DRC | 20 | 20 | 0 | 0.313031 | 0.006608 | 186.084143 | 9.317466 | 0.05230 | 0.9064 | 0.0488 | — | 1.008 |
| humanoid | student_t_3 | LQR | 20 | 20 | 0 | 0.412126 | 0.006545 | 424.250360 | 16.050052 | 0.05230 | 0.8913 | 0.0552 | — | 0.983 |
| humanoid | student_t_5 | DRC | 20 | 20 | 0 | 0.313362 | 0.006332 | 185.136242 | 9.166761 | 0.05205 | 0.8986 | 0.0481 | — | 0.995 |
| humanoid | student_t_5 | LQR | 20 | 20 | 0 | 0.412207 | 0.006692 | 422.585395 | 16.228847 | 0.05205 | 0.8955 | 0.0541 | — | 0.967 |
| humanoid | student_t_10 | DRC | 20 | 20 | 0 | 0.310847 | 0.005351 | 181.843323 | 8.637228 | 0.04200 | 0.8899 | 0.0502 | — | 0.986 |
| humanoid | student_t_10 | LQR | 20 | 20 | 0 | 0.409938 | 0.006672 | 417.544043 | 16.261482 | 0.04200 | 0.8874 | 0.0568 | — | 0.926 |
| humanoid | tail_contamination | DRC | 20 | 20 | 0 | 0.314135 | 0.005476 | 188.901615 | 7.061993 | 0.05097 | 0.8901 | 0.0501 | — | 1.025 |
| humanoid | tail_contamination | LQR | 20 | 20 | 0 | 0.412970 | 0.004844 | 426.464135 | 11.949495 | 0.05097 | 0.8871 | 0.0564 | — | 0.955 |
| microgrid | gaussian | DRC | 20 | 20 | 0 | 0.003176 | 0.000037 | 0.542364 | 0.005623 | 0.06112 | 0.0100 | — | 0.0000 | 0.478 |
| microgrid | gaussian | LQR | 20 | 20 | 0 | 0.005092 | 0.000090 | 0.791836 | 0.024295 | 0.06112 | 0.0005 | — | 0.0000 | 0.478 |
| microgrid | student_t_3 | DRC | 20 | 20 | 0 | 0.003057 | 0.000065 | 0.552793 | 0.087288 | 0.07457 | 0.0329 | — | 0.0000 | 0.486 |
| microgrid | student_t_3 | LQR | 20 | 20 | 0 | 0.005120 | 0.000049 | 0.822830 | 0.076599 | 0.07457 | 0.0046 | — | 0.0000 | 0.484 |
| microgrid | student_t_5 | DRC | 20 | 20 | 0 | 0.003142 | 0.000027 | 0.540278 | 0.006841 | 0.08358 | 0.0231 | — | 0.0000 | 0.479 |
| microgrid | student_t_5 | LQR | 20 | 20 | 0 | 0.005130 | 0.000039 | 0.801197 | 0.011835 | 0.08358 | 0.0012 | — | 0.0000 | 0.480 |
| microgrid | student_t_10 | DRC | 20 | 20 | 0 | 0.003160 | 0.000033 | 0.540076 | 0.005359 | 0.07578 | 0.0154 | — | 0.0000 | 0.479 |
| microgrid | student_t_10 | LQR | 20 | 20 | 0 | 0.005112 | 0.000070 | 0.795830 | 0.020127 | 0.07578 | 0.0005 | — | 0.0000 | 0.479 |
| microgrid | tail_contamination | DRC | 20 | 20 | 0 | 0.002799 | 0.000047 | 0.518256 | 0.017003 | 0.04979 | 0.0523 | — | 0.0000 | 0.478 |
| microgrid | tail_contamination | LQR | 20 | 20 | 0 | 0.005082 | 0.000080 | 0.799784 | 0.028446 | 0.04979 | 0.0130 | — | 0.0000 | 0.479 |

**Tail-only comparison.** `gaussian` is the nominal Gaussian law.
`student_t_3/5/10` use covariance-matched Student-t laws. `tail_contamination`
is a covariance-matched two-scale Gaussian mixture with epsilon=0.05 and tail
scale=4: with probability 0.95 a Gaussian with reduced scale is used, and with
probability 0.05 a Gaussian with scale 4 is used; the marginal covariance is
exactly the nominal one. It therefore isolates increased tail mass, but it does
not preserve the original central variance. `Tail event fraction` is the
empirical fraction of the mixture's tail component for the contamination law,
and the fraction with standardized Euclidean norm greater than 3 for Gaussian
and Student-t laws. All rows use the nominal nonlinear stress parameters, 20
seeds where available, and the same unknown-history Kalman observer as the
main nonlinear experiment. This is a stress comparison, not a proof that a
fixed Wasserstein radius contains these alternative laws.

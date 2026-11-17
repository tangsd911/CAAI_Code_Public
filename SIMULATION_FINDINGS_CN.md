# CAAI 修改稿仿真补充与核查结论

## 1. 输出反馈闭环

采用 `y=Cx`、未知扰动历史下的稳态 Kalman 观测器和 `u=K\hat x+p` 真实闭环，20 个相同种子与全状态结果对照。所有 OFB 行均未发散。

| 场景 | 控制器 | 全状态 MAE/代价 | 未知历史 OFB MAE/代价 | 重建 RMSE |
|---|---|---:|---:|---:|
| Humanoid | DRC | 0.045482 / 41.161535 | 0.121189 / 58.581330 | 2.122e-01 |
| Humanoid | LQR | 0.290970 / 370.762661 | 0.314970 / 388.859961 | 2.122e-01 |
| Microgrid | DRC | 0.002810 / 0.488517 | 0.003176 / 0.543027 | 1.806e-03 |
| Microgrid | LQR | 0.004939 / 0.745440 | 0.005093 / 0.791925 | 1.806e-03 |

结论：未知扰动历史下 OFB 可行且稳定，但 Humanoid 上不能声称恢复全状态性能。Humanoid 的 OFB 消融中，仿射 nominal LQR 的 MAE 为 0.118065，略优于完整 DRC 的 0.121189；因此不能声称 DRC 在 Humanoid 的 OFB 上全面最优。Microgrid 上完整 DRC 的代价 0.543027 优于无偏置 DRC 0.641337 和 LQR 0.791925。

## 2. Algorithm 4、IV 与 E/N 辨识

使用独立扰动通道探针 `q` 作为排除性工具变量。无 `q` 时，即使回归矩阵满秩，`E` 与 `A+EN` 混杂，代码明确返回不可辨识结果，不再使用伪识别模型。

| 场景 | 方法 | eA | eB | eE | 策略 eN | 最小真实 rho | 稳定接受 |
|---|---|---:|---:|---:|---:|---:|---|
| Humanoid | IV/2SLS | 6.315e-04 | 1.079e-04 | 3.851e-04 | 1.323e-02 | 0.985659 | 5/5 |
| Microgrid | IV/2SLS | 1.342e-03 | 1.854e-05 | 6.821e-03 | 1.361e-01 | 0.802267 | 5/5 |

固定策略 `N=4000–32000` 的 IV 斜率：Humanoid `eA,eB,eE,eK,eN = -0.644,-0.495,-0.511,-0.538,-0.501`；Microgrid `-0.507,-0.535,-0.490,-0.498,-0.522`。这支持局部单步 `O_p(N^{-1/2})` 结论，不支持外层全局收敛结论。Microgrid 的 `N` 辨识仍有弱方向，不能声称所有情形精确达到理论阶。

## 3. SDPI 代码核查

- 缩放系统求得的 `K,p,N,z` 回代原系统：映射为同坐标赋值，并显式验证 `rho(A+BK+EN)<1`。
- `H_uu>0`、`H_mm<0` 和两个 Schur 补均检查条件数和特征值；奇异/边界时停止，不使用伪逆。
- 停止条件同时检查 `K,p,N,z`。
- Humanoid 固定点残差：`P=4.307e-13`、`chi=4.965e-16`、`psi=0`。
- Microgrid 固定点残差：`P=2.575e-12`、`chi=8.909e-16`、`psi=1.110e-16`。
- 人为设定 `lambda=0.01` 的边界测试均在 `schur_complement_boundary` 停止。

这些结果只支持“实现满足停止条件的稳定固定点”，不支持无条件的单调性或全局收敛。

## 4. 硬 Wasserstein 约束与各自最坏对抗

- `zeta=0.005,0.01,0.02,0.05,0.1,0.2` 共 24 组 DRC/LQR 半径扫描全部 ball-feasible。
- 每组均显式构造 atomwise worst-case distribution 并检查 `W2_max <= zeta`。
- DRC 与 LQR 使用各自的对抗策略，LQR 的自身 FOC 残差 Humanoid `1.042e-14`、Microgrid `7.944e-15`。
- `g=1,2,5,10,20`，每个控制器 10 次 support draw，共 200/200 条 `W2_max <= 0.1`。
- 边界活跃时互补松弛量约为数值零；selected lambda 是有限时域内部可行候选，不能用其非零互补量判断 KKT 失败。
- Microgrid DRC 的部分 support draw 没有活跃边界，因此不能声称所有 g>1 情形都满足精确边界互补条件。
- 所有硬球结论均为有限时域采样轨迹证书，不是无限时域全局 Bellman 最优证明。

## 5. 非线性、接触、饱和、范围与尾部

所有正式非线性/Tail 表均为 20 个种子，0 个 divergence。

| 场景 | Stress | DRC MAE/代价 | LQR MAE/代价 |
|---|---|---:|---:|
| Humanoid | nominal | 0.316901 / 191.357729 | 0.416696 / 436.462854 |
| Humanoid | moderate | 0.487660 / 820.311316 | 0.571464 / 1149.860971 |
| Humanoid | extreme | 0.882560 / 3011.977274 | 0.919440 / 3264.384978 |
| Microgrid | nominal | 0.003176 / 0.542364 | 0.005092 / 0.791836 |
| Microgrid | moderate | 0.004461 / 1.085545 | 0.007925 / 1.875447 |
| Microgrid | extreme | 0.007529 / 2.219027 | 0.011511 / 3.773923 |

- Humanoid extreme 的接触比例约 0.9966/0.9969、饱和比例约 0.9996/0.9991，只能说明有限时域数值稳定，不能说明硬件安全。
- Microgrid 三档频率越界比例均为 0，最大越界为 0。
- Gaussian、Student-t `df=3,5,10` 和 covariance-matched tail contamination 下，DRC 的 cost 均低于 LQR。该结果不是硬 Wasserstein 球成员证书。

## 6. 环境与复现

```bash
cd CAAI_Code_Final
python3 -m pip install -r requirements.txt
MPLCONFIGDIR=.mplconfig PYTHONDONTWRITEBYTECODE=1 \
  python3 run_complete_suite.py
```

测试命令：

```bash
python3 theory_tests.py
python3 algorithm4_tests.py
python3 scan_hardcoded.py
python3 validate_outputs.py
```

完整结果、日志和逐候选审计文件位于 `output/`。所有结果值由脚本计算，硬编码结果扫描为 0。

## 7. 理论级反向核查

新增 `paper_theory_verification.py`，直接对论文两个应用模型逐条核验理论：

- 可控性秩、可观测窗口秩；
- 仿射最坏情况 Bellman 等式与 u、ω 两个一阶条件；
- `P` 的 Lyapunov 残差、`Phi_P>0`、`P>0`、`H_ww<0`；
- 输出窗口精确重构；
- 数据驱动重构统计误差的 `N^{-1/2}` 趋势；
- 缩放回映射、`X`-LMI 稳定性判据、SDPI 每次迭代稳定性；
- 辨识模型的 `eP、eK` 等策略/值函数误差速率。

Humanoid 的最大 Bellman 残差约为 `8.8e-11` 以下级别，Microgrid 也低于 `1e-10`；两个模型的 `P` 残差均低于 `1.1e-12`。数据驱动重构的输入补偿和状态重构误差斜率约为 `-0.47` 至 `-0.51`，与定理中的 `O_p(N^{-1/2})` 阶一致；重构审计还显式计入 `||T_d|| ||d_window-vbar_window||`，Humanoid 与 Microgrid 的 held-out 点均为 `48/48` 满足经验界。完整结果见 `output/PAPER_THEORY_VERIFICATION.md`。

反向核查发现并修正了两处表述/实现不一致：

1. 原文数据驱动重构写作 `M_y O_l = I`，但一般只能得到 `M_y O_l = T^{-1}`；正确理论应在等价状态坐标 `x^T=T^{-1}x` 下表述。已在论文理论部分改为带相似变换的版本。
2. 原文 `T_d` “由残差协方差估计”不可构造，协方差本身不能识别输入到状态的映射；已改为对残差输出窗口与实测扰动窗口做回归 `E_l=tilde Y D_w^T(D_wD_w^T)^\dagger`、`T_d=M_y E_l`。同时在 `sdpi.py` 中删除了固定点诊断里残留的错误 support-spread 常数项。

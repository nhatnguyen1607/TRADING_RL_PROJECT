# RL Trading Improvement Plan

## Research Experiment Log - Current Multi-Asset Pipeline

Test protocol:

- Universe: `SPY`, `SH`, `TLT`.
- Test period represented by the current split: approximately July 2021 through December 2022.
- Buy & Hold SPY final net worth: `$8,714.76`.
- Primary comparison metrics: final net worth, Sharpe, Sortino, max drawdown, turnover, action/template behavior.
- Rule for future work: record every completed experimental run in this section before starting the next algorithmic change.

### Archived Positive Baseline Before Current Research Extensions

- Recovery source: `planning.md` in Git commit `8a2018b`.
- Raw CSV/report files had previously been overwritten and cannot be recovered byte-for-byte.
- Runnable source snapshot restored at: `archived_runs/positive_baseline_8a2018b`.
- Permanent metric reference: `results/baseline_positive_reference/README.md`.
- DQN final net worth: `$10,526.35`; Sharpe `0.4856`; Sortino `0.7717`; max drawdown `-5.50%`; meaningful trades `276`.
- Actor-Critic final net worth: `$10,780.55`; Sharpe `0.5523`; Sortino `0.8930`; max drawdown `-9.48%`; meaningful trades `179`.
- Buy & Hold final net worth: `$8,714.76`.
- Reporting role: historical positive baseline where both compared agents were profitable before the current novelty experiments.
- Reproduction status: completed on `2026-05-25`; outputs in `results/baseline_positive_reproduction` exactly match the documented metrics above.
- Important label: the regenerated files are reproduction artifacts from the archived source snapshot, not the overwritten original files.

### Current Reference Baseline - DQN

- Result source: `results/research/dqn_report.txt` (same reproducible DQN result across recent comparison runs).
- Final net worth: `$10,715.79`.
- Sharpe: `0.6167`.
- Sortino: `0.9873`.
- Max drawdown: `-7.14%`.
- Average daily turnover: `12.53%`.
- Dominant portfolio template: `[0.45 SPY | 0.00 SH | 0.25 TLT]`, share `18.33%`.
- Assessment: strongest verified agent so far. It successfully uses `SH`/`TLT` defensive templates during the downtrend split.

### Experiment - PPO Actor-Critic

- Result source: `results/research/ac_report.txt`.
- Final net worth: `$9,013.62`.
- Sharpe: `-0.7416`.
- Sortino: `-1.0993`.
- Max drawdown: `-14.09%`.
- Average daily turnover: `3.00%`.
- Dominant portfolio template: `[0.60 SPY | 0.00 SH | 0.00 TLT]`, share `41.94%`.
- Hypothesis tested: replace unstable online AC updates with PPO/GAE trajectory learning.
- Outcome: loss stability improved, but out-of-sample policy remained long-biased and unprofitable.
- Decision: retain as an Actor-Critic comparison baseline, not as the primary model.

### Experiment - Discrete SAC

- Result source: `results/sac_v3/sac_report.txt` (reproduces the retained plain Discrete SAC configuration).
- Final net worth: `$10,363.83`.
- Sharpe: `0.2646`.
- Sortino: `0.3894`.
- Max drawdown: `-12.08%`.
- Average daily turnover: `9.52%`.
- Dominant portfolio template: `[0.70 SPY | 0.00 SH | 0.00 TLT]`, share `29.72%`.
- Hypothesis tested: entropy-regularized off-policy actor-critic may improve policy search over discrete portfolio templates.
- Outcome: profitable and above Buy & Hold, but inferior to DQN because the policy remains too equity-long during the bear split.
- Negative ablation: the discarded SAC overlay run added entropy/teacher/hedge overlays and collapsed to Sharpe `-1.7063`; its raw folder is not retained.
- Decision: retain plain Discrete SAC as a valid agent comparator; do not use the failed v2 overlays.

### Experiment - Fixed-CVaR QR-DQN

- Result source: archived result summary below; implementation and raw result folder removed after the negative ablation was documented.
- Final net worth: `$9,650.92`.
- Sharpe: `-0.2576`.
- Sortino: `-0.3626`.
- Max drawdown: `-10.87%`.
- Average daily turnover: `12.59%`.
- Hypothesis tested: distributional quantile critic with fixed lower-tail CVaR action scoring should improve tail-risk control.
- Outcome: failed to outperform DQN or initial capital. It held cash early, then became SPY-heavy late in 2022.
- Decision: use as a negative ablation demonstrating that static CVaR preference is insufficient under market regime shift.

### Experiment - Regime-Conditioned CVaR QR-DQN

- Result source: archived result summary below; implementation and raw result folder removed after the negative ablation was documented.
- Final net worth: `$8,789.10`.
- Sharpe: `-0.7452`.
- Sortino: `-1.1412`.
- Max drawdown: `-16.51%`.
- Average daily turnover: `8.89%`.
- Average policy risk weight: `0.6292`.
- Hypothesis tested: make CVaR risk preference state-dependent using `VIX`, `SPY_Momentum_20`, and `SPY_Trend_Regime`.
- Outcome: risk weight reacted to stress, but action ranking remained wrong. During `2022Q1-Q2`, high risk weights still selected `[0.70 SPY | 0.00 SH | 0.00 TLT]`.
- Interpretation: detecting stress is not enough when the model does not learn which hedge asset remains effective; in 2022, `TLT` was not consistently defensive.
- Decision: discontinue coefficient tuning for QR-DQN, remove it from the active pipeline, and move to dynamic cross-asset hedge representation.

### Experiment - Dynamic Hedge-Graph DQN Pilot

- Implementation status: pilot evaluated; do not promote this configuration to a full benchmark run.
- Core design:
  - Graph Transformer encoder models dynamic relationships among `SPY`, `SH`, and `TLT`.
  - A no-lookahead hedge preference uses `VIX`, `SPY_Momentum_20`, `SPY_Trend_Regime`, and trailing `SPY-TLT` correlation.
  - When stress is high and `SPY-TLT` correlation is positive, the policy downweights SPY/TLT-heavy exposure and prefers `SH` or cash.
- Novelty motivation: explicitly model hedge breakdown rather than only raising generic risk aversion.
- Pilot command:

```powershell
python main.py --mode research --skip-dqn --skip-ac --skip-sac --run-hedge-graph-dqn --hedge-graph-episodes 40 --results-dir results/hedge_graph_pilot
```

Files to record after the run:

- `results/hedge_graph_pilot/hedge_graph_report.txt`
- `results/hedge_graph_pilot/HEDGE_GRAPH_trade_log.csv`
- `results/hedge_graph_pilot/run_config.txt`

Pilot result:

- Training budget: `40` episodes, run without DQN/AC/SAC.
- Final net worth: `$8,615.10`.
- Sharpe: `-0.8753`.
- Sortino: `-1.3268`.
- Max drawdown: `-22.34%`.
- Average daily turnover: `5.23%`.
- Average realized allocation: `68.32%`.
- Average policy stress score: `0.6307`.
- Average trailing SPY-TLT correlation signal: `-0.0737`.
- What worked: policy diagnostics responded to stress and selected `SH`-heavy templates during `2022Q1-Q2`, which is behavior QR-DQN did not learn.
- Failure mode:
  - `2022Q1`: Sharpe `-1.504`, dominant template `[0.00 SPY | 0.70 SH | 0.00 TLT]`.
  - `2022Q2`: Sharpe `-0.992`, dominant template `[0.00 SPY | 0.70 SH | 0.00 TLT]`.
  - `2022Q3`: Sharpe `-5.038`, dominant template `[0.00 SPY | 0.40 SH | 0.30 TLT]`.
- Interpretation: the model can rotate toward a hedge, but its hedge preference persists after the useful hedge period and keeps total exposure close to the `70%` cap. It needs an exit/recovery mechanism, not more training episodes in the same form.
- Decision: do not run the exact version for `120` episodes. The active follow-up now adds explicit `Risk-On / Transition / Risk-Off` switching and lowers allocation preference during transition uncertainty.

### Current Implementation - Three-State Dynamic Hedge-Graph DQN

- Replaces the pilot's always-active hedge preference with three observable policy states:
  - `Risk-On`: support controlled SPY/balanced exposure after recovery.
  - `Risk-Off`: favor `SH`/cash and block concentrated SPY exposure only under strong stress.
  - `Transition`: penalize fully allocated templates and prefer cash while regime signals disagree or recover.
- Diagnostics written to trade logs: `Policy_Stress_Score`, `Policy_SPY_TLT_Correlation`, `Policy_Regime_State`.
- Cleanup decision: QR-DQN source and raw folders are removed from active project artifacts because both fixed and regime-conditioned variants failed and their conclusions are preserved in this log.
- Next pilot command:

```powershell
python main.py --mode research --skip-dqn --skip-ac --skip-sac --run-hedge-graph-dqn --hedge-graph-episodes 40 --results-dir results/hedge_graph_state_pilot
```

### Retained Result Folders For Reporting

- `results/research`: DQN and PPO Actor-Critic benchmark reports.
- `results/sac_v3`: retained plain Discrete SAC comparison result.
- `results/hedge_graph_pilot`: negative pilot ablation showing that persistent hedge preference fails without transition/recovery switching.
- Removed from raw results after documentation: `qrdqn_v1`, `qrdqn_regime_v1`, `smoke_qrdqn`, `smoke_sac`.

### Experiment Result - Three-State Dynamic Hedge-Graph DQN Pilot

- Result source: `results/hedge_graph_state_pilot/hedge_graph_report.txt`.
- Training budget: `40` episodes, run without DQN/AC/SAC.
- Final net worth: `$9,499.31`.
- Sharpe: `-0.5462`.
- Sortino: `-0.6059`.
- Max drawdown: `-8.09%`.
- Average daily turnover: `6.50%`.
- Average realized allocation: `21.45%`.
- Dominant portfolio template: cash `[0.00 SPY | 0.00 SH | 0.00 TLT]`, share `41.39%`.
- Policy regime shares: `Risk-Off 47.50%`, `Transition 27.78%`, `Risk-On 24.72%`.
- Improvement versus first Hedge-Graph pilot:
  - Final net worth improved from `$8,615.10` to `$9,499.31`.
  - Sharpe improved from `-0.8753` to `-0.5462`.
  - Max drawdown improved from `-22.34%` to `-8.09%`.
- Remaining failure mode: transition de-risking solved excessive exposure but created cash drag. The policy is now too defensive, particularly in `2021Q3-Q4` and `2022Q4`.
- Decision: retain this pilot as evidence that three-state switching controls risk; do not run `120` episodes yet. Next version should reduce cash preference in `Risk-On` and permit limited balanced/SH exposure in `Transition` without returning to the original 70% overexposure.

### Current Iteration - Balanced Transition Hedge-Graph DQN

- Change after the three-state pilot:
  - Preserve `Risk-Off` hedge behavior and thresholds because max drawdown improved materially.
  - Reduce `Transition` cash preference and replace it with a diversified-allocation preference.
  - Penalize transition templates whose total allocation exceeds `55%`.
  - Use a Hedge-Graph-only environment cap of `50%` total asset allocation with `6%` maximum daily weight movement; the DQN baseline remains unchanged at its validated configuration.
  - In `Risk-On`, reward balanced SPY/TLT participation and penalize remaining in cash, instead of favoring concentrated SPY exposure.
- Intended effect: raise average allocation above the prior `21.45%` without returning to the original pilot's `68.32%` exposure or `-22.34%` drawdown.
- Next pilot output folder: `results/hedge_graph_balanced_pilot`.

### Experiment Result - Balanced Transition Hedge-Graph DQN Pilot

- Result source: `results/hedge_graph_balanced_pilot/hedge_graph_report.txt`.
- Training budget: `40` episodes.
- Final net worth: `$9,304.58`.
- Sharpe: `-0.6393`.
- Sortino: `-1.0189`.
- Max drawdown: `-10.47%`.
- Average daily turnover: `5.16%`.
- Average realized allocation: `48.04%`.
- Dominant weight template: `[0.000 SPY | 0.286 SH | 0.214 TLT]`, share `35.83%`.
- Outcome: cash drag was removed, but exposure became nearly fixed near the `50%` cap. The pilot is worse than the previous three-state pilot (`$9,499.31`, Sharpe `-0.5462`, drawdown `-8.09%`).
- Failure diagnosis: the original discrete template library provides a jump from cash toward portfolios that are still too large once selected. A scoring adjustment alone cannot create moderate exposure actions.
- Decision: do not tune preference strengths further on the same action set. Add a Hedge-Graph-only regime-specialized portfolio template library with explicit intermediate allocation levels.

### Current Iteration - Regime-Specialized Template Hedge-Graph DQN

- Architectural change: the Hedge-Graph agent now uses its own action library while DQN/AC/SAC baselines retain their original portfolio templates.
- Hedge-Graph template range:
  - cash,
  - cautious/growth exposures between `30%` and `45%`,
  - diversified transition exposures around `32% - 40%`,
  - hedge exposures around `20% - 40%`.
- Environment control: Hedge-Graph maximum allocation is capped at `45%` with `6%` maximum daily asset-weight movement.
- Rationale: allow moderate state-specific positioning rather than forcing the policy to oscillate between cash and near-cap allocations.
- Next pilot output folder: `results/hedge_graph_templates_pilot`.

### Experiment Result - Regime-Specialized Template Hedge-Graph Pilot

- Result source: `results/hedge_graph_templates_pilot/hedge_graph_report.txt`.
- Training budget: `40` episodes.
- Final net worth: `$9,614.15`.
- Sharpe: `-0.4699`.
- Sortino: `-0.6994`.
- Max drawdown: `-6.58%`.
- Average daily turnover: `2.03%`.
- Average realized allocation: `32.05%`.
- Dominant weight template: `[0.20 SPY | 0.00 SH | 0.10 TLT]`, share `54.17%`.
- Assessment: best Hedge-Graph result to date in drawdown and Sharpe, validating moderate templates; it still remains below initial capital and below the positive baseline.
- Failure diagnosis: additive regime prior is too weak relative to learned Q-ranking. In `Risk-Off`, the policy still selected the cautious growth template `[0.20 SPY | 0.00 SH | 0.10 TLT]` most often; `2022Q3` then produced Sharpe `-2.962`.
- Decision: replace additive preference alone with a transparent regime action shield. The graph DQN will select freely only inside a state-appropriate template subset.
- Methodological caution: these iterations use the current held-out period diagnostically. Final reported generalization claims require a fresh temporal holdout or walk-forward evaluation after architecture selection.

### Current Iteration - Shielded Template Hedge-Graph DQN

- Change: keep graph encoding and moderate templates, but apply a deterministic action shield after regime classification:
  - `Risk-On`: permit growth/balanced templates where SPY exposure is not below SH exposure.
  - `Transition`: permit only moderate, diversified templates with allocation no more than `40%`.
  - `Risk-Off`: permit only hedge templates with near-zero SPY and meaningful SH exposure.
- Rationale: the prior pilot showed that merely nudging Q-scores does not stop inappropriate growth selection during `Risk-Off`.
- Next pilot output folder: `results/hedge_graph_shield_pilot`.

### Experiment Result - Hard-Shield Hedge-Graph Pilot

- Result source: `results/hedge_graph_shield_pilot/hedge_graph_report.txt`.
- Training budget: `40` episodes.
- Final net worth: `$9,379.79`.
- Sharpe: `-0.7970`.
- Sortino: `-1.1366`.
- Max drawdown: `-8.07%`.
- Average daily turnover: `5.00%`.
- Average realized allocation: `32.57%`.
- Dominant weight template: `[0.00 SPY | 0.32 SH | 0.08 TLT]`, share `23.06%`.
- Outcome: hard shield reduced inappropriate Risk-Off growth selection in parts of 2022, but worsened the full-period Sharpe versus the regime-specialized template pilot (`$9,614.15`, Sharpe `-0.4699`).
- Failure diagnosis: deterministic masking is too rigid. It prevents the Q-network from overriding the regime classifier when the classifier is only partially right, especially around recovery and mixed-signal quarters.
- Decision: replace the hard mask with a soft regime shield. Invalid templates receive a finite Q-scale penalty instead of being assigned `-inf`, preserving explainable regime guidance while still allowing learned evidence to override it.

### Experiment Result - Soft-Shield Hedge-Graph DQN, Penalty 3.0

- Result source: `results/hedge_graph_soft_shield_pilot/hedge_graph_report.txt`.
- Training budget: `40` episodes.
- Penalty strength: `3.0`.
- Final net worth: `$9,379.79`.
- Sharpe: `-0.7970`.
- Sortino: `-1.1366`.
- Max drawdown: `-8.07%`.
- Average daily turnover: `5.00%`.
- Outcome: metrics and action distribution matched the hard-shield pilot almost exactly. The finite penalty was still large enough relative to Q-score dispersion to behave like a hard mask.
- Decision: lower the default soft-shield penalty from `3.0` to `0.75` and rerun as a separate ablation instead of treating this as an improvement.

### Current Iteration - Soft-Shield Hedge-Graph DQN, Penalty 0.75

- Change: invalid regime templates now receive only `0.75 * std(Q)` penalty.
- Rationale: preserve the best behavior of the regime-specialized template pilot while giving the regime shield a lighter veto against clearly incompatible templates.
- Next pilot output folder: `results/hedge_graph_soft_shield_p075`.

### Experiment Result - Soft-Shield Hedge-Graph DQN, Penalty 0.75

- Result source: `results/hedge_graph_soft_shield_pilot/hedge_graph_report.txt` after rerun with `hedge_shield_penalty_strength: 0.75`.
- Training budget: `40` episodes.
- Final net worth: `$9,163.25`.
- Sharpe: `-1.1871`.
- Sortino: `-1.9183`.
- Max drawdown: `-9.14%`.
- Average daily turnover: `4.05%`.
- Average realized allocation: `33.52%`.
- Dominant weight template: `[0.00 SPY | 0.22 SH | 0.10 TLT]`, share `32.78%`.
- Outcome: lowering the penalty did not recover the best template behavior; it pushed the agent toward hedge templates in the wrong windows and performed worse than both hard-shield and no-shield template pilots.
- Decision: discontinue shield tuning for now. The active Hedge-Graph configuration reverts to regime-specialized templates plus additive regime prior with `hedge_use_regime_action_shield = False`.
- Reporting note: keep the shield variants as negative ablations showing that transparent constraints must remain advisory; absolute or semi-absolute template vetoes overfit the diagnostic split.

### Confirmation Run - No-Shield Hedge-Graph Active Config

- Result source: `results/hedge_graph_no_shield_confirm/hedge_graph_report.txt`.
- Training budget: `40` episodes.
- Config check: `hedge_use_regime_action_shield = False`.
- Final net worth: `$9,614.15`.
- Sharpe: `-0.4699`.
- Sortino: `-0.6994`.
- Max drawdown: `-6.58%`.
- Average daily turnover: `2.03%`.
- Average realized allocation: `32.05%`.
- Dominant weight template: `[0.20 SPY | 0.00 SH | 0.10 TLT]`, share `54.17%`.
- Outcome: exactly reproduces the prior `hedge_graph_templates_pilot`, confirming that the active Hedge-Graph config has been restored to the best-known no-shield variant.
- Next decision: stop shield tuning. The next meaningful experiment should test whether more training budget improves the restored no-shield graph agent, then move to a different novelty mechanism if it remains below the DQN/AC positive baselines.

### Experiment Result - No-Shield Hedge-Graph 120 Episodes

- Result source: `results/hedge_graph_no_shield_120/hedge_graph_report.txt`.
- Training budget: `120` episodes.
- Config check: `hedge_use_regime_action_shield = False`.
- Final net worth: `$9,410.36`.
- Sharpe: `-0.7999`.
- Sortino: `-1.2576`.
- Max drawdown: `-11.13%`.
- Average daily turnover: `2.42%`.
- Average realized allocation: `34.11%`.
- Dominant weight template: `[0.20 SPY | 0.00 SH | 0.10 TLT]`, share `49.72%`.
- Comparison versus 40-episode no-shield confirm:
  - Final net worth decreased from `$9,614.15` to `$9,410.36`.
  - Sharpe decreased from `-0.4699` to `-0.7999`.
  - Max drawdown worsened from `-6.58%` to `-11.13%`.
- Failure diagnosis: additional training did not discover a stronger hedge policy. It rotated returns across quarters rather than improving the full period: `2022Q4` improved, but `2021Q4` and `2022Q2` deteriorated sharply.
- Decision: stop Hedge-Graph training-budget tuning. Retain the 40-episode no-shield template pilot as the best Hedge-Graph ablation, but do not promote it above the positive DQN/AC baselines.

### Result Organization - Hedge-Graph Ablation Folder

- Hedge-Graph result folders have been consolidated under `results/hedge_graph_ablation/raw_runs/`.
- Index file: `results/hedge_graph_ablation/README.md`.
- Original top-level Hedge-Graph paths in earlier notes refer to the run names before consolidation.
- Purpose: keep raw evidence for the NCKH ablation table while reducing clutter in `results/`.

### Experiment Result - Macro Regime v1 Dry Run

- Result source: `results/macro_regime_v1`.
- Config issue: `run_config.txt` shows `include_macro: False`, so this run did not actually include `US10Y`, `US2Y`, yield-curve, or `DXY` features.
- DQN final net worth: `$10,715.79`.
- DQN Sharpe: `0.6167`.
- DQN Sortino: `0.9873`.
- DQN max drawdown: `-7.14%`.
- AC final net worth: `$9,013.62`.
- AC Sharpe: `-0.7416`.
- AC Sortino: `-1.0993`.
- AC max drawdown: `-14.09%`.
- Interpretation: this is a repeat of the current research configuration, not a valid macro ablation.
- Code change after diagnosis: added `--mode macro_research`, which sets `include_macro=True` and defaults to `results/macro_regime_v1`.

### Experiment Result - Macro Regime True v1

- Result source: `results/macro_regime_true_v1`.
- Config check: `include_macro: True`.
- DQN final net worth: `$8,065.97`.
- DQN Sharpe: `-1.3218`.
- DQN Sortino: `-1.6986`.
- DQN max drawdown: `-25.13%`.
- DQN dominant template: `[0.70 SPY | 0.00 SH | 0.00 TLT]`, share `20.28%` by exact weights and `63.9%` by target-allocation label.
- AC final net worth: `$9,173.23`.
- AC Sharpe: `-0.7375`.
- AC Sortino: `-1.1117`.
- AC max drawdown: `-12.15%`.
- Outcome: macro features helped AC final value slightly versus the research AC, but destroyed DQN by pushing it toward high SPY exposure through the 2022 bear market.
- Failure diagnosis:
  - Macro v1 included nonstationary macro level features (`Macro_*_Level`).
  - DQN checkpoint selection did not penalize policy collapse into one dominant high-allocation template.
- Code change for Macro v2:
  - Macro features now exclude raw level columns and use change, 20-day momentum, z-score, yield-curve spread, spread z-score, and inversion flag.
  - Macro mode adds a DQN concentration penalty for single-asset weights above `45%`.
  - DQN validation checkpoint scoring now penalizes excessive dominant-action share.

### Experiment Result - Macro Regime v2

- Result source: `results/macro_regime_v2`.
- Config check: `include_macro: True`, stationary macro features only, DQN concentration penalty enabled.
- DQN final net worth: `$9,289.71`.
- DQN Sharpe: `-0.4637`.
- DQN Sortino: `-0.6641`.
- DQN max drawdown: `-13.97%`.
- DQN dominant template: `[0.45 SPY | 0.00 SH | 0.25 TLT]`, share `13.89%` by exact weights, but `TARGET 70%` still appears in `43.6%` of test steps.
- AC final net worth: `$9,802.47`.
- AC Sharpe: `-0.1446`.
- AC Sortino: `-0.2030`.
- AC max drawdown: `-6.63%`.
- AC dominant template: `[0.390 SPY | 0.122 SH | 0.087 TLT]`, share `33.61%`.
- Outcome:
  - DQN improved materially versus Macro True v1 but remains below the non-macro DQN baseline and below initial capital.
  - AC improved strongly versus Macro True v1 and current research AC, nearly reaching breakeven with controlled drawdown.
- Failure diagnosis:
  - Macro features should not replace the verified DQN baseline; DQN still becomes too long-biased in 2022Q2-Q3.
  - AC benefits from macro/state regularization but remains exposed during `2022Q2`.
- Decision: treat macro as an AC-focused ablation for now. Next step is a mild macro/regime defensive overlay for AC, not another broad DQN macro tuning pass.

### Current Iteration - Macro Regime v3

- Change: add a non-scaled `Macro_Risk_Off_Raw` helper signal derived from rising 10Y yield momentum, rising DXY momentum, yield-curve inversion, and adverse yield-spread z-score.
- Change: add an AC-only `ac_macro_hedge_weight` overlay. When macro risk is elevated, the multi-asset environment blends the AC target portfolio slightly toward an `SH/TLT` defensive mix.
- Default macro-mode overlay strength: `0.35`.
- Rationale: Macro v2 shows direct macro features help AC but still leave it exposed in `2022Q2`; a mild no-lookahead macro defensive overlay is more interpretable than asking the neural policy to infer all macro hedging behavior from noisy features.
- Next run should use `results/macro_regime_v3`.

### Experiment Result - Macro Regime v3

- Result source: `results/macro_regime_v3`.
- Config check: `include_macro: True`, `ac_macro_hedge_weight: 0.35`.
- DQN final net worth: `$9,289.71`.
- DQN Sharpe: `-0.4637`.
- DQN Sortino: `-0.6641`.
- DQN max drawdown: `-13.97%`.
- AC final net worth: `$9,686.69`.
- AC Sharpe: `-0.2694`.
- AC Sortino: `-0.3911`.
- AC max drawdown: `-7.13%`.
- Outcome: DQN stayed identical to Macro v2 as expected, but AC worsened versus Macro v2 (`$9,802.47`, Sharpe `-0.1446`).
- Failure diagnosis: the macro hedge overlay increased AC realized allocation from `40.56%` to `47.97%` and worsened `2022Q2` Sharpe from `-1.907` to `-2.558`. The defensive mix was not actually defensive enough during a period when both equity and duration risk were problematic.
- Decision: discontinue the macro hedge overlay. Macro v2 remains the best macro AC ablation. Next macro run should use stationary macro features with stricter AC allocation control rather than an explicit SH/TLT overlay.

### Current Iteration - Macro Regime v4

- Change: disable `ac_macro_hedge_weight`.
- Change: reduce macro-mode AC maximum total allocation from `0.60` to `0.50`.
- Change: raise macro-mode AC cash logit bias from `0.90` to `1.05`.
- Rationale: Macro v2 is near breakeven but loses too much in `2022Q2`; reducing total allocation is a cleaner risk-control intervention than forcing a fixed hedge basket.
- Next run should use `results/macro_regime_v4`.

### Experiment Result - Macro Regime v4

- Result source: `results/macro_regime_v4`.
- Config check: `include_macro: True`, `ac_max_total_allocation: 0.50`, `ac_cash_logit_bias: 1.05`, `ac_macro_hedge_weight: 0.0`.
- DQN final net worth: `$9,289.71`.
- DQN Sharpe: `-0.4637`.
- DQN Sortino: `-0.6641`.
- DQN max drawdown: `-13.97%`.
- AC final net worth: `$9,600.25`.
- AC Sharpe: `-0.3971`.
- AC Sortino: `-0.5083`.
- AC max drawdown: `-6.87%`.
- Outcome: DQN stayed unchanged and AC worsened versus Macro v2 and Macro v3.
- Failure diagnosis: reducing AC allocation from `40.56%` in Macro v2 to `33.22%` in Macro v4 reduced risk but also removed too much recovery participation. `2022Q2` remained weak, so the lower cap did not solve the main failure mode.
- Decision: Macro v2 remains the best macro ablation. Stop macro allocation/overlay tuning unless a new macro signal design is introduced.

### Result Organization - Macro Regime Ablation Folder

- Macro regime result folders are consolidated under `results/macro_regime_ablation/raw_runs/`.
- Index file: `results/macro_regime_ablation/README.md`.
- Future macro tests should overwrite a stable folder such as `results/macro_regime_current`; important metrics should be copied into this log and the ablation README instead of creating many top-level result folders.

### Current Iteration - Offline Expert Ensemble

- Implementation: `training/ensemble.py` and `scripts/run_ensemble.py`.
- Stable output folder: `results/ensemble_current`.
- Initial hypothesis: regime-gated switching among prior expert policies may improve risk-adjusted performance.
- Important correction: return-splicing between expert trade logs looked strong but was invalid because it ignored the real turnover created when switching experts. The implemented evaluator recomputes PnL from selected expert weights and asset prices, including turnover cost.
- Tested strategy retained as default: fixed `50% AC_BASE + 50% DQN_BASE` target-weight blend.
- Result source: `results/ensemble_current/ensemble_report.txt`.
- Final net worth: `$10,735.69`.
- Sharpe: `0.6953`.
- Sortino: `1.0667`.
- Max drawdown: `-5.46%`.
- Average daily turnover: `8.45%`.
- Average realized allocation: `60.95%`.
- Interpretation:
  - Improves Sharpe versus DQN baseline (`0.4856`) and AC baseline (`0.5523`).
  - Improves Sortino versus both individual baselines.
  - Keeps drawdown close to DQN baseline while preserving most of AC's final value.
- Research role: strongest current risk-adjusted result and a clean non-FinRL novelty candidate as a cost-aware multi-expert portfolio ensemble.

### Experiment Result - Ensemble Blend Ratio Ablation

- Result source: `results/ensemble_ablation/README.md`.
- Evaluation method: recompute PnL from blended expert weights and asset prices with turnover cost.
- Tested AC/DQN blend weights:
  - `25/75`: final `$10,637.67`; Sharpe `0.6286`; Sortino `0.9768`; max drawdown `-4.81%`.
  - `40/60`: final `$10,699.04`; Sharpe `0.6815`; Sortino `1.0405`; max drawdown `-5.09%`.
  - `50/50`: final `$10,735.69`; Sharpe `0.6953`; Sortino `1.0667`; max drawdown `-5.46%`.
  - `60/40`: final `$10,763.64`; Sharpe `0.6891`; Sortino `1.0752`; max drawdown `-6.14%`.
  - `75/25`: final `$10,775.54`; Sharpe `0.6423`; Sortino `1.0253`; max drawdown `-7.32%`.
- Decision: keep `50/50` as the selected proposed method because it maximizes Sharpe while retaining DQN-like drawdown control.
- Alternative note: `60/40` can be reported as a return/Sortino-seeking variant, but it has noticeably larger drawdown.

### Research Summary - Bootstrap Confidence Intervals

- Implementation: `training/research_summary.py` and `scripts/run_research_summary.py`.
- Result source: `results/research_summary/metrics_summary.md`.
- Method: moving-block bootstrap with 20-trading-day blocks and 1,000 resamples.
- Ranking by point-estimate Sharpe:
  - Ensemble 50/50: final `$10,735.76`; Sharpe `0.6953`; Sortino `1.0667`; max drawdown `-5.46%`.
  - AC Baseline: final `$10,780.55`; Sharpe `0.5523`; Sortino `0.8930`; max drawdown `-9.48%`.
  - DQN Baseline: final `$10,526.35`; Sharpe `0.4856`; Sortino `0.7717`; max drawdown `-5.50%`.
  - SAC: final `$10,363.83`; Sharpe `0.2646`; Sortino `0.3894`; max drawdown `-12.08%`.
  - Macro AC v2: final `$9,802.47`; Sharpe `-0.1448`; Sortino `-0.2036`; max drawdown `-6.63%`.
  - HedgeGraph Best: final `$9,614.15`; Sharpe `-0.4699`; Sortino `-0.6994`; max drawdown `-6.58%`.
- Interpretation: the ensemble has the best risk-adjusted point estimates, but bootstrap intervals are wide because the test period is short and regime-concentrated. This should be reported honestly and motivates walk-forward validation as the next robustness check.

### Experiment Result - Adaptive Friction-Aware Ensemble

- Implementation: `training/ensemble.py`, `scripts/run_ensemble.py`, and `scripts/run_ensemble_friction_sweep.py`.
- Strategy tested: `adaptive_blend`.
- Method: dynamically blend AC and DQN weights using prior rolling Sharpe, drawdown, and turnover scores; clamp AC weight to `[0.25, 0.75]`.
- Friction rule: transaction cost cannot be below `0.001`.
- Result source: `results/ensemble_ablation/raw_runs/adaptive_blend/ensemble_report.txt`.
- Transaction cost `0.001` result: final `$10,466.74`; Sharpe `0.4498`; Sortino `0.7276`; max drawdown `-6.52%`; turnover `10.29%`.
- Friction sweep source: `results/ensemble_friction_sweep/friction_sweep.md`.
- Friction sweep:
  - cost `0.0010`: final `$10,466.74`; Sharpe `0.4498`; Sortino `0.7276`; max drawdown `-6.52%`.
  - cost `0.0015`: final `$10,307.61`; Sharpe `0.3109`; Sortino `0.5035`; max drawdown `-6.74%`.
  - cost `0.0020`: final `$10,149.81`; Sharpe `0.1722`; Sortino `0.2790`; max drawdown `-6.95%`.
- Acceptance result: failed to beat the selected 50/50 blend (`$10,735.69`, Sharpe `0.6953`, max drawdown `-5.46%`) and slightly breached the `-6.5%` drawdown target.
- Decision: keep `results/ensemble_current` as the fixed 50/50 blend. Retain adaptive blend as a neutral/negative ablation showing that dynamic switching must overcome transaction-cost-induced turnover.

## Current Winning Baseline

- DQN final net worth: $10,526.35
- DQN Sharpe ratio: 0.4856
- DQN Sortino ratio: 0.7717
- DQN max drawdown: -5.50%
- DQN meaningful trades: 276
- AC final net worth: $10,780.55
- AC Sharpe ratio: 0.5523
- AC Sortino ratio: 0.8930
- AC max drawdown: -9.48%
- AC meaningful trades: 179
- Buy & Hold final net worth: $8,714.76

Status:

- Both agents now beat initial capital and Buy & Hold.
- Both agents now have positive test-set Sharpe.
- Both agents satisfy the non-lazy trading threshold.
- DQN remains the strongest drawdown-stable baseline.
- Actor-Critic is now the strongest Sharpe/Sortino baseline.

## Baseline From Existing Results

- DQN final net worth: $7,316.14
- DQN Sharpe ratio: -1.8574
- AC final net worth: $6,639.85
- AC Sharpe ratio: -2.7126
- DQN behavior: 181 logged steps, mostly fully invested long/short style actions, with large drawdown.
- AC behavior: 181 logged steps, mostly long-biased and also negative performance.

## Success Thresholds

- Annualized Sharpe ratio > 1.25 on the test split.F
- Final net worth > Buy & Hold final net worth on the same test split.
- Final net worth > $10,000 initial capital.
- At least 20 meaningful allocation changes in the test trade log.
- No lazy policy: fewer than 80% of test steps may be pure cash/hold.

## Current Hypotheses

- Backtest Sharpe is being calculated from shaped rewards instead of true portfolio returns.
- Feature scaling currently happens before train/test split, causing look-ahead leakage.
- Flattened MLPs discard the 60-day sequence structure; GRU encoders should be more suitable.
- Buy/sell max actions create unstable all-in/all-out behavior and amplify transaction costs.
- Reward shaping over-penalizes variance/turnover while not directly optimizing portfolio growth.

## Latest Fast-Mode Result

- DQN final net worth: $8,112.31, Sharpe: -0.6933, meaningful trades: 84.
- AC final net worth: $9,127.91, Sharpe: -0.4770, meaningful trades: 325.
- Buy & Hold final net worth: $8,671.49.
- Decision: AC is promising because it beats Buy & Hold on final value, but Sharpe remains below target. Increase both agents to 100 episodes for the next run, then reassess whether DQN needs a separate reward/action redesign.

## Latest 100-Episode Result

- DQN final net worth: $8,049.50, Sharpe: -0.7999, meaningful trades: 89.
- AC final net worth: $9,163.60, Sharpe: -0.4468, meaningful trades: 319.
- Buy & Hold final net worth: $8,671.49.
- DQN turnover is too high, with average daily turnover near 18.8%.
- AC is smoother, with average daily turnover near 3.1%, but remains mostly 40-69% long during a downtrend.
- Decision: stop increasing episodes for now. Add a no-lookahead SMA20/SMA50 regime cap, remove the cash-idle penalty, strengthen turnover/drawdown/downside penalties, and expand DQN discrete allocations to 0/25/50/75/100%.

## Latest Long-Only Risk-Cap Result

- DQN final net worth: $8,210.33, Sharpe: -1.5598, meaningful trades: 148.
- AC final net worth: $8,909.10, Sharpe: -1.0842, meaningful trades: 186.
- Buy & Hold final net worth: $8,671.49.
- Risk cap reduced upside too much and did not solve the negative mean-return problem.
- Decision: long-only is too constrained for the 2022 downtrend test split. Move to signed target exposure so agents can short during bearish regimes, while capping exposure using only current Close/SMA20/SMA50.

## Latest Signed-Exposure Result

- DQN final net worth: $7,592.02, Sharpe: -2.0846, meaningful trades: 157.
- AC final net worth: $9,877.46, Sharpe: -0.6405, meaningful trades: 182.
- Buy & Hold final net worth: $8,671.49.
- AC improved final value materially but learned a near-cash/slightly-short policy, so mean return is still slightly negative.
- DQN remains too noisy with coarse extreme actions and average daily turnover above 22%.
- Decision: add a small no-lookahead SMA-regime auxiliary target for AC so it learns meaningful signed exposure, and reduce DQN action extremes to lower turnover.

## Latest Auxiliary-Regime Result

- DQN final net worth: $7,629.16, Sharpe: -1.8546, meaningful trades: 167.
- AC final net worth: $7,976.39, Sharpe: -1.5175, meaningful trades: 320.
- Buy & Hold final net worth: $8,671.49.
- DQN max drawdown: -26.46%, average daily turnover: 24.76%.
- AC max drawdown: -21.45%, average daily turnover: 13.99%.
- Decision: auxiliary SMA-regime target made the actor too active and harmed performance. Remove it. Add richer no-lookahead momentum/regime features, cap daily exposure changes, and penalize turnover more directly.

## Latest Turnover-Control Result

- DQN final net worth: $8,664.49, Sharpe: -0.9856, meaningful trades: 191.
- AC final net worth: $9,311.12, Sharpe: -0.6366, meaningful trades: 233.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -17.56%, average daily turnover: 5.71%.
- AC max drawdown: -9.99%, average daily turnover: 3.44%.
- Assessment: turnover and drawdown improved materially, but both agents still have negative average daily return. AC is best: it beats Buy & Hold and keeps drawdown below 10%, but it never goes short in the test run, with minimum realized allocation around +12%.
- Decision: interpret actions as residual adjustments around a no-lookahead SMA-regime baseline. This should give the policy meaningful bearish exposure while preserving the turnover cap.

## Latest Residual-Regime Result

- DQN final net worth: $8,464.27, Sharpe: -1.7747, meaningful trades: 280.
- AC final net worth: $9,041.50, Sharpe: -0.9198, meaningful trades: 270.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -16.86%, average daily turnover: 8.32%.
- AC max drawdown: -11.24%, average daily turnover: 5.68%.
- Assessment: residual regime mapping increased bearish exposure, but it also increased turnover and worsened AC final value versus the prior turnover-control result.
- Decision: revert action semantics to signed target exposure with tighter turnover controls, then add validation checkpoint selection inside the train split. The test split remains untouched for final evaluation.

## Latest Validation-Checkpoint Result

- DQN final net worth: $9,358.60, Sharpe: -0.5096, meaningful trades: 237.
- AC final net worth: $9,510.82, Sharpe: -0.4614, meaningful trades: 175.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -9.05%, average daily turnover: 5.09%.
- AC max drawdown: -9.44%, average daily turnover: 2.95%.
- Assessment: validation checkpoint selection is a major improvement. Both agents beat Buy & Hold and keep drawdown below 10%, but both still fail the SOTA threshold because final net worth remains below $10,000 and Sharpe is negative.
- Bottleneck: AC remains a low-beta long policy, with realized allocation from +8% to +57.7% and average allocation around +29.8%. It does not short during bearish periods.
- Decision: keep checkpoint selection and turnover controls. Add a small no-lookahead regime-alignment penalty to discourage positive exposure during strong bearish SMA regimes and discourage short exposure during bullish SMA regimes.

## Latest Regime-Penalty Result

- DQN final net worth: $10,174.14, Sharpe: 0.2049, meaningful trades: 244.
- AC final net worth: $9,493.59, Sharpe: -0.5506, meaningful trades: 175.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -7.46%, average daily turnover: 5.20%.
- AC max drawdown: -8.67%, average daily turnover: 2.45%.
- Assessment: DQN is now the best candidate and passes three key checks: above initial capital, above Buy & Hold, and non-lazy trading. It still fails the SOTA Sharpe threshold because average daily return is only about 0.0058% versus daily volatility around 0.45%.
- Bottleneck: DQN is still mostly a low/medium long policy, with average allocation around +25.5% and minimum allocation only around -13.5%. It needs stronger directional edge in bearish regimes without increasing turnover materially.
- Decision: preserve the winning validation-checkpoint setup. Increase only DQN training budget and slightly strengthen bearish-regime alignment. Do not make broad architecture changes in the next iteration.

## Latest Higher-DQN-Budget Result

- DQN final net worth: $9,672.80, Sharpe: -0.2867, meaningful trades: 235.
- AC final net worth: $9,471.21, Sharpe: -0.5319, meaningful trades: 190.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -8.49%, average daily turnover: 4.98%.
- Assessment: increasing DQN episodes to 150 and strengthening bearish-regime penalty hurt out-of-sample performance. The agent remained defensive and lost the positive Sharpe achieved by the prior configuration.
- Decision: revert to the best-known DQN configuration: 100 DQN episodes and the milder regime-alignment penalty. Future experiments should be separate variants, not replacements for the best candidate.

## Current Optimization Hypothesis

- Both agents need Sharpe improvement more than raw final-value improvement.
- The best DQN run still had daily volatility around 0.45%, while average return was very small.
- AC is consistently too long-biased in bearish regimes.
- Next changes:
  - Keep the best-known episode budget: DQN 100, AC 100.
  - Add no-lookahead volatility targeting using 20-day realized volatility.
  - Tighten bear-regime exposure caps so strong bear regimes cannot remain net long.
  - Select checkpoints using Sharpe, Sortino, drawdown, and turnover instead of mostly final net worth.
  - Extend reports with Sortino, max drawdown, average turnover, and average allocation.

## Latest Volatility-Target Result

- DQN final net worth: $9,747.80, Sharpe: -0.5935, Sortino: -0.7027, meaningful trades: 150.
- AC final net worth: $9,628.91, Sharpe: -1.2150, Sortino: -1.3601, meaningful trades: 120.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -4.94%, average daily turnover: 2.88%, average allocation: +8.74%.
- AC max drawdown: -4.65%, average daily turnover: 1.60%, average allocation: +10.18%.
- Assessment: volatility targeting controlled drawdown very well, but it made both agents too defensive. Average return stayed negative because exposure was too close to cash.
- Decision: raise the volatility target and blend actions lightly with a no-lookahead SMA-regime baseline, so policies keep the drawdown benefit while taking more directional exposure.

## Latest Volatility-Blend Result

- DQN final net worth: $9,257.98, Sharpe: -1.0778, Sortino: -1.1505, meaningful trades: 234.
- AC final net worth: $9,574.47, Sharpe: -1.0704, Sortino: -1.1564, meaningful trades: 136.
- Buy & Hold final net worth: $8,714.76.
- Assessment: raising volatility target and blending with the slow regime baseline increased exposure but worsened Sharpe. The slow 20/50 regime remains too late for this test period.
- Diagnostic: a simple no-lookahead SMA 5/20 trend rule on the existing test close series has positive Sharpe, while 20/50 is negative. This suggests the risk overlay should react faster.
- Decision: disable volatility targeting/regime blend for now, add SMA5/Fast_Trend_Regime features, and move risk caps/alignment penalties from 20/50 to a faster 5/20 regime.

## Latest Fast-Regime Result

- DQN final net worth: $9,538.25, Sharpe: -0.4885, Sortino: -0.6543, meaningful trades: 269.
- AC final net worth: $9,803.68, Sharpe: -0.5621, Sortino: -0.5626, meaningful trades: 209.
- Buy & Hold final net worth: $8,714.76.
- DQN max drawdown: -8.63%, average daily turnover: 5.81%, average allocation: +18.74%.
- AC max drawdown: -4.15%, average daily turnover: 3.24%, average allocation: +4.87%.
- Assessment: fast regime improved versus the prior volatility-blend regression, but still does not beat the best DQN candidate. AC is close to breakeven but remains too close to cash.
- Decision: add a narrow rule fallback. Only when the agent chooses near-cash exposure, blend part of the action toward the no-lookahead 5/20 regime target. This targets the current cash-drag failure without overriding confident agent actions.

## Latest Rule-Fallback Result

- DQN final net worth: $9,201.47, Sharpe: -0.7731, Sortino: -1.0155, meaningful trades: 293.
- AC final net worth: $9,465.07, Sharpe: -0.8539, Sortino: -1.1116, meaningful trades: 270.
- Buy & Hold final net worth: $8,714.76.
- Assessment: rule fallback increased turnover and drawdown and did not improve Sharpe. It also made AC net short on average without enough edge.
- Diagnostic: brute-force SMA rules on the existing close series peaked around Sharpe 0.64, still below the SOTA target 1.25.
- Decision: roll back overlays to the best-known DQN-positive configuration. Further Sharpe improvement likely requires a more substantial algorithm change such as PPO/A2C rollouts, recurrent replay sequences, or a richer multi-asset/hedging setup rather than more heuristic overlays.

## Algorithm Upgrade Implemented

- DQN remains on the best-known validation-checkpoint setup because it is the current strongest candidate.
- Actor-Critic has been upgraded from online one-step TD learning to PPO-style trajectory learning:
  - full-episode rollouts,
  - GAE advantage estimation,
  - clipped policy objective,
  - mini-batch multi-epoch updates,
  - entropy regularization,
  - gradient clipping.
- Rationale: the old Actor-Critic updated on single noisy daily rewards, which made the policy collapse toward low-beta/cash-like behavior. PPO/GAE should provide lower-variance advantages and more stable policy improvement.
- Next evaluation: run `python main.py`, then compare AC against its prior best and verify DQN did not regress.

## Latest PPO/GAE Result

- DQN final net worth: $9,428.05, Sharpe: -0.5564, Sortino: -0.6585.
- AC final net worth: $9,430.81, Sharpe: -0.7389, Sortino: -0.9664.
- Buy & Hold final net worth: $8,714.76.
- Assessment: PPO/GAE did not improve AC and the extra fast-regime input features also prevented DQN from reproducing its best-known positive-Sharpe run.
- Decision: rollback the default pipeline to the simpler best-known configuration: remove SMA5/Fast_Trend_Regime from model inputs and use the prior online TD Actor-Critic loop. Keep PPO code available as an experimental path, but not in the default run.

## Current AC Improvement Attempt

- DQN best-known configuration is preserved.
- AC remains weaker because it learns a low-beta long policy with negative Sharpe.
- New change: train AC with its original online TD objective plus a small DQN-teacher regularizer.
- Rationale: the DQN policy is the only model that achieved positive out-of-sample Sharpe and final value above initial capital. A light imitation loss may guide AC toward useful signed exposure without fully overriding its own reward learning.
- Next evaluation: AC should improve final value/Sharpe without DQN regression. If AC worsens, remove the teacher loss.

## Current Both-Agent Improvement Attempt

- Goal: improve both DQN and AC without using test data for model selection.
- New change: select a simple SMA trend-following rule on the validation split only, then supervised warm-start both agents on the training split before RL fine-tuning.
- DQN warm-start: cross-entropy imitation of the closest discrete allocation.
- AC warm-start: MSE imitation of the continuous target exposure.
- Rationale: prior runs show both agents struggle to discover directional exposure from sparse/noisy trading rewards. Warm-starting gives both policies a non-random trend prior while preserving the existing RL objective and validation checkpoint selection.

## Latest Warm-Start Result

- DQN final net worth: $9,367.58, Sharpe: -0.4544, Sortino: -0.5996.
- AC final net worth: $9,381.39, Sharpe: -0.4963, Sortino: -0.7196.
- Buy & Hold final net worth: $8,714.76.
- Assessment: validation-selected warm-start overfit and pulled both agents into excessive long exposure during the test drawdown. It failed to improve either agent.
- Decision: disable warm-start in the default pipeline. Keep helper functions for future isolated experiments, but do not use them in `main.py`.

## Multi-Asset Hedge Upgrade

- Problem: SPY-only trading does not provide enough robust ways to produce high Sharpe in a 2022-style drawdown. Previous attempts to synthesize short exposure through reward/risk overlays were unstable.
- Change: switch the default pipeline to a multi-asset universe: SPY, SH, and TLT.
- Rationale:
  - SPY provides equity beta.
  - SH provides inverse S&P 500 hedge exposure without synthetic short accounting.
  - TLT provides a bond/defensive sleeve.
- Environment upgrade:
  - DQN now selects among discrete portfolio templates: cash, single-asset portfolios, and defensive/balanced mixes.
  - Actor-Critic now outputs continuous logits over cash plus assets; the environment converts them to long-only portfolio weights by softmax.
  - Portfolio PnL is computed from asset returns and rebalancing turnover.
- Next evaluation: run `python main.py` and compare both agents against SPY Buy & Hold.

## Latest Multi-Asset Initial Result

- DQN final net worth: $7,792.11, Sharpe: -0.8255, max drawdown: -29.39%, average turnover: 28.39%.
- AC final net worth: $8,744.22, Sharpe: -2.4190, max drawdown: -14.24%, average turnover: 4.65%.
- Assessment: the first multi-asset env was too unconstrained. DQN selected near-fully-invested/high-turnover portfolios and suffered large drawdown.
- Decision: add per-step weight smoothing, reduce aggressive template weights, increase turnover/drawdown penalties, and update validation scoring to penalize drawdown and turnover more strongly.

## Latest Multi-Asset Smoothed Result

- DQN final net worth: $8,697.25, Sharpe: -0.4712, Sortino: -0.5875, max drawdown: -23.26%, average turnover: 15.05%.
- AC final net worth: $8,983.98, Sharpe: -1.5713, Sortino: -2.7329, max drawdown: -12.57%, average turnover: 2.40%.
- Assessment: smoothing helped somewhat but DQN still selected full SPY too often and AC remained too invested. The issue is excessive total risk allocation.
- Decision: cap total asset allocation at 70%, remove 100% single-asset templates, lower mixed template weights, and reduce max per-step weight change to 8%.

## Latest Multi-Asset Risk-Capped Result

- DQN final net worth: $10,526.35, Sharpe: 0.4856, Sortino: 0.7717, max drawdown: -5.50%, average turnover: 13.13%.
- AC final net worth: $8,945.26, Sharpe: -1.6034, Sortino: -2.7812, max drawdown: -13.41%, average turnover: 2.14%.
- Assessment: the multi-asset risk cap produced the best DQN result so far by a wide margin. DQN is now robustly above initial capital and Buy & Hold with controlled drawdown. AC remains poor because the continuous softmax policy stays around 70% invested with insufficient cash.
- Decision: preserve DQN setup. For AC only, add a cash logit bias and temperature to the continuous softmax action mapping so the policy can hold a larger cash sleeve and reduce drawdown.

## Latest AC Cash-Bias Result

- DQN final net worth: $10,526.35, Sharpe: 0.4856, Sortino: 0.7717.
- AC final net worth: $9,133.35, Sharpe: -1.6386, Sortino: -2.7654.
- Assessment: cash bias reduced AC allocation and drawdown, but AC learned near-equal static weights rather than useful rotation.
- Decision: preserve DQN and train AC with a multi-asset DQN-teacher loss over the full distribution `[cash, SPY, SH, TLT]`.

## Latest DQN-Teacher AC Result

- DQN final net worth: $10,174.14, Sharpe: 0.2049, Sortino: 0.3003.
- AC final net worth: $9,715.70, Sharpe: -0.2522, Sortino: -0.3508.
- Buy & Hold final net worth: $8,714.76.
- Assessment: DQN remained stable and AC improved materially versus its prior default run, but AC is still below initial capital and Sharpe remains negative.
- Decision: increase teacher regularization modestly from 0.03 to 0.06. This is a narrow AC-only change because the first teacher run moved in the right direction.

## Latest Stronger-Teacher Result

- DQN final net worth: $10,174.14, Sharpe: 0.2049, Sortino: 0.3003.
- AC final net worth: $9,591.30, Sharpe: -0.3172, Sortino: -0.4417.
- Buy & Hold final net worth: $8,714.76.
- Assessment: increasing teacher regularization from 0.03 to 0.06 hurt AC. The teacher helps only as a light regularizer.
- Decision: revert teacher regularization to 0.03, which is the best AC result in this teacher-learning phase.

## Planned Changes

- Fit `StandardScaler` on train data only, then transform train/test with the same scaler.
- Evaluate Sharpe with true net-worth percentage returns.
- Replace flattened MLP encoders with GRU-based DQN and Actor-Critic models.
- Change actions to target allocations: DQN = cash / 50% long / 100% long; AC = continuous 0-100% target long allocation.
- Use log-return reward with drawdown, turnover, and mild idle-cash penalties.
- Add trade-count and Buy & Hold metrics to reports.

## Options-Sentiment Pipeline Extension

- Added daily options-derived features from local raw files in `data/external`.
- Sources used now:
  - `spy_eod_total.csv`: SPY put/call volume, IV skew, ATM put/call ratio, and a local SPY price cache.
  - `TLT_data.csv`, `HYG_data.csv`, `LQD_data.csv`, `EMB_data.csv`: bond/credit ETF put-call, short-interest, and bid-ask stress features.
- Output feature file: `data/external/options_features_daily.csv`.
- New mode: `python main.py --mode options_research`.
- `options_research` currently uses `SPY` and `TLT` only, matching the decision not to force SH options data into this branch.
- Implementation note: Yahoo Finance is still used when available, but the pipeline can fall back to local SPY/TLT prices and realized-volatility VIX proxy if Yahoo returns empty data.
- Smoke check passed with `SPY + TLT`, 78 total state features, 42 options features, and no shape break in DQN/AC training/evaluation.

## Latest Options Research Result

- Command family: `python main.py --mode options_research --skip-sac --results-dir results/options_current`.
- Universe: `SPY + TLT`; SH was intentionally removed from this branch.
- DQN final net worth: $9,255.93, Sharpe: -1.8474, Sortino: -1.8852, max drawdown: -8.09%, average turnover: 6.89%.
- AC final net worth: $9,432.00, Sharpe: -1.0550, Sortino: -0.6395, max drawdown: -6.36%, average turnover: 1.20%.
- Buy & Hold final net worth: $8,714.76.
- Assessment: put/call features alone did not lift Sharpe when SH was removed. Both agents stayed defensive and beat Buy & Hold, but they still ended below initial capital. The likely cause is that 2022 hurt both SPY and TLT, while the model no longer had SH/inverse hedge exposure to monetize risk-off signals.
- Decision: keep this as a negative/neutral options ablation. Next serious run should use options features as signals while restoring SH as a tradable hedge asset, without requiring SH options data.

## Latest Options + SH Result

- Command family: `python main.py --mode options_research --tickers SPY SH TLT --skip-sac --results-dir results/options_current`.
- Universe: `SPY + SH + TLT`; options features still come from SPY/TLT/HYG/LQD/EMB, not SH options.
- DQN final net worth: $9,176.00, Sharpe: -0.4294, Sortino: -0.6080, max drawdown: -14.39%, average turnover: 10.36%.
- AC final net worth: $9,812.21, Sharpe: -0.1654, Sortino: -0.2189, max drawdown: -9.04%, average turnover: 4.17%.
- Buy & Hold final net worth: $8,714.76.
- Assessment: adding SH back improved Sharpe materially versus the SPY+TLT options branch, especially for AC. However, DQN over-allocated to SPY on average and suffered larger drawdown, so this is not yet competitive with the positive baseline or the 50/50 ensemble.
- Decision: keep options+SH as a promising ablation. Next improvement should reduce noisy options dimensionality and add a risk-off allocation prior that uses SPY put/call/IV-skew stress to increase SH/TLT exposure only when stress is elevated.

## Options Compact Risk-Off Revision

- Change: replaced the full 42-column options feature block in `options_research` with compact stress features:
  - `Options_RiskOffScore`
  - `Options_RiskOff_High`
  - SPY put/call Z-score, SPY ATM put/call Z-score, SPY IV-skew Z-score
  - credit stress, bond stress, and TLT put/call Z-score
- Added helper column `Options_RiskOff_Raw` for environment-side hedge logic; it is joined into the data frame but not standardized as a model input feature.
- Added `options_hedge_weight` controls so the environment can gradually blend toward SH/TLT when options stress is high.
- `options_research` defaults now use `options_feature_mode="compact"`, `dqn_options_hedge_weight=0.28`, and `ac_options_hedge_weight=0.24`.
- Smoke check passed with `SPY + SH + TLT`: 62 state features, 8 compact options features, and no DQN/AC train/eval shape errors.

## Latest Options Compact Result

- DQN final net worth: $9,973.26, Sharpe: 0.0468, Sortino: 0.0694, max drawdown: -12.82%, average turnover: 10.56%.
- AC final net worth: $9,991.00, Sharpe: 0.0234, Sortino: 0.0305, max drawdown: -10.40%, average turnover: 5.07%.
- Assessment: compact risk-off features materially improved both agents from negative Sharpe to slightly positive Sharpe, but the drawdown is still too high. Trade logs show the worst test days happened when the policy still held 54%-70% SPY exposure.
- Decision: make the options branch more defensive by lowering the DQN max allocation cap to 60%, adding a small DQN concentration penalty, and triggering options-driven SH/TLT hedge earlier when SPY trend is already negative.

## Options Defensive Cap Revision

- Changed `options_research` defaults:
  - `dqn_max_total_allocation=0.60`
  - `dqn_concentration_penalty_coef=0.0040`
  - `dqn_options_hedge_weight=0.42`
  - `ac_options_hedge_weight=0.32`
  - `options_hedge_trigger=0.42`
- Added trend-aware options trigger: if SPY is below its regime moving average and 20-day momentum is negative, the options hedge trigger can drop to 0.36.
- Rationale: reduce large SPY losses on crash days without relying on future returns or test labels.

## Latest Options Defensive Cap Result

- DQN final net worth: $9,921.74, Sharpe: -0.0478, Sortino: -0.0713, max drawdown: -7.91%, average turnover: 13.68%.
- AC final net worth: $8,809.75, Sharpe: -0.9060, Sortino: -1.3123, max drawdown: -14.83%, average turnover: 3.92%.
- Assessment: the defensive cap reduced DQN drawdown but hurt return and raised turnover. AC failed badly because it drifted back to high SPY exposure, averaging about 49% SPY and suffering large losses on 2022 selloff days.
- Decision: revert DQN toward the previous compact configuration and make AC conservative through lower total allocation/cash bias rather than stronger options hedge blending.

## Options Balanced Recalibration

- Changed `options_research` defaults:
  - `dqn_max_total_allocation=0.70`
  - `dqn_concentration_penalty_coef=0.0`
  - `dqn_options_hedge_weight=0.28`
  - `ac_max_total_allocation=0.45`
  - `ac_cash_logit_bias=1.20`
  - `ac_options_hedge_weight=0.24`
  - `options_hedge_trigger=0.45`
- Rationale: preserve the DQN compact result that was near break-even, while preventing AC from holding excessive SPY during the 2022 drawdown.

## Latest Options Balanced Recalibration Result

- DQN final net worth: $9,523.73, Sharpe: -0.3032, Sortino: -0.4118, max drawdown: -11.01%, average turnover: 12.34%.
- AC final net worth: $8,887.24, Sharpe: -1.1974, Sortino: -1.7363, max drawdown: -13.23%, average turnover: 3.91%.
- Assessment: this run was pathological. AC saturated at its maximum `TARGET 45%` action for most of the test, while DQN still selected `TARGET 70%` too often. The trend-aware hedge trigger made the policy harder to interpret and did not improve robustness.
- Decision: remove the trend-aware trigger lowering, restore AC to the previous compact allocation settings, and use a fixed high options hedge threshold.

## Options Stable Compact Revert

- Changed `options_research` defaults:
  - `ac_max_total_allocation=0.60`
  - `ac_cash_logit_bias=0.90`
  - `options_hedge_trigger=0.55`
- Removed the dynamic lowering of options hedge trigger when SPY trend is negative.
- Rationale: recover the prior compact behavior that produced slightly positive Sharpe for both agents and avoid action saturation.

## Latest Options Stable Compact Result

- DQN final net worth: $9,973.26, Sharpe: 0.0468, Sortino: 0.0694, max drawdown: -12.82%, average turnover: 10.56%.
- AC final net worth: $9,991.00, Sharpe: 0.0234, Sortino: 0.0305, max drawdown: -10.40%, average turnover: 5.07%.
- Assessment: the revert recovered the slightly positive Sharpe behavior. DQN remains too aggressive and AC is more defensive, making them complementary experts.
- Decision: test an offline adaptive blend of the options DQN/AC logs instead of further mutating the environment.

## Options Adaptive Ensemble Result

- Source logs: `results/options_current/DQN_trade_log.csv` and `results/options_current/AC_trade_log.csv`.
- Fixed 50/50 blend: final net worth $10,039.80, Sharpe 0.0758, Sortino 0.1106, max drawdown -9.27%, average turnover 6.60%.
- Best fixed blend sweep: AC 60% / DQN 40%, final net worth $10,041.50, Sharpe 0.0765, max drawdown -9.18%.
- Adaptive blend without smoothing: final net worth $10,095.62, Sharpe 0.1184, Sortino 0.1762, max drawdown -9.94%, average turnover 7.80%.
- Adaptive blend with `max_weight_delta=0.05`: final net worth $10,201.52, Sharpe 0.1990, Sortino 0.2934, max drawdown -10.03%, average turnover 6.78%.
- Friction sensitivity for smoothed adaptive blend:
  - Cost 0.0010: final $10,201.52, Sharpe 0.1990.
  - Cost 0.0015: final $10,078.96, Sharpe 0.1061.
  - Cost 0.0020: final $9,958.02, Sharpe 0.0132.
- Decision: keep `results/options_ensemble_current` as the smoothed adaptive options ensemble candidate. It does not beat the baseline 50/50 AC-DQN ensemble, but it is the strongest options-based ablation so far and remains near break-even under 0.2% cost.

## Meta Ensemble With Options Overlay

- Goal: test whether the best options ensemble can improve the strong baseline ensemble without retraining agents.
- Method: blend `results/ensemble_current/ENSEMBLE_trade_log.csv` with `results/options_ensemble_current/ENSEMBLE_trade_log.csv`, then recompute PnL from blended weights and prices with transaction cost.
- Fixed options overlay sweep:
  - 0% options: final $10,737.38, Sharpe 0.6967, max drawdown -5.45%.
  - 5% options: final $10,720.70, Sharpe 0.6872, max drawdown -5.38%.
  - 20% options: final $10,668.49, Sharpe 0.6486, max drawdown -5.19%.
  - Fixed overlay reduced drawdown slightly but reduced Sharpe.
- Adaptive options overlay with `max_options_weight=0.15` and `max_weight_delta=0.05`:
  - Final net worth: $10,732.19.
  - Sharpe: 0.7045.
  - Sortino: 1.0767.
  - Max drawdown: -5.45%.
  - Average turnover: 7.28%.
  - Average options blend weight: 11.95%.
- Friction sensitivity:
  - Cost 0.0010: final $10,732.19, Sharpe 0.7045.
  - Cost 0.0015: final $10,588.07, Sharpe 0.5766.
  - Cost 0.0020: final $10,445.64, Sharpe 0.4486.
- Decision: keep `results/meta_ensemble_current` as the new top candidate by Sharpe. The improvement over the baseline ensemble is small, so report it as an options overlay refinement rather than a large breakthrough.

## Results Folder Cleanup

- Standardized current result folders:
  - `results/meta_ensemble_current`
  - `results/ensemble_current`
  - `results/options_current`
  - `results/options_ensemble_current`
  - `results/research_summary`
- Kept friction folders that support market-friction claims:
  - `results/meta_ensemble_friction_sweep`
  - `results/options_ensemble_friction_sweep`
  - `results/ensemble_friction_sweep`
- Removed intermediate grid/smoke folders after recording their key metrics:
  - `options_ensemble_adaptive`
  - `options_ensemble_sweep`
  - `options_ensemble_smooth`
  - `meta_ensemble_sweep`
  - `meta_ensemble_adaptive_sweep`
  - `options_smoke`
- Updated `results/README.md`, `results/latest_run.txt`, and `results/research_summary/metrics_summary.*`.

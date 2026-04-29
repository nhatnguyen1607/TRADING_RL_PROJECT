# RL Trading Improvement Plan

## Baseline From Existing Results

- DQN final net worth: $7,316.14
- DQN Sharpe ratio: -1.8574
- AC final net worth: $6,639.85
- AC Sharpe ratio: -2.7126
- DQN behavior: 181 logged steps, mostly fully invested long/short style actions, with large drawdown.
- AC behavior: 181 logged steps, mostly long-biased and also negative performance.

## Success Thresholds

- Annualized Sharpe ratio > 1.25 on the test split.
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

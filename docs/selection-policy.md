# Real selection policy (Phase 6.1–6.3)

## Phase 6.4 optional stability

`RealCostBasedSelector` and `RealCommunicationCostModel` remain pure. `StableRealCostSelector` wraps the raw decision when explicitly enabled. Its JSON-safe config contains `enabled`, `min_absolute_saving_ms`, and `min_relative_saving`. The default production runtime leaves stability disabled pending CUDA calibration and network evidence. With no previous successful mode, raw cheaper wins (tie: TOKEN). On a proposed switch, candidate saving is previous-mode cost minus candidate cost at the same `MESSAGE_READY` boundary; the saving must meet **both** absolute and relative thresholds. Relative saving divides by previous-mode estimated cost. Equality passes. The final decision always remains TOKEN or KV. No estimate or calibration point is altered.

The state is the previous **successfully executed and correctness-verified** mode. Runtime calls `record_success` only after the selected path completes, so exceptions and failed correctness do not advance it. Metadata distinguishes `raw_candidate_mode`, `final_selected_mode`, `previous_mode`, candidate saving, thresholds, `switch_allowed`, and `switch_suppressed`. Decision margin is the absolute and relative gap between estimated TOKEN/KV totals; a configurable near-tie flag in sensitivity output is analysis only. `REPRESENTATION_READY` is available for offline analysis but Runtime remains `MESSAGE_READY`, including Sender KV preparation.

`run_stable_selector_demo.py` uses a **synthetic fixture, not a measured hardware profile**, to show raw flapping near crossover and stable two-way switching with illustrative thresholds. `run_selection_sensitivity.py` sweeps token count, bandwidth, base latency and both boundaries and marks exact/interpolated/extrapolated points. Base latency adds equally to both modes, so it cancels from cost gap and crossover. `run_prediction_error_report.py` summarizes saved real runtime results as MAE, median absolute percentage error and max absolute error; it never updates a profile. CUDA threshold selection remains pending. See server validation（原研发资料，未随公开副本提供：`server-validation.md`）.

## Dashboard semantics (Phase 6.3)

`Estimated` 是 Selector 在 Data Plane 前用 calibration、协议 wire size 和当前网络快照预测的两路 MESSAGE_READY 成本。`Selected` 是 `RealCostBasedSelector` 对当前 context 的决定，`estimated_saving_ms` 由后端定义。`Actual` 只属于唯一执行的路径，包含单次本机阶段测量与模拟链路；`prediction error` 是选中路径 actual 与 predicted total 的绝对差及相对差。未选路径显示 `Not executed`，不是 0 ms。Dashboard 不提供 force mode，也不从两路数字自行计算选择。MPS profile 是开发校准；CUDA server 应替换为设备兼容的新 profile。

## Runtime execution contract (Phase 6.2)

Runtime 固定 `MESSAGE_READY`：Sender 首轮 generation 在 mode comparison 之前；后续轮次 Receiver output IDs 已经是新的 message。每轮只有一份 immutable canonical IDs，Selector 读取其 count，然后选中一路接收完全相同的 IDs。Selection 在任何 Token packet、KV clean Prefill、Receiver Prefill/restore 之前。未选路径完全不执行，也不产生 actual measurement。实际成本仅汇总选中路径、与 `MESSAGE_READY` 预测同边界的阶段；结果记录 predicted/actual/absolute/relative error，但不自动修改 profile。`REPRESENTATION_READY` 仍只供离线分析，不作为当前 Runtime execution policy。Runtime 的锁保证网络状态在决策与执行期间一致；A/B 方向只在 correctness 通过后切换。

The Selector runs **before** either real Data Plane path. A canonical handoff span already exists as Token IDs; only its count enters `RealSelectionContext`. Production selection must not Prefill, serialize a packet, restore a cache, or benchmark both modes. Phase 6.2 will execute only the selected mode.

## Boundaries and exact inputs

`MESSAGE_READY` is the default and fair same-span boundary:

```text
TOKEN = token serialize + token link + token deserialize + Receiver Prefill
KV    = Sender clean Prefill(M) + KV serialize + KV link + KV deserialize + Receiver restore
```

`REPRESENTATION_READY` is secondary analysis for a naturally existing KV representation. It excludes Sender clean Prefill from KV only. It does not change the default policy.

The decision receives canonical token count, current decimal Mbps bandwidth, base latency, model identity, structural KV metadata (layers, KV heads, head dimension, batch size, dtype), source device, profile identity, and protocol versions. Token Protocol v1 estimates its exact packet length from its own header and width constants. KV Protocol v1 derives raw body and metadata size from its own binary descriptors and model/device metadata lengths. Calibration verifies both estimators against actual packets at each point. `SimulatedNetwork` computes link latency deterministically from wire bytes and current state; link cost is never stored in the calibration profile.

## Compute calibration

`RealCommunicationCostProfile` schema v1 contains model and device identity, protocol versions, measurement policy, warmup/runs, and ascending token-count points. Each point stores medians for Token serialize/deserialize/Receiver Prefill and KV Sender prepare/serialize/deserialize/restore. The KV serialize stage includes device to host and encode; KV restore includes CPU reconstruction, host to device and cache construction. Local in-memory transfer and Receiver continuation are diagnostic and excluded. Profile JSON contains no Token IDs, packet, tensor, cache or model object.

At an exact point the median is used directly. Between points, each compute stage uses piecewise linear interpolation. Outside the range, it uses the adjacent end segment's linear extrapolation, clamped to zero. Every estimate says `exact`, `interpolated`, or `extrapolated`. Extrapolated estimates can be less reliable; adding server calibration points is preferable for workloads outside 8–128 tokens.

The profile is strictly checked against the runtime context for model, structural metadata, dtype, batch, device type, source device, profile name and both protocol versions. An MPS profile cannot silently serve CUDA. The current `macbook-development` MPS profile is for Cost Model, Selector, Adaptive Runtime and Dashboard development only. It is **not** a final server performance result, paper benchmark, or lab deployment calibration.

## Crossover and interpretation

The selector compares totals and chooses TOKEN on a tie. There is no fixed bandwidth threshold. A crossover bandwidth, when positive, is derived analytically from compute-cost difference and wire-byte difference; equal base latency cancels. If KV's compute cost is not lower, there is no positive crossover. A missing crossover is a valid result and mode switching is never forced. MPS timings vary with thermal/load and cache state; medians improve stability but do not promise precise prediction.

The current 8/32/128 MPS development profile yields no `MESSAGE_READY` crossover at any positive bandwidth. `REPRESENTATION_READY` yields no crossover at 8 or 32 and about 1161.87 Mbps at 128. These values are development observations, not final conclusions.

## Commands and CUDA migration

```bash
HF_HUB_OFFLINE=1 python run_selector_calibration.py --device mps --output benchmark-results/mac-mps-selector-profile.json
python run_real_selector_demo.py --profile benchmark-results/mac-mps-selector-profile.json
HF_HUB_OFFLINE=1 python run_selector_validation.py --profile benchmark-results/mac-mps-selector-profile.json
```

Calibration prints JSON but does not write unless `--output` is supplied. `benchmark-results/` is gitignored. On the CUDA server, run `python run_selector_calibration.py --device cuda --profile-name lab-server-rtx-xxx --output benchmark-results/cuda-selector-profile.json`; then have Phase 6.2 load that profile and build `RealSelectionContext` from actual runtime metadata and current network state. **Do not change** `RealCostBasedSelector`, cost formulas, wire-size estimators, or interpolation architecture. If CUDA hardware has a different KV layout or protocol, that is a compatibility failure requiring an explicit new contract rather than silent MPS reuse.

Offline validation calls `select()` first and only then benchmarks both real paths. Its measured costs use the same formal boundaries and independent medians. This dual execution is benchmark-only and must never enter production selection.


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

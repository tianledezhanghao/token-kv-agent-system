# 实验复现与证据口径

[安装与启动](getting-started.md) · [文档索引](README.md)

## 复现层级

| 层级 | 本仓库提供什么 | 需要额外准备什么 |
| --- | --- | --- |
| 工程复现 | Python / 前端测试、真实 codec/事务、tiny/fixture 路径 | Python、Node、依赖；无需下载预训练权重 |
| 本地真实推理 | TOKEN / KV / AUTO 的正式执行入口 | 完整模型快照、足够内存、相应设备；正式 Dashboard 仅 MPS |
| 历史数值/性能复核 | 筛选的研究说明；不含完整冻结证据 | 原研发库 raw、原历史提交与相同环境设备，均不随本副本提供 |
| 物理无线/服务器复现 | 历史软件拓扑和模板 | 不在当前验证范围；没有可宣称已复现的空口/双GPU结果 |

普通 clone 不包含本地 raw 数据、权重、运行日志、完整历史证据或用户 `exports/`。部分 exporter 会读取严格冻结的源码 SHA，甚至重新写入报告；先读源码和 frozen spec，用对应历史 checkout 及新的输出目录复核，不要直接覆盖现有证据。路径兼容不等于当前源码 SHA 与历史版本相同。

## 先做工程验收

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest
cd frontend
npm test
npm run build
cd ..
```

[本轮验收](release/public-release.md) 将既有环境与干净安装分开记录。测试通过不能代替语义质量或真实性能实验。

## 短 TOKEN 基线

先按安装指南准备固定 revision；同一终端解析其目录。若已有本地 Dashboard 正在占用模型资源，先保存结果并正常停止自己启动的服务，避免重复加载。下面会执行真实模型，请自行选择合适的资源窗口；本轮整理只检查参数/启动，不重跑该研究实验。

```bash
MODEL_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_files_only=True))")
mkdir -p benchmark-results
TASK_RUN_DIR=$(mktemp -d benchmark-results/reproduction.XXXXXX)
python run_discussion_demo.py \
  --model "$MODEL_DIR" --device mps \
  --question 'Calculate 23 + 48. Give the numeric result and a brief check.' \
  --rounds 1 --max-new-tokens 24 --max-context-tokens 4096 \
  --mode token --output "$TASK_RUN_DIR/token.json"
```

输出 JSON 保留模型元数据、冻结配置、实际发言/交接、终止原因及未交付 draft。`completed` / receipt 表示工程流程，答案仍需独立判断；达到 Token 上限时保留截断，不自动当作错误或重跑替换。

## Shannon + Synthetic 课堂场景

在同一终端、同一模型目录与新输出目录下运行：

```bash
python run_discussion_demo.py \
  --model "$MODEL_DIR" --device mps \
  --question 'Calculate 23 + 48. Give the numeric result and a brief check.' \
  --rounds 4 --max-new-tokens 24 --max-context-tokens 4096 \
  --mode auto --auto-policy performance_experiment --auto-kv-version v0 \
  --idealized-simulation --network-mode shannon \
  --channel-json '{"bandwidth_mhz":10,"snr_mean_db":10,"snr_sigma_db":6,"correlation":0.85,"seed":42,"efficiency":0.8}' \
  --base-latency-ms 0 --output "$TASK_RUN_DIR/shannon-synthetic.json"
```

这是 **Synthetic 本地计算成本假设 + 真实推理/协议 + 模拟网络**。CLI 使用默认六系数；UI 在高级配置中可修改并冻结系数。默认 `TOKEN=20+2N+0.002NH ms`、`KV=5+0.05N+0.2k ms`。同 seed 且相同参数更新顺序可重现 SNR；消息长度/历史不同会改变预测和策略，不保证每次切换。

[9.0 原始场景](shannon-dynamic-channel-simulator.md) 的已记录 0.5B 实验在 #5 执行真实 V0、#6 回 TOKEN；该结论只证明当时 Synthetic 场景的路由与实际交接。不要预填模式/数据或为了切换篡改 Measured Profile。

## Measured、KV 与质量研究

- Measured：严格使用当前模型/revision、硬件、dtype、算法与 workload 域匹配的 Profile；无校准、域外、缺角点或误差不合格时保守处理。优先读 [泛化成本方法](generalized-auto-cost-model.md)，CLI 为 `python run_generalized_calibration.py --help`；不要把 Canonical Profile 导入讨论。
- KV：从 [V0/V1 审计](kv-v0-v1-implementation-audit.md)、[V2](kv-v2-suffix-recompute.md)、[V3](kv-v3-feasibility-and-validation.md) 进入冻结方法。手动 CLI 支持 `--mode kv --kv-strategy v0/v1/v2/v3`；V2 用 `--kv-recompute-tokens` 或 `--kv-recompute-ratio`，不是同时使用。
- 质量：[TOKEN 冻结基线](token-communication-baseline.md)、[系统质量研究](token-kv-system-quality-validation.md) 将工程交付、信息提取、私有信息使用和最终答案分开。失败、unknown、not_applicable、保护中止及 pilot 都需保留，不合并成“通用准确率”。
- 性能：[性能拆解](token-kv-performance-characterization.md)、[传输通路优化](kv-transfer-path-optimization.md) 区分 warmup、单次测量、统计量、准备/恢复与模拟链路；更低数值误差或更少 Prefill 不自动等于更低 E2E 成本。

## 复现记录至少包含

Git commit、Python/依赖版本、系统/设备/精度、模型与 Tokenizer revision/配置、System Prompt/阶段提示、实际 Token IDs 与终止条件、轮数/上下文/生成预算、策略/修正参数、网络模式/seed/参数版本、Profile 身份与校准域或 Synthetic 六系数、完整 bytes 口径、实测/模拟/预测/派生来源、失败与资源保护结果。

历史大数据如需分享，必须先完成权限与安全审查。现有 raw 不是本轮已获公开许可的数据集；可用 [脚本清单](../scripts/README.md) 查找对应生成、分析和导出工具，不通过目录美化删除负面证据。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

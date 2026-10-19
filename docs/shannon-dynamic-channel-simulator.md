# 9.0 — 香农容量与动态 SNR 模拟器

2026-10-16，起点 `2f21965`（8.6C）。课程正式入口改为 Mac Local：真实 LLM 推理、独立 Agent 上下文与 Cache、真实 TOKEN/KV 协议处理，加上通信理论信道模拟。无需服务器、双笔记本或无线设备。模拟不替代模型，不证明硬件吞吐、KV 加速或协作质量。

## 1. 公开副本说明

本公开副本省略原研发库的本地整理与远端部署交接记录。以下技术方法与已记录结果保留；没有新增服务器或无线实验。

## 2. 公式、单位与范围

使用

```text
SNR_linear = 10^(SNR_dB / 10)
C_Mbps = B_MHz × log2(1 + SNR_linear)
R_Mbps = η × C_Mbps
T_network_ms = 8 × actual_or_predicted_wire_bytes / (1000 × R_Mbps) + L_ms
```

MHz 为频谱宽度，Mbps 为速率；二者不互相替代。η 仅为用户设定的理想容量折减假设，C 是理论上界，R 是模拟速率。L 是每次单向交接收取一次的额外延迟，不是 RTT。默认逻辑模拟不 sleep、不限制 InMemory 实际执行、不把模拟耗时混入 observed wall time。

| 参数 | 新草稿默认 | 接受范围/含义 |
|---|---:|---|
| B | 20 MHz | 0.001–100000 MHz，数值探索域，不代表真实设备带宽 |
| μ | 10 dB | −120–120 dB |
| σ | 4 dB | 0–40 dB，标准差，**不是最大振幅** |
| ρ | 0.85 | 0≤ρ<1 |
| seed | 42 | uint32 整数，0–4294967295 |
| η | 0.8 | 0<η≤1，拒绝下溢为零速率的极端组合 |
| L | 0 ms | 有限非负数 |
| SNR 截断界 | 留空 | 可选 −120–120 dB；下界≤上界 |

为数值安全，未设截断界时也在 ±120 dB 硬限制内计算；同时保存未截断值与 clipped 标记。使用 `log1p` 计算容量，拒绝 NaN/Inf、bool 冒充数字、非法范围、超大传输时间溢出。有限参数仍可能产生无法表示的费用，届时明确失败，不发布 Inf。

## 3. AR(1)、采样与重现

内部保留标准化状态 z：首样本 `z₁=ε₁`（平稳初始化），后续 `z_t=ρz_(t−1)+sqrt(1−ρ²)ε_t`，`SNR_t=μ+σz_t`，ε 为独立标准正态。无截断时边际标准差为 σ、相邻相关系数为 ρ。ρ=0 是独立样本特例；默认有时间相关性。σ=0 时 SNR 恒定。

**每个 handoff 在选择前冻结一次，整次交接使用同一快照**。横轴为 handoff index，不是秒。每个 Session 有独立 `random.Random(seed)`、AR 状态与锁；重复 snapshot、REST GET、刷新和 WS 重放不会消耗新样本。一次新快照只能推进一个序号。异常不隐式重试。

当前范围只实现基础 AR(1)；可用显式 PATCH 调整 SNR 场景。没有额外的突发衰落/拥塞过程，没有臆造无线设备协议。加入截断后，σ/ρ 不再保证等于截断后样本统计量。

## 4. PATCH 与冻结边界

活动 Local Session 使用原 API：

```bash
curl -sS -X PATCH "http://127.0.0.1:8000/api/discussions/$SESSION_ID/network" \
  -H 'Content-Type: application/json' \
  -d '{"channel":{"snr_mean_db":0,"snr_sigma_db":6},"base_latency_ms":2}'
```

部分 channel 字段合并当前配置，再完整校验。更新与 snapshot 共享 RLock；只影响下一未冻结序号，旧快照不改。记录 before/after、revision、接受时间、从哪个序号生效、B 是否变化。B 默认固定；若修改 `channel.bandwidth_mhz`，界面明确提示固定带宽实验条件改变，图表按 revision 分段。

同 seed 的 μ/σ/ρ/η/B 修改保留标准化 AR 状态与 RNG 推进；修改 seed 明确从新 seed 的首个平稳状态开始，但全局 handoff index 不重置。历史展示使用保存的样本，不重新生成。终态或所有 handoff 已冻结时拒绝更新。Shannon 模式禁止直接 PATCH `bandwidth_mbps`，因为 R 由 B/SNR/η 推导。

manual/trace API 和旧 Session 继续可读、可运行；trace 的 live override 规则不变。旧 API 未显式给轮数时保留两轮语义；Shannon 新请求和空白 Dashboard 草稿默认四轮。

## 5. AUTO、真实字节与本地成本

仍然只比较 TOKEN 与用户指定的一个 KV 版本；质量优先默认排除未经认证 KV，V3 额外许可、数字槽位和协议/事务约束保持。

- TOKEN 使用完整 TKVT wire（含 EOS）：10 + 4×生成 Token 数。
- KV 使用完整 TKVI envelope + metadata + content TKVT + TKVK；所有头、摘要和封装均计入预测。
- 预测使用相同冻结 R/L，数据口径都是各自完整模型通信协议 bytes，不混用 Token 内容字节与 KV 应用字节。
- 选中路径执行正式 handoff 后记录 actual bytes，以及 `wire_prediction_error_bytes=actual−predicted`；来源保留 `wire_estimate_method`。
- 实测本地交接 `observed_local_ms`、内部阶段、模拟网络耗时、Profile 预测、Synthetic 假设和实际整场运行 wall time（created_at→updated_at，含框架开销）各有独立语义，不将组合成本称为物理端到端测量。

8.5 generalized Profile 继续按 N（内容 Token）、H（Receiver 历史）、S（Sender 历史）完整角点做有界插值，并验证模型/revision/tokenizer/config/dtype、硬件/实现、KV 策略、termination/pending/framing/contract、误差与适用域。精确旧 Profile 的 workload SHA 规则也不放宽。Shannon 改变网络输入，不改本地测量口径或 Token/KV 算法。

没有合法 Profile 时，质量优先保留 TOKEN，本地/总成本为 null、明确未校准；不填零，不声称比较过两条性能。3B 未校准，禁止套用 0.5B 数据。理想化开关继续默认关闭，只能显式性能实验启用；假设公式仍为 `TOKEN=20+2N+0.002NH ms`、`KV=5+0.05N+0.2k ms`，会执行真实协议，但预测不是实测性能。

## 6. 临界条件与参数探索

AUTO 决策新增 `crossover`，只取当前已通过合法性检查的两个成本与完整 wire 长度。若任一路被排除或没有本地成本，显示不可分析。

```text
R*_Mbps = 8(D_K−D_T) / [1000(P_T_ms−P_K_ms)]
SNR*_dB = 10 log10(expm1(ln(2) R*_Mbps/(ηB_MHz)))
```

仅当两个差值同号、均非零时给出正的有限交叉点，保存两边分别偏向哪个策略；相等/优势支配/不可表示条件均不伪造阈值。指数逆解使用稳定的大值分支。它是当前消息与历史的点条件，Profile 误差和质量门槛仍可保持 TOKEN。

通信证据里的“临界条件预分析”允许探索不同 B，仅改变分析显示，不修改 Session 或 Selector。标明是否位于当前 μ±2σ 和可选截断界内；μ±2σ 不是保证边界。

课堂候选：本机缓存 **Qwen2.5-0.5B、BF16/MPS**，4 轮，每次 24 Tokens，context4096；B=10MHz、μ=10dB、σ=6dB、ρ=.85、seed42、η=.8、L=0。**仅 Synthetic 理想化成本场景**出现本次实际变化。消息内容 N 为 16/24，H 随讨论增长，各次 R* 约27.07–35.94Mbps；#5 的 R*=28.6643Mbps、SNR*=10.407dB，实际 SNR=10.442dB、R=28.747Mbps，选择真实 KV V0。#6 衰落至5.643dB/R17.779Mbps，真实回到 TOKEN。没有按序号指定模式。

这组条件不保证其他模型、消息或真实 Profile 产生切换；实测 Profile 下没有合法交叉点应如实保持某一方案。当前没有新的 3B 或一般性能交叉区间结论。

## 7. Dashboard 与旧分布式记录

正式标题栏删除架构下拉和其留白，保留配置/新实验、TOKEN/V0–V3/AUTO、0.5B/1.5B/3B 模型菜单、展示/调试模式及最终答案。

网络配置核心展示 B/μ/σ/L，ρ、seed、η 和可选截断界渐进展开。底部轻量展示当前 SNR/C/R/revision。网络详情内新增 SNR/C/R 可切换曲线、当前 handoff 标记、版本分段和实际策略表；详细原始数据在原抽屉，未新建第二套日志系统。

1360px 容器上限、850px 正文、18px/32.4px 和内部滚动保持。1440×900、1920×1080 的完整 DOM 尺寸验证无整页横纵溢出。1440 截图是完整画布；1920 视口截图由浏览器采集器右裁为1728×1080，另一个临界分析图被裁为1440×810，文件名如实标记，不称为完整1920截图。原 Distributed URL 仍走独立命名空间只读恢复；新建被禁用并提供“返回本地实验”，不会与 Local 相同 ID 串线。后端分布式 API 和测试继续保留。

## 8. 轮数、上下文与资源

空白草稿默认4轮，普通1–10轮，勾选高级后最多20轮（请求需 `advanced_rounds=true`）；Runtime、API、Mixed 的 rounds×2 计划及 CLI 同步支持上限20。旧分布式入口保持其1–3轮约束。

对超过3轮的 Local 讨论，创建前按真实 tokenizer 估算双方 system/question、每个阶段指令、发送/接收 framing、双方全部生成上限和最终总结；超过模型/用户上下文上限则拒绝创建，不占执行槽。默认新草稿 context8192，可调128–32768；历史原值保留。实际上下文还保留每阶段 reserve/截断/失败保护，不丢历史。MPS 提前估算五倍事务 Cache 峰值与可用内存，critical 压力或不足时拒绝；运行中安全生成边界遇 critical 或相对基线 Swap 增长>256MiB 则 resource_abort，保留已完成轮次和诊断，不改 OS/MPS 限制。

20轮由受控工程模型和真实随机 tiny-Qwen2 CPU Cache/Decode 测试覆盖；没有声称跑完20轮3B。

## 9. 本地操作

当前 8000 后端和 5173 前端保留可用。不要在已占端口重复启动。若服务已停止，可分别运行：

```bash
# 项目根目录；只使用已有本地权重，禁止下载或 CPU fallback。
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python run_dashboard.py \
  --discussion --device mps --host 127.0.0.1 --port 8000 \
  --model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# 另一个终端，已有前端依赖：
cd frontend && npm run dev -- --strictPort
```

打开本地 Dashboard，配置任务轮数与小预算；网络选 Shannon，设置 B/SNR；通信选 AUTO。质量优先无需 Profile；仅在明确理解假设时使用性能实验+理想化开关。点具体交付标签查看两方案成本和临界条件，详情→网络轨迹回看全部样本。

独立 CLI 示例（会另加载模型，运行前先评估空闲资源，避免与已有服务重复占内存）：

```bash
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python run_discussion_demo.py \
  --device mps --rounds 4 --max-new-tokens 24 --max-context-tokens 4096 --mode auto \
  --network-mode shannon \
  --channel-json '{"bandwidth_mhz":10,"snr_mean_db":10,"snr_sigma_db":6,"correlation":0.85,"seed":42,"efficiency":0.8}' \
  --output benchmark-results/shannon-new.json
```

以上 CLI 默认质量优先；添加 `--auto-policy performance_experiment --idealized-simulation` 才使用显式 Synthetic 成本。没有启动服务器/SSH或无线进程的步骤。

## 10. 真实验收与证据

正式 React 页面创建两场 Qwen0.5B/MPS/BF16 实验，每场4轮、9次生成、8次真实交接。共享模型权重，两 Agent 保持独立 Session 状态/ledger/Cache。两场同 seed、完整配置的8点 SNR 轨迹逐点相等。质量场8 TOKEN；Synthetic 场7 TOKEN/1 KV V0，KV Receiver 内容 Prefill=0、后续真实 Decode，总计15 TOKEN/1 KV。16次预测包长误差均0。

18条完整生成和终止原因、所有字节/成本/快照/接收与事务结果保留。模型多次输出不确定、复述或预算截断；不当作语义质量认证。

107次外部资源采样，pressure最高1，基线Swap3606.62MiB、本轮增长0。单一0.5B owner；无1.5B/3B额外推理。为载入新代码正常停止空闲自有API/模型worker，三条原 completed/cache_released 元数据已备份并逐字段恢复核对；只恢复展示记录，不恢复 Cache。原研究原始数据未改。最终软件更新再次正常重启时同样保存当前记录。

小型证据：`audit.json`（原研发资料，未随公开副本提供：`shannon-90/evidence/audit.json`）、两份完整 Session JSON、1440/1920截图。大日志/资源/重启备份：忽略目录 `benchmark-results/shannon-90`。同 seed、同参数与推进顺序重现网络过程；LLM输出不因 seed 而保证跨硬件/后端版本相同。

## 11. 工程验证

最终统计以 `validation.json`（原研发资料，未随公开副本提供：`shannon-90/evidence/validation.json`） 为准，含全量Python、前端、TS/Vite、编译与diff检查。新增数学、极值、随机统计、同seed/异seed、Session隔离、并发重复冻结、PATCH分段、广义Profile与错模型、真实tiny CPU TOKEN/KV路径、1/4/10/20轮和创建前/运行中保护测试。前端覆盖MHz参数、标准差/边界、更新不发送独立Mbps、曲线、临界探索、无阈值、20轮高级许可及旧Distributed只读隔离。

测试失败原始日志保留；旧轮数/架构控件断言按新需求更新，没有放宽算法、Profile或协议断言。验收后任务范围自动commit，不push。

## 12. 未完成的研究

3B合法实测Profile及独立holdout、真实测量成本下稳定策略交叉、普遍KV语义/质量收益尚未完成。η、AR(1)、固定额外延迟是教学假设，不包含调制编码、频率选择性、干扰、重传、半双工调度或无线硬件开销。没有时钟连续采样、实时时间仿真或真实网络限速。原分布式/无线历史实现保留作研究材料，不是9.0正式演示链路。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

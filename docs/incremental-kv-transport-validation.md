# 7.6A Mac/Qwen/MPS 增量消息 KV 验收

日期：2026-10-10（Asia/Shanghai）。起始 main，HEAD `fe888eef1909a2365db00acd679fc011b21ea5a5`，工作树干净；Git 确认 7.5B 已提交。只验收发送/bytes/解包，未完成 KV 讨论。

模型 Qwen/Qwen2.5-0.5B-Instruct，共享一份模型权重，双方独立 Cache；MPS/BF16，24 层、2 KV heads、head_dim=64，raw KV 每内容 Token 12,288 B。原 TOKEN 讨论两轮和 A 总结正常完成，模型答案为 71；最终 A/B Cache=220/185。

## 四次双向消息

| 消息 | 原始/内容 Tokens | Decode 时/提取时 Cache | Sender Delta span | TOKEN wire B | Delta raw B | KV v1 wire B | Envelope total B |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| A:1 | 16/15 | 89/91 | [74,89) | 74 | 184,320 | 186,601 | 187,641 |
| B:1 | 8/7 | 124/126 | [117,124) | 42 | 86,016 | 88,297 | 89,304 |
| A:2 | 12/11 | 151/153 | [140,151) | 58 | 135,168 | 137,449 | 138,474 |
| B:2 | 8/7 | 183/185 | [176,183) | 42 | 86,016 | 88,297 | 89,304 |

四条均 termination=eos，raw 最后 ID=151645（EOT）。全部内容在 Decode 中已缓存，增量出口补算 0 Token / 0 ms；原 TOKEN 路径把 EOT 和本地普通 closing 尾部补入 Sender Cache。提取排除 EOT/尾部，并排除 assistant header、阶段指令和全部旧历史。所有层 seq_len 分别 15/7/11/7，而 Sender 总 Cache 增长到 91/126/153/185；没有携带完整工作 KV。

四包通过独立 InMemoryByteTransport，累计 504,723 B；每包包含真实 content IDs、合约/位置标识、v1 Tensor metadata 和 checksum，不能用 Delta raw bytes 代替通信量。每条 `decoded == source envelope`，全部 K/V dtype、shape 和 raw bytes 位级一致；同时 Receiver 实际 TOKEN ledger 内容 IDs exact。KV Receiver append=0，Sender/Receiver 的状态对象在旁路前后相同。

## 精确生成文本与 IDs

### A:1

The answer to 23 + 48 is 71.

raw IDs: `[785, 4226, 311, 220, 17, 18, 488, 220, 19, 23, 374, 220, 22, 16, 13, 151645]`

content IDs: `[785, 4226, 311, 220, 17, 18, 488, 220, 19, 23, 374, 220, 22, 16, 13]`

### B:1

Agent A's calculation is correct.

raw IDs: `[16810, 362, 594, 21937, 374, 4396, 13, 151645]`

content IDs: `[16810, 362, 594, 21937, 374, 4396, 13]`

### A:2

Agent B's feedback confirms that the calculation is correct.

raw IDs: `[16810, 425, 594, 11055, 42396, 429, 279, 21937, 374, 4396, 13, 151645]`

content IDs: `[16810, 425, 594, 11055, 42396, 429, 279, 21937, 374, 4396, 13]`

### B:2

Agent A's calculation is correct.

raw IDs: `[16810, 362, 594, 21937, 374, 4396, 13, 151645]`

content IDs: `[16810, 362, 594, 21937, 374, 4396, 13]`

## 轻量单次诊断

| 消息 | 提取含 host copy ms | Pending ms | KV serialize ms | Envelope serialize ms | 完整 deserialize ms | 嵌套 content TOKEN B | Envelope overhead B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A:1 | 15.715 | 0.000 | 0.066 | 0.243 | 0.271 | 70 | 970 |
| B:1 | 12.363 | 0.000 | 0.061 | 0.212 | 0.241 | 38 | 969 |
| A:2 | 14.325 | 0.000 | 0.057 | 0.205 | 0.226 | 54 | 971 |
| B:2 | 16.016 | 0.000 | 0.052 | 0.179 | 0.200 | 38 | 969 |

均为单次本地诊断，不是最终 benchmark，不据此声明 KV 比 TOKEN 快。Envelope serialize 包含 KV serialize 子阶段；完整 deserialize 包含 envelope/checksum/嵌套 TOKEN 和 KV parse，不叠加计费。初始化 actual weights fingerprint 的一次性扫描另在这些逐消息计时之外。

## 真实 pending 内容补齐

另一个独立真实 MPS 用例只生成 2 个内容 IDs `[785, 4226]`，termination=max_new_tokens，span=[74,76)，原 Cache=75，已有 1/2 个内容位置。出口在私有 clone 上只 forward `[4226]` 一次，补算 1 Token / 14.577 ms。Delta=2 Tokens / 24,576 raw B，KV v1=26,857 B，envelope=27,852 B，全部 K/V round-trip exact。原 A/B Cache、状态对象和 pending 保持不变。角色初始化之后禁止调用 backend.prefill，实际调用数为 0，没有 Clean Prefill(M) 或 Receiver KV Prefill。此用例不再执行 TOKEN handoff，避免把同一 pending 在另一副本重复补算作为验收路径。

机器证据保留在 gitignored `benchmark-results/incremental-kv-transport-add.json` 和 `incremental-kv-pending-mps.json`，权重与大二进制载荷未写入仓库。

## 回归与限制

新增标准库协议/损坏/兼容校验和真实 tiny Qwen2 CPU/受控预测测试。覆盖正常 Decode、不同 EOS/EOT、截断 pending、无 Clean Prefill(M)、仅新 span、多方向多轮、全部 dtype/shape、bytes 传输、原始位级 oracle、资源限制、重复序号、实际不同权重/RoPE config 拒绝、Agent Cache 与接收序号失败隔离，以及旁路与原讨论的完整结果/ledger/最终 KV 相同。原 7.2A 研究测试和 Canonical TOKEN/KV 回归保留。

最终测试结果由 `.agent/TASK.md` 和本页末尾记录。V0/V1 Receiver 位置目标、追加和质量/混合讨论尚未实现；V2/V3 未实现。仍限 dense batch-one、同构不可变模型、append-compatible framing；空内容增量 KV 明确拒绝；无 Socket/真实网络，无 Llama/CUDA 实测。已具备进入 7.6B 的协议与出口基础，不能标记完整 KV 讨论成功。


最终检查：`HF_HUB_OFFLINE=1 .venv/bin/pytest`：**440 passed**（新增64），2条既有FastAPI/Starlette上游弃用警告；`compileall -q src tests *.py` 通过；全部25个 `run_*.py --help` 通过；`git diff --check` 通过。最终 main/HEAD 仍为 `fe888ee`，12个已跟踪文件修改、9个新文件未提交；没有自动commit。

最终schema严格校验修订后的CLI复验仍4/4成功；与首测完整discussion结果、IDs、spans及包长完全相同。复验JSON：gitignored `benchmark-results/incremental-kv-transport-final.json`。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

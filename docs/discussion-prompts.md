# 7.7B 通用讨论 Prompt 与覆盖规则

## 修改前审计（main / 3f6d320）

7.7 已独立提交，起始工作树干净；`.agent` 的“未提交”描述过时。本轮不混合 7.7 内容。

真实来源是 `discussion.py`，不是前端或 KV repair。初始化 `LLMAgent(system_prompt=...)` 后，`ChatFraming` 用加载的 tokenizer 对 system role 执行 `apply_chat_template`，核对 tokenize/text 一致、边界可追加，再由 `initialize_agent_state()` Prefill 一次。两端 ledger/KV 独立。

旧指令完整如下：

| 来源 | 实际文字 |
| --- | --- |
| A System Prompt | You are Agent A, a solver. Solve the user's arithmetic question. Be concise. Consider feedback from Agent B and correct mistakes when necessary. Do not invent disagreements. |
| B System Prompt | You are Agent B, a reviewer. Independently check Agent A's calculation for the user's original arithmetic question. If it is correct, say so. If it is wrong, identify and correct the error. Be concise. |
| A 第一轮本地阶段指令 | Solve the original question and send your initial calculation to Agent B. |
| B 第一轮检查指令 | Independently check Agent A's latest calculation above against the original question. Reply briefly to Agent A. |
| A 后续本地阶段指令 | Consider Agent B's feedback above. Confirm or correct your calculation of the original question. Reply briefly to Agent B. |
| B 后续本地阶段指令 | 与 B 第一轮相同，没有区分更新是否解决之前问题。 |
| 总结本地阶段指令 | Give the final answer to the original arithmetic question. Use the discussion above to check your result. Keep it concise. |

原题分别通过 `append_local_user` 追加；peer 是带 `Message from A to B:\n` / `Message from B to A:\n` 标签的 user turn，内容使用传来的原始 IDs。每轮 `generate_message(local_prompt=...)` 把新的本地阶段指令作为 user turn 增量 forward，再追加 assistant prefix 做 cached Decode；历史不重复 Prefill。总结 `generate_local_message` 使用指定端同一历史和 Cache，输出由模型产生。

Qwen 实测模板（新旧走相同 adapter）为：

```text
<|im_start|>system
{该 Agent 的实际 System Prompt}<|im_end|>
<|im_start|>user
{原题}<|im_end|>
<|im_start|>user
Message from A to B:
{收到的 A 内容 IDs}<|im_end|>
<|im_start|>user
{B 当前阶段指令}<|im_end|>
<|im_start|>assistant
```

A 第一轮没有 peer turn；后续阶段只在已有历史后追加。System Prompt 与本地阶段指令是两个不同来源，不把阶段指令算作 System Prompt。Qwen 不走 `LLMAgent.generate_reply()` 的普通拼接 prompt，也不调用 `model.generate()`，使用原 raw-logit greedy continuation。

明确问题：所有问题被角色、阶段和总结重复要求按算术处理；A 后续要求 confirm 容易诱发重复；B 正确即确认但没有要求给出核查依据。`calculation is correct` 不是 Python 固定输出，是模型在这些约束下产生的回复。固定轮次仍执行全部交付，是当前实验语义，不是本轮提前结束的依据。真实全 TOKEN 基线复现了地理错误与固定算术措辞，故不能主要归因于 KV。改用通用指令后仍有事实错误、盲目确认、错读历史、语言不遵循与重复，证明 Prompt 修复不能代替 Qwen 0.5B 能力验收。

## 当前权威定义与默认值

`src/token_kv_agent_system/discussion_prompts.py` 是唯一默认定义与解析来源。`SOLVER_PROMPT` / `REVIEWER_PROMPT` / `FINAL_PROMPT` 仅保留兼容别名。默认/阶段版本为 `general-v1`，实际文字随 Session 完整记录，不仅记录版本。

Agent A 默认：

```text
You are Agent A. Answer the user's question with a brief reason. Stay on topic and use the language requested by the user. Say when you are unsure. Read Agent B's feedback and fix errors if needed. Avoid repeating confirmations; give a useful answer.
```

Agent B 默认：

```text
You are Agent B. Independently review Agent A's answer to the user's question. Check relevance, facts, logic and any calculations. Point out specific errors with a reason. If correct, give a brief check or reason. If unsure, say so. Do not blindly agree or invent objections. Use the language requested by the user.
```

| 阶段 | 实际本地指令 |
| --- | --- |
| A 第一轮 | Answer the original user question with a brief reason. Use the requested language. |
| B 第一轮 | Review Agent A's answer above against the original user question. Check it yourself. Explain any specific error or your check; say if unsure. Use the requested language. |
| A 后续 | Read Agent B's actual feedback above. Correct your answer if needed. If there is no new issue, give a concise useful answer to the original question without repeating agreement. Use the requested language. |
| B 后续 | Check Agent A's updated answer above against the original question and prior feedback. Does it resolve the issues? Give a specific check or remaining error; say if unsure. Avoid repeating agreement. Use the requested language. |
| 总结 | Answer the original user question using your discussion history above. Give the answer and a brief reason, including any remaining uncertainty. Use the requested language. |

无测试题标准答案，无检索/工具，无人格规划框架。仍按 1–3 完整 A/B 轮运行，再指定 A/B cached 总结。

## 解析、预览与固定 Session

`agent_a_system_prompt` / `agent_b_system_prompt` 各自可省略、null、空串或纯空白，使用对应后端默认。非空文本原样保留（不 trim 有效自定义内容），来源 `custom`；另一个 Agent 不受影响。每个最多 4000 个 Python Unicode 字符；非字符串、超长、非法控制字符/孤立 surrogate、当前 tokenizer 的特殊控制 token 返回明确 422，不预留活动槽或 Prefill。长度上限不保证所有文字都能放入 4096 Token 完整讨论预算；实际预算超限按原 Runtime 规则 failed，绝不裁剪或重新 Prefill 历史。

`POST /api/discussions/prompts` 只接收这两个字段，共用 Runtime 的 `resolve_prompts()`，返回 `agent_a/agent_b` 的 `text/source` 与 `version/stages`。无模型生成、无 Cache，未加载模型也能返回默认定义；有模型时校验其特殊控制 token。Dashboard 从此 endpoint 获取预览，前端没有默认 Prompt 文本副本。输入变更后短暂等待服务器预览，并丢弃过期响应；预览错误清楚展示并禁止开始。

启动请求 `POST /api/discussions` 同样接收这两个可选字段。预留 Session 时存实际解析结果；Runtime 构造时解析并冻结，再实际初始化 system Cache。Prompt 配置是 frozen value object；已有 Runtime 拒绝替换 prompt 字段，运行时整个配置锁定。没有 PATCH 修改 Session 的入口，同 ID 不能重新启动。新值必须新建 Session，不能只替换字符串并沿用旧 Cache。

Session REST/结果保存：原始 request `config`、`prompt_config`（A/B 实际 text/source、版本、所有阶段文字）、model/device/dtype、原题、轮数、Token 上限、策略/plan、全部实际发言、handoff trace、总结、初始/最终 Cache 长度和失败状态。Runtime CLI 结果另有双方 `models` 身份（不把可变加载耗时当模型配置），可保存实际 generated/received IDs。启动和终态 WS 也含 `prompt_config`，使用既有 collector/broadcaster；HTTP/WS 无 KV/ledger 内容。

刷新用 `GET /api/discussions/latest` 或 `GET /api/discussions/{session_id}` 恢复该次的实际 Prompt。UI 将“新 Session 实际生效 Prompt 预览”与“该 Session 实际 Prompt 配置（固定）”分开展示；编辑新草稿不会改变已完成 Session。默认输入为空；恢复已存在 Session 时可还原其自定义覆盖值，但历史实际配置始终独立保留。最近三个 Session 进程内保存，进程重启不持久化。

## 使用

Dashboard「问题与策略」→ 展开「Agent 提示词配置」→ 单独输入 A 或 B 多行 Prompt。各自恢复按钮只清空本方值；“恢复默认”清空双方值。附近明确标注留空使用默认。运行中编辑器和按钮锁定；阶段指令只读预览。

CLI：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py --device mps \
  --question '中山大学在哪里，用中文回答' \
  --agent-b-system-prompt 'You are Agent B. Review the actual question and answer independently. Say if unsure.' \
  --output benchmark-results/custom-discussion.json
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_prompt_validation.py --device mps
```

所有 KV span/Receiver 位置继续从真实 ledger/cache/framing 长度取得。Token/KV/Envelope、extraction、repair、Receiver append、State/transport、Canonical/Cost Model/Selector 未修改，无 Full Prefill 或 fallback，无提前结束。完整生成文本、实验限制与检查见 7.7B 实验记录（原研发资料，未随公开副本提供：`discussion-prompt-validation.md`）。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

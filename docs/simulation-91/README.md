# 9.1 模拟通信实验配置与顶部 UI

基线：`421427f`（9.0）。验收日期：2026-10-16。

## 配置与计算

入口：配置 → 通信 → AUTO → 性能实验 → 勾选「理想化场景仿真」→「高级：计算成本假设」。默认折叠，支持修改 a/b/c/d/e/f、错误反馈及「恢复默认假设」。仅用于创建新实验；运行中的配置页只显示冻结值，已完成实验的配置页编辑的是下一次实验草稿，不会改变当前 Session。

默认公式完全保留：TOKEN = 20 + 2N + 0.002NH ms；KV = 5 + 0.05N + 0.2k ms。

- N：消息内容 Tokens，不含 EOS；H：接收方已有缓存 Tokens。
- k：Selector 当前候选的 `recomputed_tokens`。V0/V1 为 0；V2 使用现有 `suffix_recompute_count` 的数量/比例取整及截断结果；V3 为合法数字槽位的单 Token 修正（无数字仍排除）。没有将 V2 的 k 套到 V0。
- 系数均为有限非负 JSON 数字，范围 [0, 1e9]，允许 0。拒绝布尔、字符串、null、负数、NaN/Infinity、越界和未知字段。前端也拒绝空输入、非十进制格式及非有限值；禁用提交并显示错误。
- `synthetic_compute_v1`、`source=synthetic_assumption`、`units=ms` 为固定元数据；缺省参数由服务端展开并写入 Session，不依赖浏览器默认值。部分 JSON 系数缺省时沿用原默认值。

实际链路：`networkAutoConfig` → POST `synthetic_assumptions` → `DiscussionStartRequest.auto_config` → 冻结的 `IncrementalAutoConfig.synthetic_assumptions` → `IncrementalSelector.select` → `idealized_prediction` → `SyntheticCostAssumptions.predict`。TOKEN 本地预测使用 a+bN+cNH，KV 使用 d+eN+fk，再加当前 Shannon/固定/轨迹网络的真实协议字节模拟传输成本，仍由原 Selector 比较总成本。没有硬编码 AUTO 结果。

假设只进入 Synthetic 本地预测；真实模型生成、TOKEN Prefill、KV 导出/传输/修正、字节数、本地实测耗时保持原链路。`cost_profile` 与理想化模式互斥；Profile JSON 导入和实测预测保持原样。显式允许全零假设时也保留合法候选，以其网络成本排序；实测 Profile 的正成本检查保持不变。

## 冻结与历史

使用不可变 dataclass 保存运行配置；REST 保存全部系数、schema、来源和单位；`DISCUSSION_STARTED` 保存相同配置。每次 AUTO 决策及通信记录保存实际使用的系数、候选 N/H/k、预测本地成本与独立网络快照，详情中的「本次配置」及 AUTO 说明直接展示当时公式。刷新和选择旧 Session 不会套用当前草稿值。新 Session 不影响旧记录。

系数没有运行中 PATCH 接口；网络 PATCH 的 strict schema 拒绝夹带系数。旧分布式运行配置及 worker 决策入口补充了新 dataclass 的严格反序列化，保留原功能。

沿用 9.0 的历史生命周期：本地服务保存最近三份内存 Session 元数据，刷新可回放；重启服务仍按原行为清空。此次没有引入新的磁盘 Session 存储或恢复 KV Cache 的承诺。保存或导出的完整 Session JSON/事件包含实际系数。

## UI 修复

真实来源：`DiscussionPanel.tsx` 的标题副信息中，行内 `span.risk-note` 命中 `styles.css` 的 `.risk-note { border-left: 2px solid #dfba86; padding-left: 12px; }`，于是 AUTO 后出现竖线与多余间距。移除该行内元素不适用的 risk-note 类，保留其他风险段落的正常提示样式；没有截图坐标补丁或全局隐藏边框。

副标题改为 `3轮交流 · AUTO · 理想化仿真`；Synthetic 来源在 tooltip、配置及决策详情明确标记为假设、非实测 Profile。实测来源显示「实测 Profile」，旧 synthetic/test_only 显示「模拟成本 · 测试」，并防止失活的 Profile 草稿污染理想化来源。

1440×900 正式 React 页面：document 1440×900 无溢出，workspace max-width 1360px，正文实测 850px、18px/32.4px，reader 内部滚动；配置 body 462px、展开后内容 1347px、overflow-y:auto，底部按钮可达。标题子元素 border-left/padding-left 均 0px。展开/折叠、六字段重置、空值阻止提交及刷新回放通过，浏览器无 error/warn。

截图：Dashboard（原研发资料，未随公开副本提供：`dashboard-1440.png`）、折叠设置（原研发资料，未随公开副本提供：`config-collapsed-1440.png`）、展开设置下部及重置（原研发资料，未随公开副本提供：`config-1440.png`）、冻结配置回放（原研发资料，未随公开副本提供：`frozen-1440.png`）。DOM 检查（原研发资料，未随公开副本提供：`ui-checks.json`）。

## 回归与数值验收

自动测试包括默认/自定义公式精确值、六个系数严格数值校验、冻结与恢复、V0/V1/V2/V3 的 k、V2 比例策略、参数驱动 AUTO 选择变化、JSON 序列化往返、REST 历史与事件记录、禁止网络 PATCH 改系数、前端校验/重置/运行中只读、标题与实测来源回归。

浏览器在隔离的 8019/5191 服务使用 random/tiny-qwen2 CPU（受控输出 ID，非预训练模型性能证据），通过正式表单设置 a=31、c=0.004，完成 3 轮、7 次生成、6 次真实 KV V0 交接。全部成功追加，接收方消息 Prefill=0；六次决策按记录系数与实际 N/H/k 重算一致，预测总成本等于本地假设加 Shannon 网络成本。刷新后仍显示原系数。详见 协议与成本证据（原研发资料，未随公开副本提供：`browser-protocol-evidence.json`）。没有运行 3B Profile、加载大型模型或连接外部服务器/设备。

全量结果见 `validation.txt`。既有用户未跟踪资料不纳入本次提交；不 push/reset/clean。原有 8000/5173 服务保持，临时验收服务完成后关闭。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

# 双 Agent 通信实验控制台设计规范

> 9.0（2026-10-16）：课程正式入口改为 Mac Local 真实 LLM + 香农容量 / 动态 SNR 模拟；架构切换 UI 已移除，原分布式实现与历史证据保留。见 [模拟器、参数与验收](docs/shannon-dynamic-channel-simulator.md)。下方服务器/无线与旧轮数描述为历史记录。

> 8.2B更新（2026-10-14）：继续沿用下方确认布局。正式UI已接入手动V0–V3、增量AUTO、manual/trace、Profile来源、候选成本和按message_id的实际交付标签；旧禁用/无模拟成本描述是8.1历史状态。当前完整功能与边界以[8.2B实施记录](docs/dashboard-dynamic-auto-integration.md)为准。原型文件保持原样。


状态：用户已确认阅读优先原型，正式React迁移与真实接口验收完成；2026-10-14。详细结果见实施与验收（原研发资料，未随公开副本提供：`docs/dashboard-ui-implementation/README.md`）。前两轮探索仅为历史参考。

## 目标与使用者

面向研究人员和开发者，让一次实验的配置、执行过程、通信证据与结果质量可以在同一工作区理解。仅面向电脑浏览器，优先深色；用信息架构和渐进展开降低复杂度，而不是删减观测字段。产品语言以中文为主，保留 TOKEN、KV、Prefill、Wire bytes 等精确术语。

## 已核验的事实与功能迁移清单

事实来源为 `frontend/src/App.tsx`、`frontend/src/discussion/DiscussionPanel.tsx`、各自 types/api/state/hooks/test，以及 `src/token_kv_agent_system/dashboard/{app,runtime,state,discussion_service,websocket}.py`。仓库没有 AGENTS.md；`.agent/TASK.md` 等实际使用大写文件名。

| 当前能力 | 重设计后的归属与必须保留的行为 |
| --- | --- |
| Canonical / 双 Agent 讨论两个 Tab，打开后保留状态 | 两个工作区，切换不会重新发起实验，草稿和运行记录保留 |
| auto、observer、interactive、real_token、real_kv、real_adaptive | 通信基准工作区顶栏显示后端实际 Runtime；浏览器切换工作区不切换后端 Runtime |
| Canonical Agent A/B 状态、通信方向、当前模式 | 常驻通信视图，区分当前执行和上次已完成记录；没有事件则显示未知或等待 |
| Canonical 带宽滑块和数值输入、应用、运行一次、下一方向 | 实验控制浮层/紧凑工具栏；支持任意正数带宽，滑块仍是 1–2000 Mbps；运行期间锁定 |
| 基础延迟 | 显示服务端值；当前 PATCH 仅接受 bandwidth_mbps，基础延迟只读 |
| Mock TOKEN/KV 估算与最终选择 | 模式比较区，持续显示 Mock / 估算来源，不能当真实推理结果 |
| 固定真实 Token 成本 | Token count、理论 bytes、wire bytes、serialize/link/deserialize/Prefill/total、identity；完整原始字段保留在详情 |
| 固定 Canonical KV 成本 | raw/body/metadata/wire bytes、Sender prepare、device-host/encode/serialize、deserialize/CPU reconstruct/host-device/cache construct/restore、两种成本边界及 correctness |
| Real Adaptive 预测与单路实际执行 | TOKEN/KV 预测比较、选中路径、Actual、绝对/相对误差、profile/calibration/device；未执行路径明确“未执行” |
| 最近通信、Agent 文本、最多 100 个事件 | 主内容及日志区；完整长文本可阅读，事件可筛选并展开 metadata，筛选不删除数据 |
| 普通讨论问题、轮数、Token 预算、总结 Agent | 讨论配置区；问题 1–512 字符、轮数 1–3、预算整数 1–128、总结 A/B |
| TOKEN、KV V0、KV V1、Mixed | 讨论通信策略；Mixed 逐次计划按 rounds×2 严格 A→B/B→A；V0/V1 实验性标签 |
| V2、V3、Discussion AUTO 尚未实现 | 在策略帮助中说明能力边界，不显示为可启动选项 |
| A/B 独立 System Prompt、各自/同时恢复默认 | 配置中的 Agent 子面板；每个最多 4000 字符，空白用服务端默认，恢复仅改新实验草稿 |
| 服务器 Prompt 预览、阶段指令、历史 Session 固定 Prompt | 预览与会话证据分开，保持 debounce、过期响应拒绝和服务器校验 |
| secret_sum 场景、两个秘密数字、独立评分 | 讨论场景切换；1–99 整数、固定一轮、A 总结、不允许自定义私有 Prompt；保留普通讨论草稿 |
| 逐轮完整发言、结束原因、handoff 详情 | 讨论主时间流；保留 source/target、requested/executed mode、strategy、Token/span/cache/M Prefill/position/repair/一致性/error/fallback |
| Final Answer、摘要、Cache 计数与释放状态 | 中央最终答案与观测区；工程成功、信息交换和答案正确分别表达；null 为无法判定 |
| REST 恢复、WebSocket 事件、revision 校准 | 保留实际 Session ID、revision 防倒退、缺口刷新、断线 REST 恢复；重连不 POST |
| Session 记录仅进程内、最近三次 | 不宣称持久化历史；当前 UI 查询 latest，历史列表或跨进程检索不作为本轮承诺 |

### 必须遵守的能力边界

- Dashboard 只消费 Control Plane 元数据，不接触或转发 Token IDs、packet、tensor 或 DynamicCache。
- Canonical clean-Prefill KV 与讨论增量 V0/V1 语义不同，不共用一个“KV 策略”控件。
- `--discussion` 的 Canonical 是独立 Mock Runtime。网络滑块和 Selector 不参与 Discussion，不把两套观测混到一起。
- 讨论使用 InMemoryByteTransport，现有 trace 没有通信延迟数据。因此讨论观测展示真实 wire bytes、handoff 数、cache、执行阶段；延迟不绘图，不借用 Canonical 数值，不使用前端动画时长冒充传输耗时。
- 后端没有暂停、取消、重置活动实验接口；不提供这些操作。新建实验用现有 POST；“恢复默认”只恢复草稿，不能伪装清空后端 Cache 或历史。
- Runtime、模型、设备和 Canonical initial prompt 由服务端启动确定；不展示无效的浏览器模型选择或热切换。
- 模型文本按完成的发言逐条显示；当前没有逐 Token streaming，不能做伪打字流。
- 工程验证通过不等于模型理解正确。V0/V1 不承诺数值等价或加速，不更改实验默认 Prompt 来迎合界面演示。

## 第三轮：以舒适阅读为中心

只面向电脑浏览器，目标1920×1080、1440×900、1280×800。前两轮原型作为已否决历史保留，不作为实施依据。用户明确撤销“所有信息同时可见”和“为了避免滚动缩小字体”的目标。

主界面只突出发起实验、观察双Agent通信、阅读讨论/最终答案。以A的连续阅读为基础，采用一个安静的阅读表面，配置浮层和实验详情抽屉保存完整功能。html/body固定视口高度，长讨论在阅读区局部滚动；允许按需切换，指标、通信证据和日志不常驻。不得为了填满空间增加面板。

参考图的具体视觉分析、采样颜色与比例规范先写入 第三轮参考提炼（原研发资料，未随公开副本提供：`docs/dashboard-reading-prototype/REFERENCE.md`），再制作原型。建议令牌：中性背景#1B1C1F、主表面#232427、展开表面#2E2F33，边框#3A3B40。标题28px，正文18px/1.8，Agent标题17px，表单/按钮16px，辅助14px；主要按钮44px高。正文最大900px，阅读内边距32–40px，发言间36px。点阵仅用于简洁双Agent通道，不能蔓延到正文或变成复杂节点画布。

只围合大模块，发言不再使用独立卡框、微型状态标签或反复分隔线；少量状态色表示结果。主动作浅灰，Agent身份中性。细节层中的复杂字段仍完整保留，工程成功、语义质量与成本来源仍分开表达。

运行前提供任务输入与配置/开始；运行中只显示真实完成的发言与当前阶段、不提供伪streaming；完成后“讨论”和“最终答案”切换同一阅读区。新实验草稿与冻结会话配置独立。原型持久标注设计数据，不接后端。

已按用户确认将docs/dashboard-reading-prototype/迁移至正式React前端。该目录保留确认时的设计基准，正式入口为frontend/，不载入原型脚本或设计数据。

## 桌面实际窗口补充规范

主面板最大1360px；1280宽窗口左右24px，1440宽窗口左右40px，更宽时居中。阅读列900px含身份34px与间隔16px，讨论正文与最终答案实际850px，字号18px/1.8。通信舞台与外层面板加宽，文字行长不随外层无限增加。

100dvh仅代表浏览器可用内容高度，不以显示器screen.height布局。上下留白18–28px、标题80–96px、通信舞台108–128px依可用高度调整，控件和正文不缩小；浏览器页面无整体滚动，讨论与浮层独立滚动。详细的原生Chrome窗口对照、菜单栏/工具栏扣除与测量限制见桌面实测（原研发资料，未随公开副本提供：`docs/dashboard-reading-prototype/DESKTOP-LAYOUT.md`）。正式前端已采用该规范，不新增常驻面板。

## 组件与交互规则

- **WorkspaceShell / WorkspaceNav**：一个 h1；工作区按钮明确选中；切换保留草稿和运行结果。连接状态与模型可用性独立显示。
- **ExperimentControlPanel**：相关字段分组，只有一个主执行动作；运行、提交、恢复期间防重复；字段错误紧邻输入，服务错误保留草稿；网络未应用值明确标识，避免误认为已生效。
- **AgentConfig**：A/B 独立编辑，默认/自定义来源可见；“草稿预览”与“本次实际配置”分开；SecretSum 保留其隔离规则。
- **CommunicationStage**：A/B 名称、方向、当前阶段和模式同屏。进行中只能表示“等待确认”，成功必须来自 finish/receipt。当前任务进行时，上轮结果明确标“上一轮”。
- **DiscussionTranscript**：按真实 sequence 排列完整发言；handoff 插在相应发言之后；读旧内容时不抢滚动，底部跟随模式提供“查看最新”；没有输出、预算终止和失败分别说明。
- **MetricSummary / MetricDetails**：主阅读界面不常驻指标；指标与全量现有字段保留在实验详情抽屉。每个值附单位与来源：实测、模拟、估算、派生。null/undefined/非有限数用“—”，缺失不是 0。
- **CostComparison / CostBreakdown**：Canonical 使用后端可用的成本分量画横向条形/堆叠条，原始数值表同步提供。父级 composite stage 与子分量不重复加总。预测与实际分开，未执行路径无实际柱。
- **CacheSummary**：讨论展示 A/B 实际 Token 计数与释放状态；不绘制虚构容量/占用百分比，没有服务端上限时只呈现绝对计数。
- **EventLog**：Canonical 与 Session 日志按类型及 Session 归属隔离，支持筛选、显示更新时间、展开 metadata；可暂停前端日志跟随，但名称必须“停止跟随”，不称“暂停实验”。仅保留后端有界事件窗口，不宣称完整历史。
- **EvidenceDetails**：span、M Prefill、repair、requested/executed、失败阶段、Prompt、profile 和 correctness 字段全部可查看；无 fallback 不等于成功。
- **ResultPanel**：原始模型文本优先；工程执行、信息交换、答案质量分别显示。后端尚未提供语义判定的普通讨论，不生成自创成功评分。

加载时 skeleton 只代表结构，不出现示例结果；空状态指出可执行的下一步；503 明确模型未加载，409 明确活动实验冲突，422 说明配置问题。断线仍保留最后收到的数据并标记状态待同步，重连恢复只 GET。

## 动画与性能

CSS 负责 120–180ms 控件反馈与展开；通信阶段转换最多 240ms、只动画 transform/opacity。只有真实 start/finish 事件触发路径反馈，没有测量进度时使用阶段指示，绝不生成百分比或以路径运动速度表现真实带宽。

如事件路径需要协调动画，局部引入 GSAP + @gsap/react，ref scope、context cleanup、contextSafe/revertOnUpdate；后台工作区和 document hidden 时停止，尊重 prefers-reduced-motion。不引入 ScrollTrigger、ScrollSmoother、SplitText 等插件。

保留事件有界窗口（当前前端 100），避免每条 WS 事件全量重建大型图表；讨论本身最多 3 轮/6 次 handoff，无需虚拟化库。考虑共享一个 event stream 分发两个工作区，但必须保留原 reducer/revision/恢复契约。网络重连须清理旧 socket、retry timer 和 online/offline listeners，避免多个订阅叠加。

## 技能与依赖决策

已检查会话现有技能与全局技能目录，没有同名 UI/UX / GSAP 技能。仅安装以下四项到 `.agents/skills/`，固定上游 commit，来源/文件哈希见该目录 README 和 provenance.json；未执行第三方脚本。新技能在下一轮可自动发现，本轮已人工阅读并用于提案。

| 技能 | 选择理由 |
| --- | --- |
| typeui-fundamentals | 分组、信息层级、控件状态、排版和可访问性基础；仅按本项目设计规范映射 |
| frontend-design | 明确产品特征和页面组织，审查通用模板化倾向 |
| gsap-react | 如引入 GSAP 时保证 React 生命周期与 cleanup 正确 |
| gsap-performance | transform/opacity、读写批处理和停止不可见动画 |

研究了 awesome-design-skills 的 Enterprise 和 awesome-design-md 的 Linear/Sentry 参考；不安装风格包。Enterprise 仅参考工作台层级，Linear 文件主要针对营销页面，Sentry 参考含营销装饰，均不整体照搬。TypeUI 的大标题/固定章节间距/营销 heading 规则与控制台不适配时，以本文具体组件和语义层级为准；可访问性阈值仍需实测。

运行依赖保持React 19 + TypeScript + Vite及原有React DOM。少量图标沿用原型的内联SVG，未增加lucide、GSAP、UI framework、图表库或全局状态库。真实交接状态只使用轻量CSS opacity反馈，尊重reduced motion。

## 实施与验收

已完成五阶段迁移。正式工作区默认双Agent讨论，Canonical按需打开且保留状态；配置与证据为原生dialog，主阅读区固定视口内部滚动。发言接收来源使用前一次实际receipt交叉核对，不以MIXED或请求策略代替实际结果；首条发言与未知/失败/回退分别标记。最终总结仅显示其上下文最近实际接收，不构造额外交接。

113项前端测试、TypeScript/Vite构建、763 Python测试（1既有可选skip）、compileall与diff检查通过。真实Qwen/MPS完成TOKEN/V0/V1/MIXED、私有数字与三轮长文本；真实WS断开后恢复相同Session，未重新启动。Canonical交互/自动/observer、固定真实Token/KV和Adaptive TOKEN已验收。详细尺寸、来源与限制见实施记录（原研发资料，未随公开副本提供：`docs/dashboard-ui-implementation/README.md`）及validation.json。

原型历史截图均标为设计数据；正式前端仅来自现有REST/WS，不接原型数据。V2/V3/Discussion AUTO仍按后端能力禁用。当前任务按项目约定测试通过后自动提交，不包含来源不明副本或已否决的前两轮探索。

## 参考来源

- [GSAP 官方技能](https://github.com/greensock/gsap-skills/tree/aed9cfd3277740755f6bfc1155c7aa645403b760)：React 生命周期、动画性能。
- [TypeUI](https://github.com/bergside/typeui/tree/2a977f1f6616ae8a5ea84a478ca35601c67f4322/skills/fundamentals)：UI/UX 基础原则。
- [Anthropic frontend-design](https://github.com/anthropics/skills/tree/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills/frontend-design)：设计意图与自审。
- [awesome-design-skills Enterprise](https://github.com/bergside/awesome-design-skills/tree/f631a09b4fcc0166f2e2c1a8c81906ef680c57e8/skills/enterprise)：Dashboard 层级参考，不安装。
- [awesome-design-md Linear](https://github.com/VoltAgent/awesome-design-md/tree/c7c1bada4eda972236d124e06ee58df129a9266b/design-md/linear.app)：深色 surface 层次参考，不照搬营销布局。
- [awesome-design-md Sentry](https://github.com/VoltAgent/awesome-design-md/tree/c7c1bada4eda972236d124e06ee58df129a9266b/design-md/sentry)：审阅后排除其营销插画与装饰风格。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

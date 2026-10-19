# 9.2 通信模式切换时间线

2026-10-19。延续正式阅读布局，仅增加真实交接记录的可视化；没有新增图表依赖、后台采样器或 Session 状态。

## 位置与语义

- 主页面 `CommunicationStage` 中央「最近交接」下方：300px 以内、约72px高的状态组件，最近8次已确认交付。
- 实验详情 → 网络轨迹：全部已确认交付，位于 Shannon 曲线之前；超过16点时仅图形容器横向滚动，保留节点间距。
- 上轨 KV，下轨 Token；阶梯线表示离散模式；A为圆点，B为菱形，未知发送方为方块，最新已确认交接加外环。横轴明确标注「交接序号」，不解释为连续秒数。历史缺号使用等间距类别轴并保留真实序号，避免节点挤在一起。

数据直接读取 `DiscussionSession.handoffs`：`sequence/source/target/executed_mode/executed_kv_strategy/append_succeeded/error`。复用 `receiverCommitted` 判断交付，旧分布式记录沿用严格 receipt 验证。仅有 requested/selected/AUTO 决策、失败、未确认或未知模式的记录不绘制为成功交接；不根据全局 Session 模式猜测结果。

原 `projectDiscussion` 的 HANDOFF_COMPLETED 投影增加实际记录，组件随 Session 更新；REST 恢复/刷新读取同一数组，没有平行历史或独立时钟。完整图与迷你图共享投影和绘制代码。

鼠标悬停、键盘聚焦、Enter/Space/点击均可显示详情：交接序号、方向、实际模式与KV版本、`predicted_local_ms/predicted_total_ms/observed_local_ms/simulated_transfer_ms`；可用时读取实际模式对应的 `auto_decision.candidate_estimates` 及 `network`（旧记录可回退到同次 AUTO 的 network）中的 SNR/C/R。0保留，缺值/非有限值隐藏；KV缺版本明确未确认。成本区分预测、模拟与实测。提示可用关闭按钮或Escape关闭；详情中的第一次Escape不关闭抽屉。

## 布局与验证

保持主面板 max-width 1360px、正文850px与18px/32.4px、原内部阅读滚动。状态区保留A/中央/B结构，状态切换修正后高度201px，迷你组件约72px；无独立卡片底色、边框或动画。完整版少量点适配抽屉宽度，长序列只在图内滚动。

- 前端245项测试 / 11文件通过，`npm run build`（TS + Vite）通过。
- 新增测试覆盖双向切换、同模式、单点、空记录、失败/未确认、缺字段、未知KV版本、0与非有限成本、稀疏序号、40点全量与8点窗口、实际事件追加与历史重放、跨Session提示清理，以及两处实际插入位置。
- 实际浏览器1440×900：document无横向溢出，正文850px，workspace max-width1360px，reader overflow-y:auto；12点完整版无裁切；迷你仅#5–#12；提示显示和Escape不关抽屉通过。另在1280×720查看迷你比例正常。控制台无warn/error。
- 原8000/5173的真实0.5B历史 Session 可正常打开，六次KV节点与SNR轨迹共存。该服务及原历史保持不变。
- 隔离8020/5192正式React + DiscussionService + random/tiny-qwen2 CPU实测：三轮6次真实协议交接，TOKEN→KV→TOKEN→TOKEN→KV→KV，全部append成功，WebSocket从空状态更新至6点，刷新仍6点；另六轮12次TOKEN验证8点窗口与全部12点。tiny使用测试Tokenizer和受控输出ID，真实执行Prefill/Decode/序列化/接收；只作工程验收，不作预训练模型质量/性能证据。无伪造模式或通信记录。
- 20轮tiny请求曾被模型2048上下文预算正常拒绝；未绕过保护。40点绘制由单元测试覆盖，真实浏览器长序列使用12点。

截图：迷你图（原研发资料，未随公开副本提供：`mini-1440.jpg`）、完整版（原研发资料，未随公开副本提供：`full-1440.jpg`）。机器可读证据：validation.json（原研发资料，未随公开副本提供：`validation.json`）。原始Session、测试日志与隔离运行脚本保留于忽略目录 `benchmark-results/mode-timeline/`；临时服务验收后停止。

当前代码已支持V0–V3/AUTO，早期 `dashboard-ui-implementation` 的「仅V0/V1、AUTO禁用」说明属于历史阶段；本任务按当前代码与已验收1360/850布局实现。

边界：未确认交付的旧记录无法还原成成功节点；失败详情仍从原通信证据查看。40点超过抽屉可见宽度时需图内横向滚动。没有扩展历史持久化、改变模型行为或重复进行大型模型实验。

## 9.2 运行中状态切换跳动修正

根因：原 `.channel-route` 是允许换行的 flex；最近交接短文字可与箭头同行，AUTO 待交付长文字移到下一行，动态改变状态栏自然高度。空时间线提示段落的 margin 折叠另造成首个节点出现时2px偏移。

仅有时间线的讨论状态区改为固定的箭头18px、说明40px（两行预留）与4px间距；按钮/只读说明使用相同高度、行高、对齐和内部文字结构。超长说明最多两行，原文保留在title中。状态标题固定22px，mini保持72px并使用flow-root阻止空状态margin外溢。Canonical原无时间线舞台沿用原布局；不改交接、AUTO、Shannon或实际模式字段。

实际浏览器使用正式CommunicationStage、ReadingSurface、ModeTimeline及discussionStage的隔离布局夹具，历史handoff取本轮已保存真实6次协议记录，运行阶段由夹具切换；这仅是布局回归，不宣称新运行模型实验。1440×900、1920×1080、1280×720分别检查生成0/交接0/生成1/交接1/生成6/交接6/完成6，共21次；每尺寸状态栏全部201px，Agent、时间线、toolbar、reader坐标和高度均不随阶段变化，无水平溢出。1440 reader.scrollTop=200在生成/交接/完成往返后保持200。浏览器无warning/error。245前端测试及TS/Vite通过。

生成状态（原研发资料，未随公开副本提供：`stable-generating-1440.jpg`） / 待交付状态（原研发资料，未随公开副本提供：`stable-pending-1440.jpg`），截图仅裁出状态区与下方工具栏方便对比；逐状态DOM证据（原研发资料，未随公开副本提供：`stable-status-layout.json`）。夹具源码保留在忽略目录benchmark-results/mode-timeline/.mode-layout-check.{html,tsx}，临时复制回frontend根目录即可用Vite复现；生产页面不增加测试入口。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

# 8.3C — 双 Agent 通信质量、可靠性与最终系统对照

2026-10-15（Asia/Shanghai），起始 HEAD `71873f406403f55fe04c42ec23f7e68bf689f38c`。**研究验收完成；结果不支持默认启用 KV，也不支持将当前双 Agent 系统称为通用语义可靠系统。** 五条通路均真实执行；适用条件下交付、Cache 事务及继续 Decode 通过，语义失败、V3 不适用与一次资源保护中止全部保留。

TOKEN 是当前最合适的默认工程参照，但会误读秘密、编造对方事实并错误审核约束题。V1/V2 在本轮部分提取任务上接近 TOKEN，不能据此认证一般等价。V0、V1、V2、V3 都有 TOKEN 未出现的语义问题；V2/V3 没有稳定质量收益。8.3B 的搬运优化保持兼容，性能收益没有消除语义问题或大包成本。

## 1. 目标、范围与证据

本轮回答交付、理解、信息利用、多轮任务、AUTO 决策、8.3B 稳定性和可对外支持的结论。没有开发新通信算法，没有修改 V0–V3、原评分、默认 Prompt、旧 benchmark、packed 算法或 Dashboard 视觉。

- 实验前审计、预登记与唯一预试验修订（原研发资料，未随公开副本提供：`system-quality-83c-plan.md`）。
- 六阶段冻结配置、原始 SHA、统计与全部输出 IDs（原研发资料，未随公开副本提供：`system-quality-83c/evidence/data/`）。
- 人工逐配对评价（原研发资料，未随公开副本提供：`system-quality-83c/evidence/manual-pairs.json`）、双向评价（原研发资料，未随公开副本提供：`system-quality-83c/evidence/manual-secrets.json`）、完整讨论评价（原研发资料，未随公开副本提供：`system-quality-83c/evidence/manual-discussions.json`）。每项含原文依据及对应 raw SHA，标记 `manual`，不是额外模型 judge。
- 逐实际 receipt 与耗时（原研发资料，未随公开副本提供：`system-quality-83c/evidence/data/handoffs.json`）、完整 runtime 结果（原研发资料，未随公开副本提供：`system-quality-83c/evidence/data/runtimes.json`）、综合审计（原研发资料，未随公开副本提供：`system-quality-83c/evidence/audit.json`）、资源（原研发资料，未随公开副本提供：`system-quality-83c/evidence/resources.json`）。

大型协议包、逐 forward/health 日志、完整 Cache 摘要及原始失败在独立忽略目录 `benchmark-results/system-quality-83c/`。Git 证据约4.7MiB，不含 Tensor 或权重。旧8.3A/8.3B原始索引核验通过，26个用户未跟踪设计/副本文件SHA保持。

## 2. 当前系统实现与工程边界

A/B 共享一份权重，独立 System、ledger、pending、长期 DynamicCache。TOKEN 发送实际生成 IDs，经 TKVT bytes，在 Receiver 自己历史后增量 Prefill。KV 从 Sender 实际 Decode 提取消息内容的 K/V，TKVI 包含内容 TKVT、TKVK Tensor body、ModelContract 及 SHA256；Receiver restore 后加入自己的历史。

V0 原始拼接；V1 真正修正 post-RoPE K；V2=V1+连续末1内容Token重算；V3=V1+原首ASCII数字槽位的真实单Token forward/write-back。位置修正不消除 `KV(M|H_A)` 与 `KV(M|H_B)` 的上下文语义差异。协议、cache length 和 ledger 一致，不等于语义等价。

正式 Selector 新增精确 workload 域检查；Discussion/SecretSum 强制启用。除此之外，仅有独立实验/证据工具及测试。InMemoryByteTransport 是单进程 bytes 边界，网络成本是模拟公式，不是物理网络。

## 3. 模型、环境与资源

离线 `Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775`，Apple M5/16GiB、MPS/BF16、batch1、seed77、raw argmax greedy。PyTorch2.14.0、Transformers5.17.0；每阶段加载前保存 checkpoint 全文件 SHA、Tokenizer/Chat Template、Prompt、答案、采样、预算和源码SHA。24层、2 KV heads、head_dim64，raw每内容Token12,288B。

没有下载权重、CPU推理替代、3B、服务器/RTX5090访问或调整Swap/MPS阈值。六个顺序模型进程，不并行加载。资源沿用既有2秒采样及 sustained pressure/rapid swap/critical/stalled forward保护。

完整讨论最后的开放题 AUTO 在第四次生成触发 `ResourceStop: sustained_pressure_and_rapid_swap`；前三段和三次交接保留，未产生最终总结，未重跑替换。该阶段约179秒，Swap采样增长351.63MiB；不是达到某固定Swap绝对量就停止。正式质量阶段压力1/2、Swap无新增；Profile及Dashboard随后使用原策略正常完成。最大阶段MPS driver采样约2.06GiB，不能视为瞬时峰或全系统物理内存账。

## 4. 冻结方案与执行数量

|阶段|预定范围|完成生成 / 尝试生成|交接成功 / 尝试|实际forward|
|---|---|---:|---:|---:|
|pilot-v1|数字52、yellow事实；不入正式统计|11/11|9/10|186|
|pilot-v2|同两题，统一简化Sender指令；不入正式统计|11/11|9/10|86|
|formal-v2|15同消息×5方案；2无消息控制；3秘密×5方案|134/134|102/105|2060|
|discussion-v2|四题×六方案，两轮256预算|114/115|91/92|9569|
|profile-v2|两个历史真实快照×两带宽，16预算继续Decode|4/4|4/4|86|
|dashboard-v2|正式React两轮256预算，活动PATCH|5/5|4/4|184|
|合计|预试验不混入正式质量分母|279/280|219/225|12171|

280包含一次资源中止的生成尝试，不冒称280条完整输出。六次拒绝均为无数字V3：预试验2、正式事实3、完整开放讨论1。其余219次适用交接均成功。初始化、framing、pending及重算forward包含在12171内，不当作额外质量样本。

pilot-v1 的 Sender 在数字题写无关Task清单并截断，在事实题未发送yellow。正式采样前只将统一 Sender stage 简化为逐字复述上条私有信息，任务集版本从v1到v2；pilot-v2验证可运行后停止改Prompt。原试验完整保留。正式任务值、Receiver问题、评分、预算及修正规则没有结果后调整。

## 5. 任务与秘密隔离

主实验五类各3题：数字27/64/138；固定非词典暗号K7M2/Q4R9/T8N5X3；cup blue/door green/bag purple；locker7/12/23颜色及钥匙位置；8/16/24 coins再加3。它们是预先固定值，不宣称随机总体抽样。

私有事实只进入A本地user输入；B实际初始ledger不含目标原子，公共说明和Receiver问题不泄露目标。期望答案仅在Decode后评分使用。每个A只真实生成一次；全部15条Sender均实际包含目标事实，部分另复述公共说明，如实保留。

原SecretSum17/26、17/31及新增13/29，每个System仅持有自身数字；公共任务无双方秘密/总和。库存题A仅知12 red boxes，B仅知7 blue boxes；公共问题无数量/19。两个无消息控制分别针对27与blue，B均回答UNKNOWN；它们只支持这两条控制，不证明模型普遍不会猜。

## 6. 配对公平性与工程证据

每条源消息保留真实IDs、真实TKVI包及SHA、A/B缓存Tensor摘要、ledger、span和pending。五分支重置同一原状态字典，正式Session在私有work中复制并提交；每分支后原基础Tensor SHA不变。相同Receiver generation输入逐ID一致，所有KV分支packet SHA相同。无clean Prefill消息KV、人工回复或Tensor捷径。

真实Sender内容5–25Token；A交接前cached85–114、B57；各组长度详见sources。V2固定k1，在15个配对样本中全部0<1<N，确实保留前段KV；无k0或k=N冒充部分重算。V3数字任务实际选槽位并修复24层；三条无数字事实明确拒绝，未fallback。

每次成功同时检查真实packet长度/magic、received IDs、发送span、接收内容span、双端消息ledger、pending关闭、Cache增长及storage独立。Trace核对实际forward IDs、完整mask、真实rotary位置、返回Cache长度；V2/V3的实际位置/forward与receipt修正索引一致。接收后继续真实Decode，不凭正确答案反推通信成功。

## 7. 工程交付成功率

下面排除两次预试验，包含正式质量、完整讨论、性能replay和Dashboard；TOKEN含质量AUTO实际TOKEN。

|实际方案|成功/尝试|结果|
|---|---:|---|
|TOKEN|59/59|全部真实交付|
|V0|38/38|全部真实交付|
|V1|37/37|全部真实交付|
|V2 k1|37/37|全部真实交付及真实重算|
|V3|30/34|4次无数字明确拒绝；合法槽位30/30|

这些是本轮有界工程观察，不能外推为生产可靠率保证。资源中止发生在生成阶段，不是一次已发送后半提交的通信失败。预试验另有18/20成功、2次V3拒绝。

## 8. TOKEN质量基线及自动评分限制

TOKEN全部15条主交接成功，但人工确认信息提取仅11/15；Q4R9、blue、green回答UNKNOWN，8 coins任务回答7 coins。全部失败保留，不能把它们归因KV。

冻结自动评分是窄原子/等式规则，结果原样保留在paired/summary。它对句末句号的整数等式返回unknown，并且仅原子命中不能证明属性归属。例如V0 `Locker 12, red key, drawer` 含三个原子，却不能确认locker颜色。**没有修parser或回写自动分数**；人工按预登记语义/归属/算式规则独立评价，原文和理由可复核。

下文提取表为人工明确确认；数学“最终结果数值正确”、完整按要求写算式、利用信息三个维度分开。`unknown`保留分母，不算yes，也不强行当no。

## 9. V0–V3同handoff消息理解对照

表格为yes/预登记3条；括号U=unknown数量，其他为明确no。每类仅3条，不合成通用“KV准确率”。

|任务类型|TOKEN|V0|V1|V2 k1|V3|
|---|---:|---:|---:|---:|---:|
|数字提取|3/3|3/3|3/3|3/3|3/3|
|暗号复述|2/3|2/3|1/3|1/3|2/3|
|单事实提取|1/3|1/3|1/3|1/3|0/3（3U：不适用）|
|多事实完整提取|3/3|0/3（1U）|2/3|2/3|2/3|
|计算前原数提取|2/3|2/3（1U）|3/3|3/3|3/3|

完整冻结集明确提取覆盖分别11/15、8/15（2U）、10/15、10/15、10/15（3U）；仅作样本覆盖核对，不解释为跨任务总体准确率。原自动原子提取V0为9/15，人工发现一条属性归属不确定降为unknown；两套数据均保留。

## 10. 信息利用与最终答案

计算题人工数值结果正确：TOKEN2/3，V0/V1/V2/V3各3/3。V0 `19` 只给正确结果，未明确原数16，因此其信息利用为unknown，不能凭19认证完整理解；TOKEN8题错误，KV反而能给11，必须保留这一KV独有成功。

计算题按要求给出正确明确等式：TOKEN2/3、V0 2/3、V1 0/3、V2 1/3、V3 1/3。其余部分是正确自然语言计算但未写等式。冻结自动parser对部分带句号等式unknown；不得将自动0等式命中写成数学都算错。

全部15题人工最终答案yes：TOKEN11、V0 9（1U）、V1 10、V2 10、V3 10（3U）；完整任务要求yes分别11、8（1U）、7、8、8（3U）。这些汇总用于复算，分类结果和原文才是主要结论。

## 11. TOKEN成功条件分析及配对差异

预定义TOKEN明确提取成功子集n=11，没有删除另外4条。KV提取结果：V0 7yes/2no/2U；V1 9yes/2no；V2同V1；V3 9yes/1no/1U。

- V0新增明确失败：locker7漏读、locker23漏key location；locker12归属及16计算原数为unknown。
- V1/V2新增明确失败：K7M2变UNKNOWN；locker23把key location错误写locker。
- V3新增失败locker23；TOKEN成功的purple事实则因无数字不适用。
- TOKEN失败的8 coins任务，各KV能提取8并给正确11；不能只挑TOKEN成功集展示。

完整配对差异（原研发资料，未随公开副本提供：`system-quality-83c/evidence/manual-paired-differences.json`）同时保存最终答案/完整任务的discordant列表。样本小、同模型确定性单次、类别不同，无法证明细微排名或一般质量非劣性；不做无依据的显著性/置信区间包装。

## 12. 双向秘密交换

每组五方案均A→B→A实际bytes，30/30交接成功，45/45生成EOS。工程成功与下表独立。

|方案|B明确读对A秘密|最终数值和正确|严格双方归属、保留自身、利用信息及最终全通过|
|---|---:|---:|---:|
|TOKEN|3/3|3/3|3/3|
|V0|0/3（2U）|2/3|0/3|
|V1|0/3（3U）|2/3|0/3|
|V2 k1|0/3（2U）|1/3|0/3|
|V3|2/3|0/3（1U）|0/3|

V1/V2的无归属算式不能当明确所有权通过。V2在17/31中把A31/自己17颠倒，和48在交换律下仍正确。V3两组读到了A17，却把自己的26/31标unknown；这时B发言本身已经缺失秘密，A无法从该消息学到26/31，不能把最终失败都归咎反向KV交付。

偏差定位：V0 17/31的B首次生成31+31=62；V1/V2 17/26的B算式含正确两个数，但反向接收后A输出17+17=34；13/29的V2/V3最终把B写13并得26。均发生在有真实成功receipt及有效Cache后的模型信息利用阶段，不能由当前证据分离成唯一某层因果机制。

## 13. 完整两轮讨论

每题同问题、角色、预算256、两轮、A总结、trace20/10000/50/1000Mbps。所有策略首A相同，第一次交接后内容自然分叉；整场结果不作为逐handoff因果比较。

|任务|TOKEN/V0/V1/V2/V3/质量AUTO|语义结果|
|---|---|---|
|23+48及检查|6/6工程完成|6/6最终71；多为重复结论，独立检查很弱|
|私有库存整合12 red + 7 blue|6/6工程完成|0/6正确；A首次就编造B15 red，B不纠正，最终27|
|双约束选方案|6/6工程完成|0/6正确；都选容量8的X，唯一可行Y容量12、成本8|
|开放通信系统设计|4完成、V3拒绝、AUTO资源中止|仅人工案例；不输出“理解准确率”|

22/24工程Session completed，不能写成22/24任务正确。库存第一条A尚未通信已错误，说明基础模型限制显著；TOKEN也错误，不能全归因KV。V1/V2/V3又产生12 blue等属性漂移。约束题B有时指出X容量不足，但同时说8=10或8>12，A没有修正最终选择。

## 14. 多轮推进、纠错和开放题人工评价

加法基本反复确认，V2重复更长并声称“updated answer does not resolve the issues”，没有新的实质验证。库存题没有纠正对方虚构，也没有使用B原System的7 blue。方案题部分B出现可用纠错片段，但伴矛盾且未被A吸收。

开放题TOKEN最终主要评价“前文正确”，缺少具体设计；V0/V1反复要求所有内容都成为primary focus；V2指责原文没有high-speed/complexity/flexibility，而原文恰已明确提到，最终总结保留自相矛盾。V3首消息无数字即终止。AUTO前三段与TOKEN相同，第四生成资源中止，没有最终答案。全部原文与按冻结rubric的评价在manual-discussions，不伪造开放题客观分数。

## 15. EOS、截断与资源中止

正式配对、秘密及无消息控制共134条完整生成，全部EOS。完整讨论完成114条：104EOS、10条max_new_tokens；另1生成resource_abort没有完整输出。截断主要发生在KV方案的长Reviewer消息，10条实际IDs仍完整交付，不能视作语义完整。

两预试验21EOS/1预算截断；Profile四条16Token继续Decode都是预算截断，仅作计算兼容性；Dashboard5条均EOS。全任务279完整输出=264EOS+15预算截断。新实验没有context_limit样本；其工程路径及历史真实1024/长截断证据复用8.2C，不能说本轮新增验证了所有终止类型的真实模型场景。

## 16. AUTO quality_first

三组客观讨论12次、开放题中止前3次、Dashboard4次，共19次真实质量AUTO决策和19次TOKEN交付。高带宽不改变质量许可，V0–V2仍quality_not_certified；V3若不在白名单为not_whitelisted，显式包含时的known_quality_regression由既有及新增工程测试核查。

默认无profile成本为null/uncalibrated，不填0；预测wire逐条等于receipt。没有隐藏回退。成本域外/域未验证的质量TOKEN仍安全执行，但不套用本地/总成本；错误模型身份仍拒绝。

## 17. 旧、新Profile域

8.2A pooled Profile与8.3A/8.3B研究点Profile保持原数据。旧schema只守消息长度和A+B总和，缺单端历史、源位置、EOS/pending和framing，不能作为普遍生产模型。

新增workload_domain要求：Sender/Receiver各自cached数、源span起点、termination、两端pending、两端ledger SHA、实际消息IDs SHA及incoming framing SHA；加原两端model/revision/config/tokenizer/dtype/device身份及N/总Cache/k范围。正式Runtime强制 `require_workload_domain=True`，缺域测量Profile明确 `profile_workload_domain_unverified`，精确条件不符 `profile_out_of_domain`。性能模式不能借此转TOKEN；没有候选就失败。质量TOKEN独立保留、成本unknown。

历史metadata-only离线分析API保留原默认，用于复算旧数学结果，不代表通过正式Runtime部署域保护。外部直接使用Selector也须显式要求严格域。精确域是保守replay守卫，不是自动质量认证，也不是对任意同ledger缓存数值/历史算法身份的证明；不在Selector中导出Tensor或遍历权重。当前受Session所有权、不可变模型约束，适合本轮冻结原快照，尚无跨内容/跨长度性能泛化保证。

## 18. performance_experiment真实行为

先验证8.3B四个原Profile在严格Runtime条件下缺域或超域被拒绝，共8次离线拒绝；未改任何成本数字。再从8.3A原实际Tensor文件SHA核验恢复EOS2与pending512两个快照，只新增对应精确域，成本及sample_count仍逐字段等于8.3B。

|快照|20Mbps真实选择执行|10000Mbps真实选择执行|实际wire|
|---|---|---|---|
|EOS2，A124/B97|TOKEN|TOKEN|22B|
|pending512，A630/B128|TOKEN|KV V0|2058B / 6,296,969B|

四次都由真实Selector选一路、正式Session执行；没有运行未选候选来回填选择。KV后Receiver真实继续Decode。未准备contract时，两点两带宽KV均因missing_startup_cost_profile排除。相同Cache总和但两端长度改变的两次测试明确域拒绝，无额外交接。

这是**精确历史快照的在线选择/交付replay**，不是新的完整讨论性能AUTO切换，也不是新的独立长消息holdout。没有将3.37Gbps写入逻辑，没有synthetic成本。8.3B单点波动和其数学交叉点限制全部保留；本轮仪器wall不重新拟合性能Profile。后续必须独立长消息holdout和目标设备真实网络测量。

## 19. 动态trace、实时PATCH与REST/WS

完整讨论按四项trace冻结，不sleep。Dashboard实际Session `79bc8d2b-afad-4a20-bf37-f8fb633b39b2`，首handoff在20Mbps/revision0完成后，真实HTTP PATCH10000Mbps/2ms/manual_override被接受；版本1从下一未冻结#2生效。后续三次decision和receipt都是10000Mbps/revision1，首条仍原版本。

REST最终结果、原事件JSONL、WS代理记录及浏览器展示一致：4条发言、4receipt、最终总结、revision24；只有一次discussion POST，刷新未重复启动。PATCH由有界控制脚本调用真实HTTP（不是伪造UI点击），浏览器实见更新历史。原配置trace保留，不冒称活动网络仍按旧trace推进。真实网络界面（原研发资料，未随公开副本提供：`system-quality-83c/evidence/screenshots/network-1440.jpg`）。

## 20. 8.3B兼容与可靠性

本轮所有成功KV交接实际export/restore均使用既有packed域，原TKVI/TKVK/contract/checksum与事务保持；正式配对、秘密、多轮和512replay均继续Decode。failed V3在选择数字槽位前已发生真实bytes发送，但无成功receipt，双端状态/原Cache SHA保留，未隐式TOKEN/V1回退。

域外legacy固定分派、checksum/坏包/ModelContract拒绝、suffix/slot非法配置、append/closing/Sender close失败回滚、EOS/pending/截断和storage隔离复用原工程测试与8.3B真实证据。新全量1063覆盖所有旧1029及34项新测试，没有为坏包再次加载预训练模型。没有声称本轮新测了>6MiB默认legacy的真实模型质量矩阵。

## 21. 成功案例与失败最小复现

成功：同消息27/64/138五方案都提取正确；K7M2 TOKEN/V0/V3正确；TOKEN三组原秘密双向正确；所有方案简单加法两轮得71。部分KV在TOKEN误读8 coins时正确给11，结果照常展示。

失败：同一K7M2源packet/Receiver状态下，V1/V2回答UNKNOWN，TOKEN为K7M2；locker23的TOKEN正确key in bag，KV遗漏或错写locker；17/26的V1/V2最终34；17/31的V2双方归属颠倒却得48；V3丢自己秘密；库存所有方案编造B15；方案选择所有方案忽略8<10。每个例子可从paired/runtimes、generations、handoffs及raw SHA复核，不只保存最佳一次。

## 22. Dashboard最终验收

沿用原正式React生产构建，没有改前端/CSS/API投影，也没有重做视口矩阵。1440×900实测document1440×900，四段正文850px、18px/32.4px，technicalClutter=false。检查问题输入、手动TOKEN/KV选项、AUTO、trace、真实最终答案、receipt/成本排除/bytes/耗时、生成EOS/预算详情及刷新恢复同Session。手动模式真实执行证据复用8.2B/C，本轮新执行为AUTO完整讨论。

正式展示截图（原研发资料，未随公开副本提供：`system-quality-83c/evidence/screenshots/dashboard-1440.jpg`）。活动PATCH和元数据在离线browser证据中；非EOS截断界面复用8.2C真实截图，不用fixture冒充本轮模型运行。新临时8013/5183服务验收后关闭，无更改用户服务或布局。

## 23. 性能与质量、统计不确定性

8.3B改善导出/restore固定成本，在N512单点存在本地和中位数数学收益；本轮确认优化后实际多轮路径能工作。但包仍约每Token12KiB，V1/V2/V3额外计算仍在，且语义不可靠；一次成本最优选择不等于质量认证。因此不足以默认启用KV或宣称真实网络加速。

每类n3、秘密n3、普通讨论每任务每方案n1，只支持这些冻结条件；没有独立随机任务分布、统计功效或跨LLM外推。Greedy/MPS/BF16近并列数值和分段forward可能改变措辞；本轮未做逐层Full数值归因。完整讨论消息分叉，不能把最终差异都归因某一个handoff。

人工评价由本轮审阅者完成，无盲评或多评者一致性；atom parser过宽/等式parser过窄均已暴露，不把它们当通用语义judge。unknown始终保留；小类别不能汇总成一个可靠算法排名。资源abort是一次受保护系统结果，不删除、不重跑改善完成率。

## 24. 工程回归、源码审计与复现

最终全量Python一次：**1063 passed / 1既有可选AutoGen skipped / 2既有依赖warnings，17.46秒**；34项新增覆盖评分、答案隔离、同快照/同包/同generation IDs、域各坐标/同总和不同端、startup/身份/Runtime强制域、unknown分母和SHA拒绝。针对性107通过。compileall、run/export/证据服务/Discussion CLI help、diff检查通过。前端源/API投影未改，按范围不重复Frontend/TypeScript/build或原完整浏览器矩阵。

所有六阶段raw索引复算，manual条目与原raw SHA对应；统计与成功/失败计数复算。正式采样后Harness仅增加独立profile_audit replay函数及import，未改变配对/生成/交付/评分函数；保存的formal实测源码SHA与冻结值完全匹配，AST核查其他执行函数不变。测试和exporter后续新增记录在source-drift，不重写旧spec。pilot-v1到v2的Prompt差异明确记录。生产算法、Token基线、历史评分/原数据保持。

```bash
# 每阶段必须新目录；原raw、失败、partial均拒绝覆盖。
HF_HUB_OFFLINE=1 .venv/bin/python run_system_quality_validation.py --phase pilot --output-dir benchmark-results/system-quality-new/pilot
HF_HUB_OFFLINE=1 .venv/bin/python run_system_quality_validation.py --phase formal --output-dir benchmark-results/system-quality-new/formal
HF_HUB_OFFLINE=1 .venv/bin/python run_system_quality_validation.py --phase discussion --output-dir benchmark-results/system-quality-new/discussion
# 精确历史replay依赖本机原8.3A Tensor，缺失即拒绝，不用IDs重造KV。
HF_HUB_OFFLINE=1 .venv/bin/python run_system_quality_validation.py --phase profile --output-dir benchmark-results/system-quality-new/profile
# 以下只读审计本轮已保留raw、人工证据和统计，不加载模型：
.venv/bin/python verify_system_quality_evidence.py
# 导出本轮标准六阶段目录；新结果须使用相同阶段命名，并先完成真实浏览器证据封存。
.venv/bin/python export_system_quality_validation.py --results-dir benchmark-results/system-quality-83c --output-dir /tmp/system-quality-evidence-new
```

Exporter要求本轮六阶段布局，验证协议/forward、SHA及统计；不会代写人工评价或补齐未执行实验。audit脚本保留本轮预登记数量断言，不是任意数据集通用评分器。原始目录仍须本机保留，Git不含可恢复全部Tensor；仅Git克隆不能假称复算大raw已验证。

## 25. 最终可支持与不能支持的结论

可演示：真实独立上下文双Agent；TOKEN及适用V0–V3增量bytes/KV/Cache/继续推理；真实质量过滤AUTO；trace/活动PATCH；receipt/IDs/bytes/耗时/截断/错误可观测；8.3B批量通路；正式桌面阅读及Session恢复。

仍属研究：跨历史KV语义、V2/V3修复收益、精确点域性能AUTO、长上下文/长期稳定性和一般多轮协作质量。未实现/未验证：物理/跨机网络、CUDA/5090、3B本轮质量、异构模型、跨内容性能泛化、在线成本学习、通用语义judge、持久化历史、生产持续watchdog、自动续写/取消/流式输出。

TOKEN能可靠交付本轮消息，但不能可靠完成所有当前任务；V0–V3合法条件都能工程交接，V1/V2部分提取接近TOKEN，所有KV仍有新增错误。V3有局部成功，但秘密丢失和无数字失败限制突出。V2/V3未提供稳定实用质量收益。默认quality_first仍应TOKEN，8.3B性能优势不足以改变该决定。双Agent可完成简单冻结任务，也可能只重复确认、共同认可错误并放大幻觉。**不能把工程成功写成语义可靠，不能把数学交叉点写成真实网络加速。**

本轮研究及工程验收完成后仅提交任务范围，不push；Git提交身份以最终Git记录为准。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

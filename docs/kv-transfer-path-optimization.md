# 8.3B — KV 导出与恢复通路优化及真实前后对照

2026-10-15（Asia/Shanghai）。起始 HEAD `3cee381ad3bf8f47db621e07b5099f6cbf64edae`。**结论 A：保留并有界启用批量搬运。** 四个实测条件的完整 V0 handoff 中位数降低17.87–19.72ms，全部20对 steady 样本均快于同轮 legacy；代表 V1 的5对也全部更快。协议包、各策略提交后的 Cache 位模式、ledger 与后续 Decode 新旧一致。

默认仅在 **MPS、同质 BF16、raw KV ≤6MiB** 使用 packed；其他条件走保留的 legacy。这个固定适配器分派不是通信 Selector，也没有修改 TOKEN、AUTO、网络、Prompt、RoPE 或 V2/V3 算法。没有据此认证跨历史 KV 的语义质量。

## 1. 证据与范围

- 冻结配置、输入/源码 SHA（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/frozen-spec.json`）、实际模型身份和环境（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/settings.json`）。
- 全部196条核心标量样本 CSV（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/samples.csv`）、全部阶段标量（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/samples.json`）、分组统计及布局（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/groups.json`）。
- 配对差值（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/paired-deltas.json`）、网络与每对交叉点（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/network.json`）、独立新研究 Profile（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/profiles.json`）。
- V0–V3真实正确性/继续Decode（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/correctness.json`）、内存与调用计数：短（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/eos-2-buffer-probe.json`）/长（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/pending-512-buffer-probe.json`）、系统资源（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/resources.json`）。
- 完成数量（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/completion.json`）、只读审计与最终源码差异（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/audit.json`）、raw SHA索引（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/raw-index.json`）、Git证据SHA（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/evidence-sha256.json`）、工程检查（原研发资料，未随公开副本提供：`kv-transfer-83b/evidence/engineering.json`）。

新 raw 独立保存于忽略目录 `benchmark-results/kv-transfer-83b/validated/`。其父目录保留首次身份检查失败及冻结源码，不覆盖失败或8.3A结果。Git仅保存约1MiB标量/冻结配置/源码证据；实际大 Tensor 仍在旧忽略目录。用户未跟踪设计资产、文档副本、历史benchmark未改，前后 SHA 复核通过。

## 2. 原通路：组合成本与同步

审计了 `.agent`、架构/工作流、8.3A和V0–V3/AUTO报告、适配器、协议、正式Session和性能Harness。8.3A的约9ms导出/上传是组合阶段，不是纯DMA或物理链路测量。

本机 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`，24层 K 和24层 V，均 `[1,2,S,64]`、BF16、MPS:0，每内容Token共12,288B。DynamicCache按层独立保存 K/V。新实验记录所有96个基础状态Tensor的shape/stride/device；恢复后的基础Cache及私有span clone为contiguous，典型stride `[128*S,64*S,64,1]`。非连续合法输入另有真实MPS和工程测试。

旧出口：验证完整Cache/位置 → 必要pending私有补算 → 48个span clone → 验证 → 前同步 → 每Tensor detach/必要contiguous/cpu/uint8 view/numpy/tobytes → 后同步 → immutable neutral layers → 原TKVK/TKVI编码/checksum。

旧入口：原checksum/ModelContract/描述符校验 → 每Tensor bytearray/frombuffer/原dtype view/reshape/CPU clone → 前同步 → 48次to(device) → 后同步 → 公开DynamicCache构造和同步 → 最终结构验证。

除显式synchronize，阻塞`.cpu()`/`.to(mps)`也是可能等待设备的边界。shape/dtype/stride读取不导出数据。短消息CPU重建旧成本约0.1ms，仅删Host clone不足解释9ms。Tensor→bytes和bytes→Tensor是真实工作；没有绕过它们。

## 3. 实现前技术判断与唯一方案

在实现前形成的判断是：**同dtype连续缓冲区批量搬运最有希望**。48→1次跨设备API调用，出口48次numpy/tobytes→1次，入口48个CPU clone→一个私有Host缓冲区；同时新增一次MPS stack分配/复制、Host拼接/切片。N512 raw6MiB，出口设备临时缓冲区约6MiB，Host可能多约两份raw，须验证内存。

候选比较：单独删CPU clone收益上限小；异步流、Sender共享storage、缓冲池带来额外生命周期/同步复杂度；重写Cache或优化pending/framing超出本轮边界。没有预设cat/stack一定快；只实施这一个方案，先小条件再有界矩阵，未开展多方案搜索。

最初设想构造器可保留views，随后检查安装源码确认：**Transformers5.17 `DynamicCache(ddp_cache_data=...)` 调用DynamicLayer.update，将输入和空Tensor cat，实际又复制48个Tensor。** 保留公开API并计费，不为省这次复制改内部字段。最终Cache拥有48份独立storage，输入views只在构造阶段共享一个私有buffer。

## 4. 优化执行流程与局部源码

仅生产 [kv_adapter.py](../src/token_kv_agent_system/model/kv_adapter.py) 改动；新增 [独立真实Harness](../src/token_kv_agent_system/kv_transfer_validation.py)、[运行入口](../run_kv_transfer_validation.py)、[只读导出审计](../export_kv_transfer_validation.py)。旧保护/算法/协议文件均未改。

```text
出口：原验证/真实span clone
  → stack([L0.K,L0.V,L1.K,L1.V,...])（原dtype）
  → 一次cpu → uint8 view → 一次numpy/tobytes
  → 按原BHSD大小切immutable bytes → 原neutral层序 → 原协议编码

入口：原协议/合约/兼容性验证
  → 按原K/V层序join bytes → 私有bytearray/frombuffer/原dtype view
  → 一次to(MPS) → reshape/unbind为48个不重叠BHSD view
  → 原公开DynamicCache构造（再次copy）→ 原结构校验
  → 原Receiver私有work/prefix/RoPE/append/closing及双端事务提交
```

没有BF16→FP16、FP32中间转换、K额外旋转或V数值变换。stack自动按逻辑顺序处理非连续输入，但仅uniform dtype进入packed，禁止隐式promotion。canonical little-endian转换保留。显式 `transfer_path='legacy'/'packed'` 可复现对照；默认auto只做上述固定设备/dtype/字节域分派，失败直接抛错，不catch后重试legacy或TOKEN。

正式auto上限6MiB来自本轮最大实测raw大小；显式packed另有64MiB缓冲区硬上限，超限在stack/Host join前拒绝。原增量协议64MiB整包限制、消息/位置/上下文及Resource Guard都保留；raw6MiB不等于wire6MiB。域外legacy是事先固定适用域分派，非异常回退。CPU/CUDA及更大payload不改变默认路径。

## 5. 所有权、生命周期与事务

出口stack只读detached Sender tensor并写新storage；neutral层保存immutable bytes。入口每次新建bytearray/上传buffer，没有pool、跨handoff复用或Sender引用。view每个区间恰为一个BHSD Tensor，offset依次为`i*numel`，大小相同且不重叠；Tensor引用持有storage，局部buffer变量释放不使其失效。

本机公开构造器再次cat，最终48个Tensor各有storage，和输入views、Sender、另一restore均不共享。即使未来公开构造器保留views，输入仍为本次独占、不重叠且有引用计数的storage；当前仅对固定安装版本验收，不宣称任意未来Cache API兼容。

原Receiver工作Cache clone、append/cat/reconstruct、storage-disjoint检查、失败双端不提交均保持。真实MPS测试在删除payload并GC后修改一个restore，不影响另一restore或Sender；新旧8次V0–V3交接后继续真实3Token预算Decode，结果/最终Cache SHA完全相同。工程补充连续六次双向handoff、构造输入view区间、构造结果独立storage、导出stack失败和upload后失败，均无legacy retry或半成品提交。

## 6. 冻结输入与同轮控制

直接读取8.3A原 `.kv` 和state JSON，不把IDs重新Prefill成消息KV、不使用随机缓存、不重新生成Sender消息。所有源文件先验SHA核查，载入后完整Cache Tensor SHA再和旧记录比较；实际权重/Tokenizer文件SHA、ModelContract和backend cost identity逐字段匹配。A/B共享一份权重，各自Cache、System/ledger独立。

原state JSON未保存version/local_input_spans等非内容元数据，也未保存原Tensor stride；本轮用同角色模板重建这些状态字段，精确恢复已保存的Tensor/ledger/draft/pending/framing。同轮两路径严格共享这个相同基础快照，不声称恢复原进程的全部Python对象或原stride。每个分组layout记录本轮真实测量布局，另验证合法stride变化。

|条件|A cached|B cached|实际源内容区间|EOS/pending|TOKEN / KV wire B|
|---|---:|---:|---|---|---:|
|2内容Token|124|97|[122,124)|EOS，出口不补内容|22 / 28,036|
|32内容Token|150|128|[119,151)|max_new_tokens，补1|138 / 396,807|
|512内容Token|630|128|[119,631)|max_new_tokens，补1|2,058 / 6,296,969|
|32内容Token长历史|150|2048|[119,151)|max_new_tokens，补1|138 / 396,807|

长消息及其KV确由8.3A的真实cached Decode产生；32/512由原合法prefix裁剪机制保留相同绝对位置，最后accepted内容Token pending。B2048是8.3A受控本地历史，非自然长对话。本轮不改其IDs/Prompt，不重做质量矩阵。

## 7. 计时、顺序与样本数

复用未改动的8.3A `measured_handoff`：draft和基础快照ready → 完整正式handoff → 双端提交完成 → MPS同步。包含真实Host导出、协议编解码/checksum、内存Transport、restore、工作clone、必要pending、本地framing/closing、修正、检查/事务。基础快照重置、SHA与导出审计在外部；原算法自己的clone/cat仍在内部。

steady不加逐阶段包装；instrumented使用原嵌套同步包装和Adapter子指标，父子不累加。每个核心候选/模式2warmup+5有效，逐轮正序/逆序交错新旧/TOKEN。V1和V0同N32块内测量。仪器模式仍分块，不能把两模式差值全部认定为仪器净费。低量级forward标量与2秒ResourceMonitor保留；Transport只加同量packet引用捕获，SHA在计时外计算，不能称零仪器吞吐。

共 **196次核心handoff = 56warmup+140有效样本**，其中steady/instrumented各70有效；另9次小条件pilot和8次V0–V3正确性handoff，合计 **213次成功handoff**。8次后续Receiver真实生成仅验证新旧一致，不用于速度排名。734次实际受保护forward包含初始化、pending、framing、continue及untimed探针，不是734个性能样本。成功进程加载模型一次、Sender新生成0次；首次身份表示错误还加载过一次模型，**任务总加载2次**，未隐瞒启动失败。

每次核心样本核查实际wire/策略/无内容forward、双端提交Cache SHA与ledger一致；分组前后基础Cache SHA不变。无失败/回退、CPU推理替代、减少payload或跳过device restore。存在慢样本，全部保留，未删outlier或扩成笛卡尔积。

## 8. Sender Host导出与Receiver Restore

以下是instrumented Adapter组合指标，单位ms，**每格n=5**；形式`median [min,max]`。Host导出含新增stack、cpu、bytes和接口/验证，不是纯DMA；restore是原Adapter完整边界，包括CPU构造、上传、Cache构造、结构检查。

|条件|Host legacy|Host packed|Restore legacy|Restore packed|
|---|---:|---:|---:|---:|
|2 EOS|9.737 [9.438,9.998]|0.270 [0.246,0.275]|10.348 [9.940,10.499]|0.767 [0.617,0.813]|
|32 pending|9.500 [9.379,9.610]|0.307 [0.283,0.337]|10.226 [10.204,10.522]|0.814 [0.685,0.850]|
|512 pending|9.566 [8.437,9.617]|0.661 [0.639,1.049]|11.459 [11.318,12.171]|1.786 [1.554,1.891]|
|32 / B2048|9.557 [9.459,9.823]|0.319 [0.307,0.382]|10.246 [10.159,10.539]|0.856 [0.824,1.429]|

Host legacy/packed MAD分别为：EOS0.040/0.005、N320.059/0.021、N5120.051/0.022、B20480.078/0.012ms；Restore MAD分别0.151/0.018、0.022/0.011、0.035/0.105、0.087/0.032ms。完整精度在groups.json。

EOS入口细分：CPU重建 **0.094→0.015ms**、上传组合 **9.658→0.347ms**、公开Cache构造 **0.327→0.335ms**。N512分别 **1.008→0.238**、**9.742→0.865**、**0.620→0.603ms**。构造复制没被省略；主要收益与跨设备接口次数减少一致。没有单独测出Python、MPS命令提交、同步等待和物理复制各自比例，不把组合差全部归因某一种。

untimed真实buffer探针核实：legacy确实调用48次 `_tensor_to_neutral` 和48次 `_neutral_to_cpu_tensor`；packed二者0次、stack1次，输入buffer storage1份。源码执行分支每端跨设备API为48→1，仍实际得到协议bytes并上传MPS。

## 9. 完整V0与TOKEN对照

**steady，每格n=5，median [min,max]ms。** 最后两列分别是两路径中位数相减、同轮配对节省的中位数；是不同统计量。

|条件|TOKEN|V0 legacy|V0 packed|两中位数节省|配对节省median|
|---|---:|---:|---:|---:|---:|
|2 EOS|38.654 [37.766,39.302]|76.576 [75.145,78.144]|56.861 [56.118,75.002]|19.715（25.75%）|19.248|
|32 pending|41.227 [40.886,41.968]|92.107 [91.612,92.230]|74.233 [73.043,86.409]|17.874（19.41%）|17.378|
|512 pending|100.431 [99.631,103.658]|104.417 [103.170,105.753]|85.472 [85.199,99.538]|18.945（18.14%）|18.945|
|32 / B2048|43.683 [43.339,56.350]|102.616 [102.352,102.670]|83.856 [83.564,85.922]|18.760（18.28%）|18.806|

steady legacy/packed MAD：EOS1.211/0.744、N320.092/1.190、N5120.934/0.273、B20480.054/0.293ms。全部20对packed更快，但单对节省范围分别 **1.574–20.608 / 5.698–19.096 / 3.944–19.417 / 16.748–19.052ms**；不能保证每次都省19ms。

instrumented完整median分别 **80.121→60.864 / 96.562→77.695 / 108.105→91.285 / 107.509→89.592ms**。其每组配对节省median为19.099/18.487/18.516/18.715ms；B2048有一对仅省0.071ms，慢样本照实保留。阶段收益和完整收益同方向，但不将阶段median之和当作完整节省。

8.3A历史79.15/117.09ms不是本轮基线。虽然输入Tensor相同，运行时负载/allocator/边界周围观测和Cache载入布局不同，因此仅用本轮76.576/104.417做严格前后比较，不把跨时间降幅归为优化。

## 10. EOS、截断及其他受影响阶段

EOS没有出口pending内容forward，完整Sender出口instrumented composite **10.284→0.790ms**；其span clone **0.342→0.352ms**。N32 pending仍私有补1Token，pending forward **15.226→15.084ms**，完整出口 **26.102→16.789ms**；N512出口 **27.228→18.218ms**。这个约15ms的剩余pending工作没被跳过或修复。

EOS Receiver prefix/closing **17.296/16.558→17.213/16.469ms**、cat/reconstruct **0.830→0.827ms**，基本保留。N512 prefix/closing **17.066/17.650→17.232/17.929ms**、cat **1.345→1.344ms**；codec encode/decode **2.683/2.493→2.507/2.505ms**。不声称这些小差异也是优化效果。

因此固定搬运开销在EOS和pending都减少约19ms，但短EOS相对降幅较大；pending还有额外模型工作。N32/B2048同样获益，cat约 **2.458→2.499ms**、两次本地framing各约21ms，表明优化没有消除历史增长带来的Cache和attention成本。

## 11. V1–V3兼容

同N32/B128代表V1 steady **146.329 [145.679,147.766]→127.856 [126.700,129.390]ms**，MAD0.650→1.005，n=5；两median差18.473ms/12.62%，配对节省median18.803，范围18.177–18.979ms。instrumented155.930→137.167ms，RoPE keys58.993→58.962ms，未优化RoPE。

真实V0、V1、V2 k1、V3各执行legacy/packed两次，同策略提交K/V SHA、新生成IDs和Decode后Cache SHA完全一致，全部仍走TKVI bytes、MPS forward，V0/V1内容forward0，V2/V3实际修正规则保持。不同算法之间的数值不要求相同；“兼容”指优化未改变各策略原有行为，并非V0–V3等价于TOKEN或解决历史质量失败。没有重新做秘密数字质量矩阵。

## 12. 协议兼容性

TKVT/TKVI/TKVK version均不变；同冻结消息所有legacy/packed及V0/V1策略实际packet SHA逐次相同，没有合法变化的非语义字段需要豁免。单独V2/V3沿原相同编码实现和正确性对照；no-fallback用实际KV策略、wire、消息内容forward边界、Cache/ledger核查。

层序严格为每层K后V，BHSD row-major、shape/dtype/缓存长度/绝对源位置保持，post-RoPE K仅由原V1规则修正、V不变。旧checksum/坏包/ModelContract、重复/方向、容量与异常事务检查保留，全量旧断言没有放宽。

## 13. 内存与资源保护

untimed真实N512探针在相同基础分配 **984.160MiB** 下：packed stack存活时 **990.160MiB（+6MiB）**；Host payload返回后回到984.160。旧出口没有这块设备stack。Restore两路径均是上传输入+公开构造结果并存：**984.160→990.160→996.160MiB**，释放后都回到984.160；packed没有增加这条恢复边界的已分配设备峰。N2 stack仅约24KiB。allocator driver保留不代表泄漏。

出口Host同时可能持有CPU packed、合并raw bytes、切开的immutable layers，逻辑规模约3R；旧路约R+当前单Tensor，新增上界量级约2R，即N512约12MiB，allocator/对象开销另计。Restore join→bytearray瞬态比旧Host层列表可能多一份R，随后删除合并raw。**Host瞬时峰未被直接捕获**，不能用RSS不变宣布零成本；真实probe RSS约581.3MiB基本平稳，OS统一内存/allocator复用不能视为物理分配逐字节账。

成功进程20个系统采样均pressure2/warning，Swap4039.75MiB且新增0；MPS current采样峰 **1028.46MiB**、driver **2224.70MiB**、RSS **636.91MiB**。无OOM、critical、资源中止或CPU fallback。2秒系统采样和几个live allocation边界不是分配器完整highwater，可能遗漏短峰。未改Swap/MPS限制、未停止其他用户服务。正式默认6MiB域限制避免把一次小缓冲试验泛化为任意大内存方案。

## 14. 网络交叉点与波动

离线保持原公式：`total=实测steady本地median + actual_wire_bytes*8/(Mbps*1000)+2ms`。不是物理网络测量，不sleep，不修改Selector。EOS/N32/B2048优化后本地仍比TOKEN慢、wire仍更大，无正带宽交叉点。

N512本地由legacy比TOKEN慢3.986ms，变为packed比TOKEN快14.959ms；wire仍多6,294,911B。中位数数学交叉点 **3366.53Mbps（约3.37Gbps）**。

|Mbps|N512 TOKEN总ms|N512 packed V0总ms|V0−TOKEN|
|---:|---:|---:|---:|
|20|103.254|2606.260|+2503.006|
|100|102.596|591.230|+488.634|
|1000|102.447|137.848|+35.400|
|10000|102.433|92.510|−9.923|

同轮5对本地KV−TOKEN为 **[−14.432,−18.396,−1.092,−14.095,−14.702]ms**，均为本地收益；但各对数学crossing为 **3.49/2.74/46.10/3.57/3.43Gbps**。10Gbps时 **4/5** 对预测KV更快，一对反而慢约3.94ms。故出现了**有实际本地收益支持的中位数数学交叉点**，而非旧“归零某阶段”的假想下界；目前仍不足声称稳定、保证或物理可信的3.37Gbps阈值。没有拼接独立极值、丢慢样本或为展示强迫AUTO选KV。

## 15. 新Profile和适用域

旧8.2A/8.3A Profile全部保留未覆盖。新增4个 `83b-packed-point-domain-v1` / measured_incremental 研究Profile，每点TOKEN/V0各5个steady，sample_count10；记录实际模型/revision/dtype/device身份、来源、精确N、A+B Cache总和、各端Cache与EOS/pending。没有startup估计、V1–V3拟合或跨长度外推，没有安装到正式AUTO。

现有Selector schema只守N与Cache总和，不能完整守各端/位置/终止状态；调用者还需人工核对精确点域和framing，故新Profile仍仅研究离线用。实现的固定byte分派不保证所有域内其他模型相同收益。N512只是一个冻结真实前缀，不能外推1024+、3B、CUDA或不同消息质量。

## 16. 正确性与工程验收

新增 **54项测试**：BF16全部65536位模式、FP16/FP32、1/2/32/512/16384Token工程尺寸、连续/非连续stride、逐字节packet/恢复、构造前不重叠views/构造后独立storage、删除payload/GC后修改、独立restore、多轮双向/释放、V0–V3 EOS/pending与后续tiny真实forward、checksum/contract/append/closing/RoPE/Sender close/ledger失败回滚、stack/upload失败无旧路重试、原空KV/容量guard，以及输入SHA篡改/统计/实测分支源码漂移拒绝。tiny CPU测试只证明工程契约，性能全部来自上述预训练真实MPS。

最终 **1029 passed / 1既有可选AutoGen skipped / 2既有依赖warning**。全量包含KV Cache、Protocol、Restore、Handoff事务、TOKEN/V0–V3、Resource Guard、旧性能Harness；compileall（src/tests/全部根Python）和新run/export、原performance、Discussion、Dashboard CLI help通过。diff检查、全部raw/原83A输入SHA、证据SHA、逐次统计重算、独占阶段之和与完整边界核查通过。前端和API未改，按范围不重跑浏览器矩阵。

第一次工程测试误假设公开Cache构造后仍共享storage，30项新断言失败；检查安装源码后改为分别验证输入views和最终独立Tensor，没有放宽旧断言。首次真实启动把dataclass tuple与JSON list直接比较，身份误拒绝、handoff0；归一JSON后逐字段严格相等，失败证据保留，新子目录成功。只读导出首次错误读取receipt不存在的fallback_reason字段，修正为验证实际KV策略/协议/forward与完成记录，没有改raw样本或重跑模型。

## 17. 启用决定与未解决问题

有代表条件下完整收益约18–20ms、全部核心新旧配对方向为正，严格bits/协议/事务通过，选择有界默认启用。默认auto只覆盖MPS/BF16/raw≤6MiB，均匀shape由旧验证保证，uniform dtype新增分派保护；域外保留legacy。显式packed任何异常不重试，原TOKEN默认/质量过滤/实际Selector保持。

剩余主要费用是Receiver两次本地framing、pending补算、全历史clone/cat及V1逐层guards/数学；本轮没动这些。未测纯DMA、Host瞬时highwater、物理链路/并行重叠、1024+内容、其他模型/设备；没有得到一般质量可靠性或稳健AUTO交叉阈值。

最终默认分派在测量后由legacy改为auto并收窄到上述域；没有重新加载模型重复跑矩阵。审计保存实测源码，验证legacy/packed执行分支AST完全一致，当前每个实测点的默认分派确实选packed；唯一生产漂移是默认/适用域分派和文档。不会将默认选择验证冒充新一轮MPS计时。

## 18. 复现和8.3C建议

```bash
# 需要本机8.3A原实际Cache文件；没有文件时明确失败，不自动下载或用IDs伪造。
HF_HUB_OFFLINE=1 .venv/bin/python run_kv_transfer_validation.py \
  --input-dir benchmark-results/performance-83a \
  --output-dir benchmark-results/kv-transfer-83b-new --warmup 2 --runs 5
.venv/bin/python export_kv_transfer_validation.py \
  --results-dir benchmark-results/kv-transfer-83b-new \
  --output-dir /tmp/kv-transfer-83b-new-evidence
HF_HUB_OFFLINE=1 .venv/bin/pytest -o addopts='' -q
.venv/bin/python -m compileall -q src tests *.py
```

所有运行/导出入口拒绝已存在输出目录，原冻结输入与当前源码漂移检查保持。未来复现若模型或源码改变，应新建实验/显式说明身份，不能更新旧SHA假装当时已用新实现。`engineering.json`和证据总SHA是最终验收附加项，导出时不加载模型。

8.3C优先另做精确A/B历史、EOS/pending/位置域的成本Profile及少量独立长消息holdout，验证3–10Gbps数学优势的波动边界；真实网络另立边界。然后只选pending复用或Receiver framing中的一个点，保留完整事务与同快照测量。V1 guards和Cache cat可分别研究，但不以本轮结果预承诺收益，不重启Prompt搜索/质量矩阵或Dashboard重构。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

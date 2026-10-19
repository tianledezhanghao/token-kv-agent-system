# 新实验配置简化验收

2026-10-19。只修改前端新实验配置与少量辅助文案，后端策略、协议、Shannon、KV、候选合法性、成本公式均未修改。

## 行为

- 首次加载、恢复历史后的新实验草稿、点击「新实验」：AUTO / performance_experiment / KV V0 / 理想化仿真开启。六系数仍为 a=20、b=2、c=0.002、d=5、e=0.05、f=0.2。
- AUTO 策略控件及标签移除；新实验配置组装固定提交 performance_experiment，即使输入草稿仍带 quality_first。
- 通信模式、KV 候选、网络、任务、模型、六系数仍可调整。关闭仿真仍要求合法 Profile，并保留身份、来源、V3 许可等校验。未改8192上下文默认或保护逻辑。
- 配置分区切换、通信模式往返、关闭/「保留草稿」保留编辑；点击「新实验」恢复上述通信默认，其他任务和网络草稿沿用原机制。
- Session 冻结配置、历史读取转换、决策记录与回放保持原值；新默认只应用到新实验草稿，不改写 Session。
- 上下文帮助移至对应输入框下方，保留三列输入；最终总结 Agent 自然排在下一行。底部只显示「轮数 · AUTO · 理想化仿真」，高级假设、详情和决策中的 Synthetic 来源标识保留。

## 自动检查

- 全量 Python：1318 passed / 1 skipped，2条既有上游弃用 warning，退出码0。
- 前端：249 passed / 11 files；覆盖默认提交、手动 TOKEN/V0–V3/MIXED、Profile 校验与导入、Measured Profile 配置、旧策略归一化、草稿保留、全部自定义系数、历史冻结配置。
- TypeScript / Vite production build、git diff --check通过。

## 真实浏览器

Chrome headless，1440×900网页视口。使用隔离8023后端与5193前端，现有8000/5173未停止或重启。后端为现有测试的 tiny Qwen2 CPU 工程夹具，真实执行生成/协议/接收；不是预训练模型质量或实测性能验证。为适配tiny上下文，浏览器启动实验调整为1轮、8生成Tokens、2048上下文；首次8192默认另行断言。

validation.json（原研发资料，未随公开副本提供：`validation.json`） 保存10组DOM几何及3次浏览器实际POST：

1. 默认AUTO：performance_experiment、token/kv:v0、idealized_simulation=true、原六系数、cost_profile=null。
2. 自定义：KV V1、六系数全部7；切换分区/通信模式及保留草稿后实际提交一致。
3. 关闭仿真：idealized_simulation=false、synthetic_assumptions=null，原Profile JSON完整提交。浏览器导入显式synthetic/test_only工程夹具；Measured Profile合法配置与身份验证由前后端测试覆盖，不将夹具称为实测。

三场均202接受并完成。另从原API创建quality_first历史场次；刷新、回放保持quality_first与关闭仿真原值，其新实验草稿显示AUTO/V0/仿真开启。history-validation.json（原研发资料，未随公开副本提供：`history-validation.json`）保存冻结配置与布局证据。

页面document始终1440×900，无横向或页面纵向溢出；弹窗内部滚动，固定底部完整可见。1360px容器上限、900px阅读列/850px正文、18px/32.4px、暗色主题及内部滚动保持。无JS异常。

截图均为真实浏览器网页视口，不包含原生浏览器外壳：

- [任务页](task-1440.png)、任务页滚动后（原研发资料，未随公开副本提供：`task-scrolled-1440.png`）
- [AUTO配置](auto-1440.png)、关闭仿真后的Profile校验（原研发资料，未随公开副本提供：`measured-1440.png`）
- 历史quality_first回放（原研发资料，未随公开副本提供：`history-1440.png`）

临时验收服务完成后关闭；用户原始资料、既有修改和未跟踪文件保留，仅提交本任务文件，不push。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。

# 独立公开副本发布审查

2026-10-19（Asia/Shanghai）。本副本从已验收的当前磁盘工作区复制，包含原 HEAD `1e3c7ed` 的最新 UI，以及工作区未提交的 README、脚本归整和前端依赖补丁。原研发库 76 条历史不导入；公开仓库使用独立初始提交。

## 范围与脱敏

- 包含 `src/`、`frontend/` 源码与 npm 锁文件、`tests/`、`scripts/`、`requirements/`、筛选的 `docs/`、两个 `experiments/` 报告辅助程序、兼容启动入口、README 与 DESIGN。
- 排除 `.git/`、虚拟环境、依赖安装目录、构建输出、权重、Tensor/抓包、全部 benchmark raw、用户 exports、开发交接、第三方技能包、未审查图片及历史服务器部署资料。
- 原审计 113 条发现逐项定位与处置见 [privacy-disposition.json](privacy-disposition.json)。所选 Shannon 文档删除无关本地清理/远端部署交接段落；AUTO 文档省略历史服务器操作说明，不改写技术结果。
- 一个为回归测试保留的历史 Measured Profile 中，四处个人缓存路径改为 `REDACTED_LOCAL_MODEL/snapshots/<revision>`。数值不变；身份已脱敏，不能当作可直接导入本机的实测校准。
- 筛选文档中未公开原始证据的引用改为明确文字标记，保留原相对路径及阶段结论；不伪造缺失数据，不声称 clone 可重放完整历史研究。
- 三张历史 Dashboard 截图和两张最新配置 UI 截图逐字节复制，逐张目视检查未见个人路径、凭据或私人浏览器外壳；原图与技术输出均不修改。

## 安全审查

Gitleaks 8.30.1 默认完整规则扫描实际发布文件，未关闭规则、未添加秘密忽略规则、未改写代码来掩盖命中。扫描报告有 6 处 `generic-api-key` 命中，经源码上下文人工复核均为误报：

| 文件 | 行 | 判定依据 |
| --- | --- | --- |
| `run_server_readiness.py` | 79 | `token_wire_bytes_for_8` 字节长度字段及相邻程序表达式，不是认证信息 |
| `src/token_kv_agent_system/private_exchange_validation_export.py` | 11–14 | 四个冻结实验 JSON 文件的 SHA256，供 `file_digest` 校验来源；不是凭据 |
| `tests/test_kv_cache_restore.py` | 174 | KV Tensor 的 `keys.dtype` / `keys.device` 断言，不是密钥 |

附加本地审查扫描所有候选文件字节：个人绝对路径、已知实验室标识、私钥标头、带凭据 URL、SSH 登录目标、内网地址及大文件/模型扩展均无未解决命中。macOS 用户目录前缀在本地只读检查脚本中作为泛化路径检测字符串保留，不包含用户名或个人目录。规则扫描与目视审查不能保证识别所有未知格式风险。

无项目级 LICENSE；未发现需随本次源码副本再分发的第三方库或模型资产。依赖安装物与第三方技能未入仓库；已核对 Python 安装包元数据、npm 锁文件许可字段及界面素材来源，见 [第三方说明](../../THIRD_PARTY_NOTICES.md)。本次没有新增 MIT 或其他开源授权。

## 独立副本验收

| 项目 | 结果 |
| --- | --- |
| Python 干净虚拟环境 | 使用 README 的 macOS/Python 3.14 constraints 安装 `.[dev,llm]` 成功；`pip check` 无破损依赖 |
| Python 全量测试 | **1317 passed / 2 skipped**；其中一个为原有可选测试，一个依赖未公开的本地忽略 Profile；不复制 raw 以填补它 |
| 前端 | 干净 `npm ci`；**249 tests passed / 11 files**；TypeScript + Vite build 成功；npm audit **0 vulnerabilities** |
| Mock 启动 | 独立端口启动、health 200、OpenAPI、前端页面、API 代理与 WebSocket 握手通过；自有进程正常关闭 |
| 正式讨论启动 | 现有离线 Qwen2.5-0.5B / MPS 缓存，独立端口 health 200；channel 与 synthetic_assumptions schema 存在；0 次新生成，正常退出 0 |
| 入口与导入 | Dashboard / Discussion / 校准 CLI help、无模型 demo、源码 compileall 通过 |
| 最新代码保真 | 136 个运行源码文件、42 个前端源码文件与当前研发工作区逐字节一致 |
| 回归测试独立性 | 原测试依赖 `git show fe888ee:...`；改用从原 blob 独立提取的提示词/字符串夹具，保留原 equality 断言和来源 SHA；不引入旧 Git 历史 |
| 文档与图片 | 内部路径/静态锚点检查通过；所有展示图片均在副本内，含最新 AUTO 配置 |

本次未下载模型或重跑研究性能/质量实验；其他平台/CUDA/物理无线与首次联网模型下载未重新验证。公开副本测试使用真实协议 tiny/fixture 路径，不将其作为预训练模型质量或普遍 KV 加速证据。原研发库、用户运行服务和已有研究输出不修改。

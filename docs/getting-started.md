# 安装与启动

[返回首页](../README.md) · [复现方法](reproducibility.md) · [本轮实测](release/public-release.md)

## 选择入口

| 目的 | 后端命令（仓库根目录） | 是否需要权重 |
| --- | --- | --- |
| 正式双 Agent 讨论、V0–V3、Shannon、AUTO | `python run_dashboard.py --discussion --device mps` | 需要；仅 Mac MPS 离线 Worker |
| 无模型交互示例 | `python run_dashboard.py --interactive` | 不需要；前端切换到“通信基准” |
| 无模型自动示例 | `python run_dashboard.py` | 不需要；默认延迟 1 秒运行 Mock 示例 |
| 空观察后端 | `python run_dashboard.py --no-demo` | 不需要；不监听其他独立进程中的 Demo |
| Canonical 固定真实 TOKEN / KV | `python run_dashboard.py --real-token --device mps` / `--real-kv` | 需要；与讨论增量路径及 Profile 不同 |

安装后也可用 `token-kv-dashboard` 代替 `python run_dashboard.py`。正式讨论入口默认参数 `--device auto` **不满足 MPS 强制要求**，必须显式写 `--device mps`。`--max-new-tokens` 是 Canonical runtime 参数；讨论预算在 UI / Session 请求中配置。

## 安装依赖

项目 Python 元数据声明 `>=3.10`，本轮在 Python 3.14.6 / macOS arm64 的干净虚拟环境验证基础、dev 与 llm 依赖。真实 KV adapter 使用当前 Transformers Cache API；声明范围不等于所有依赖组合都已验证。

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/tested-macos-py314.txt -e '.[dev,llm]'
```

命令中的 constraints 适用于本轮验证平台。其他平台或 Python 版本使用 `python -m pip install -e '.[dev,llm]'` 后另做测试。仅运行 Mock 时可用 `python -m pip install -e '.[dev]'`；包基础依赖只有 FastAPI、Uvicorn 与 WebSocket，LLM 库是 optional extra。

若 `.venv` 已存在，不覆盖它；先检查其中 Python/依赖或自行使用另一环境目录。所有后端命令均使用完成 editable install 的 Python。

前端 Node 范围是 `^20.19.0 || ^22.12.0 || >=24.0.0`，由 Vite、React 插件与 Vitest 的共同要求得到。本轮 Node 26.0.0、npm 11.12.1。优先使用锁文件：

```bash
node --version
npm --version
cd frontend
npm ci
cd ..
```

`npm ci` 用于全新前端依赖安装，会重建 `node_modules`；有个人依赖修改时先自行保存。不要用旧 Node 或通过忽略 engines 掩盖错误。

## 模型准备与离线加载

默认模型为 `Qwen/Qwen2.5-0.5B-Instruct`。历史主验收使用 revision `7ae557604adf67be50417f59c2c2f167def9a775`；需联网下载一次，随后运行时不联网：

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775'))"
```

下载位置由 Hugging Face 缓存规则及 `HF_HOME` / `HF_HUB_CACHE` 决定，不复制到仓库。0.5B 主权重约 1 GB，运行还需要 Cache、临时 Tensor 和框架内存；可用内存不足时保护机制可能拒绝。1.5B / 3B 不是 Quick Start 必需。

若需要精确锁定快照，先离线解析下载结果，再将该目录传给 `--model`（下面不是个人固定路径）：

```bash
MODEL_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_files_only=True))")
python run_dashboard.py --discussion --device mps --model "$MODEL_DIR"
```

目录必须包含 `config.json`、`tokenizer_config.json`、`tokenizer.json` 及全部 safetensors 权重/分片。以默认模型名启动时，`cached_model()` 优先使用本地 `refs/main`；没有该引用时要求只找到一个完整快照。多个完整快照或错误 main 引用时，显式路径更可靠。

讨论服务设置 `HF_HUB_OFFLINE=1`，Worker 还禁用 telemetry、MPS fallback。模型菜单只解析已有本地完整缓存，没有自动下载选项。通用 `--real-token`、`--real-kv` 和部分历史模型 Demo 使用 `from_pretrained`，联网情况下可能自动下载；这与正式讨论入口不同。讨论 CLI 的下载必须显式 `--allow-download`。

本轮没有从网络重新下载权重；下载 API 参数已核对，固定 revision 的缓存解析与 MPS 启动另有实测。网络、镜像、代理、远端权限及其他模型首次下载未验证。

## 两个终端运行

终端一从仓库根目录执行，等待模型加载完成及 Uvicorn ready：

```bash
source .venv/bin/activate
python run_dashboard.py --discussion --device mps
```

终端二从仓库根目录执行：

```bash
cd frontend
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`；后端 `http://127.0.0.1:8000/api/health` 返回健康状态，`http://127.0.0.1:8000/docs` 提供 OpenAPI。后端健康不等于模型讨论已认证正确，请核对顶栏模型和讨论错误。开始短 TOKEN 实验后，再选择 KV 或 AUTO。[配置及复现](reproducibility.md)。

端口被占用时，优先保留原服务，使用成对的新地址：

```bash
# 终端一：仓库根目录
python run_dashboard.py --discussion --device mps --port 8001
```

```bash
# 终端二：frontend 目录
DASHBOARD_LOCAL_URL=http://127.0.0.1:8001 npm run dev -- --port 5174 --strictPort
```

访问 `http://127.0.0.1:5174`。只修改前端端口不会改变 API 代理目标。开发代理配置见 [vite.config.ts](../frontend/vite.config.ts)。`npm run build` 生成静态构建，但 `npm run preview` 不能直接当作已配置完整 API / WS 代理的正式部署方案。

## 正常停止

在前端终端按 Ctrl+C 停止 Vite，在后端终端按 Ctrl+C 停止 API 并等待模型 Worker 关闭。停止活动讨论会丢失未持久化状态；UI 当前没有通用取消/暂停接口。需要保留结果时，先保存完整 Session / 事件。服务重启不恢复旧 Cache，也不承诺恢复最近三次内存历史。

## 常见错误

| 现象 | 核查与处理 |
| --- | --- |
| `ModuleNotFoundError` / 缺少 llm optional dependencies | 激活正确环境；用同一 Python 完成 editable install 与 llm extra |
| `Discussion model worker requires MPS` | 正式讨论必须 `--device mps` 且机器有 MPS；无 MPS 可使用 Mock 或独立 CLI 的 CPU 路径，不能宣称正式 Dashboard 支持 CPU |
| `Expected one complete local snapshot` | 检查缓存目录、分片、refs/main；先单独下载，再使用固定 revision 的显式本地目录 |
| 模型加载 / 内存预算拒绝 | 减小模型、轮数、生成/上下文预算；检查自己的其他进程，不绕过保护或偷偷回退设备 |
| 5173 自动跳到其他端口 / API 代理失败 | 阅读 Vite 实际地址；使用 `--strictPort` 与匹配的 `DASHBOARD_LOCAL_URL` |
| 503 | 模型不可用/Worker 加载失败；读后端异常及顶栏模型状态，不反复提交实验 |
| 409 | 活动实验、模型切换或历史会话生命周期冲突；先确认活动状态 |
| 422（尤其 `synthetic_assumptions` 未识别） | 检查运行进程是否仍加载旧后端 schema；先保存结果，再正常重启同一版本；不要自动提高 Token 上限规避 |
| 回答错误、复述、`max_new_tokens` | 这可能是模型/预算局限；工程 receipt 成功不证明语义正确，保留原文与终止原因 |

## 测试

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest
cd frontend
npm test
npm run build
```

测试使用 tiny/fixture 与真实协议工程路径，不要求下载预训练模型。真实模型启动烟测与历史性能/质量证据分开记录。

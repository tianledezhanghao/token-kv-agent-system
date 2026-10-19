# 脚本说明

从仓库根目录运行，先按 [安装指南](../docs/getting-started.md) 安装本项目。

- `run_dashboard.py`：正式 Dashboard / Mock 入口。
- `run_demo.py`：无需权重的通信示例。
- `run_discussion_demo.py`：真实模型讨论，参见 [复现方法](../docs/reproducibility.md)。
- `run_generalized_calibration.py`：设备与模型身份绑定的校准；不内置通用 Profile。
- `analysis/`、`exports/`、`legacy/`：原根目录兼容包装调用的分析、导出与历史诊断实现。
- `maintenance/check_doc_links.py`：只读检查本副本内部文档链接。

根目录其他 `run_*.py` / `export_*.py` 保留研究入口兼容性；`run_server_readiness.py` 是供回归测试使用的本地前提检查，不含远端地址、SSH 或部署凭据。分布式/Relay 代码仍是研究原型，Quick Start 不需要部署服务器。历史 exporter 可能要求未公开的 raw 与原始源码 SHA；不能据此声称本副本可重放完整历史实验。

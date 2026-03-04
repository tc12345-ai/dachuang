# Dachuang 信号处理与插件化平台

这是一个以 Python 为核心的信号处理/分析项目，采用模块化与插件化架构，覆盖：

- 核心算法与数据模型（`core/`）
- 图形界面（`gui/`）
- HTTP API（`api/`）
- 插件生态（`plugins/`）
- I/O 与工程管理（`io_manager/`）
- 自动化测试（`tests/`）

## 项目结构

```text
.
├── api/                # 服务端接口
├── core/               # 事件总线、算法、插件管理等核心逻辑
├── gui/                # Tk/界面相关面板
├── io_manager/         # 导入导出与项目读写
├── plugins/            # 各业务插件
├── tests/              # 单元测试与测试入口
├── utils/              # 工具函数
└── main.py             # 程序入口
```

## 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

## 测试

```bash
python -m pytest -q
```

## 插件说明（简要）

- 插件目录位于 `plugins/`，每个插件包含 `manifest.json`。
- `core/plugin_manager.py` 负责插件发现、加载、卸载及 UI/API 挂载。
- 插件间通过 `core/event_bus.py` 事件总线通信，降低直接耦合。

## 许可证

当前仓库未显式声明 LICENSE；如需开源发布，建议补充许可证文件。

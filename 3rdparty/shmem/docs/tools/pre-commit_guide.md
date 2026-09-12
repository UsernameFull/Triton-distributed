# Pre-commit 代码检查使用指南

## 介绍

本项目使用 [pre-commit](https://pre-commit.com/) 框架在代码提交前自动执行代码质量检查，确保代码风格一致、无常见错误。

## 安装

### 1. 安装 pre-commit

```bash
pip install pre-commit
```

### 2. 安装 git hooks（推荐）

```bash
pre-commit install
```

安装后，每次 `git commit` 会自动运行检查。

## 使用方式

### 自动检查（推荐）

安装 git hooks 后，每次提交代码会自动触发检查：

```bash
git add .
git commit -m "your message"
```

如果检查失败，部分工具会自动修复（如 ruff-format、clang-format），修复后重新提交即可。

### 手动检查

检查暂存的文件：

```bash
pre-commit run
```

检查指定文件：

```bash
pre-commit run ruff-check --files path/to/file.py
pre-commit run clang-format --files path/to/file.cpp
```

检查单个 hook：

```bash
pre-commit run ruff-check
pre-commit run pylint
pre-commit run clang-format
```

### 跳过检查（不推荐）

```bash
git commit --no-verify -m "your message"
```

## 检查工具说明

| 工具 | 语言 | 功能 | 配置文件 |
|------|------|------|----------|
| ruff | Python | 代码格式化 + Lint | `tools/pre-commit/pyproject.toml` |
| pylint | Python | 代码质量检查 | `tools/pre-commit/pyproject.toml` |
| bandit | Python | 安全漏洞检查 | `tools/pre-commit/pyproject.toml` |
| codespell | 通用 | 拼写检查 | `.pre-commit-config.yaml` |
| typos | 通用 | 拼写检查 | `tools/pre-commit/typos.toml` |
| clang-format | C/C++ | 代码格式化 | `.clang-format` |

## 配置文件说明

### 主配置文件

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) - 定义要运行的检查工具和参数

### Python 工具配置

[`tools/pre-commit/pyproject.toml`](../../tools/pre-commit/pyproject.toml) - ruff、pylint、bandit 的规则配置

### C++ 格式化配置

[`.clang-format`](../../.clang-format) - clang-format 的格式化规则

### 拼写检查白名单

[`tools/pre-commit/typos.toml`](../../tools/pre-commit/typos.toml) - typos 工具的误报白名单

## 常见问题

### Q: 检查失败怎么办？

部分工具支持自动修复（如 ruff-format、clang-format），直接重新提交即可。对于需要手动修复的问题，根据错误提示修改代码后重新提交。

### Q: 如何更新 pre-commit hooks？

```bash
pre-commit autoupdate
```

### Q: 如何查看某个工具的详细错误信息？

```bash
pre-commit run pylint --verbose
```

### Q: 如何临时禁用某条规则？

**Python (ruff/pylint):** 在代码行尾添加注释

```python
x = 1  # pylint: disable=invalid-name
```

**C++ (clang-format):** 使用注释包围

```cpp
// clang-format off
int unformatted_code = 1;
// clang-format on
```

### Q: 首次运行很慢怎么办？

首次运行需要下载和安装各个检查工具的环境，之后会使用缓存，速度会快很多。

## 最佳实践

1. **安装 git hooks**：使用 `pre-commit install` 自动检查每次提交
2. **不要频繁使用 `--no-verify`**：跳过检查可能导致问题代码进入仓库
3. **及时更新 hooks**：定期运行 `pre-commit autoupdate` 获取最新版本
4. **配置 IDE 集成**：在 IDE 中配置 ruff、clang-format 插件，实时检查

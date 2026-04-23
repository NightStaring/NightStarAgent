# NightStarAgent

完整 Agent 项目框架。当前已落地模块是 `RAGKnowledge`（离线知识库构建），其余 Agent 能力将逐步补充。

## 当前已实现：RAG 离线知识库构建

本项目用于将本地文档离线处理并写入 Qdrant，构建可用于 RAG 检索的知识库。

## 功能概览

- 文档解析：支持 `docx` / `md` / `pdf` / 图片
- 文本切分：将长文本切成可检索 chunk
- 向量编码：
  - 文本 dense 向量 + BM25 稀疏向量
  - 图片/页面向量
  - OCR / Caption 文本增强
- 向量入库：写入 Qdrant 文本、图片、页面集合

## 核心流程

1. 解析文档，产出：
   - `*.md`（文本）
   - `images/`（抽取图片）
   - `pages/`（页面预览图，可选）
2. 文本切分与向量化
3. 多模态数据入库到 Qdrant

## 快速开始

### 1) 配置 Qdrant

编辑 `RAGKnowledge/config/config.yaml`：

```yaml
vector_db:
  db_type: Qdrant
  text_collection: Chart_MRAG_text
  image_collection: Chart_MRAG_image
  page_collection: Chart_MRAG_page
  text_dimension: 1024
  image_dimension: 4096
  page_dimension: 4096
  Qdrant:
    url: http://127.0.0.1
    port: 6333
    api_key: ${QDRANT_API_KEY}
```

建议在 `.env` 中设置：

```env
QDRANT_API_KEY=...
API_TIMEOUT=60
CHUNK_TYPE=default
```

### 2) 执行离线构建

```bash
python -m RAGKnowledge.run
```

Windows 可双击：

- `RAGKnowledge/run_click.bat`

运行日志会写入 `logs/run_yyyyMMdd_HHmmss.log`。

## 清空向量集合

```bash
python -m RAGKnowledge.clear_db
```

## 目录说明

- `RAGKnowledge/run.py`：入口脚本
- `RAGKnowledge/file_processer.py`：离线构建主流程
- `RAGKnowledge/utils/file_paser.py`：文档解析
- `RAGKnowledge/utils/text_splitter.py`：文本切分
- `RAGKnowledge/vector_db/`：Qdrant 封装
- `RAGKnowledge/config/`：配置加载与管理

## 预留扩展（Agent 其他模块）

本仓库定位为完整 Agent 工程，除 RAG 离线构建外，后续将按模块逐步补齐，例如：

- 在线检索与召回编排
- 对话/推理与工具调用
- 任务调度与工作流
- 监控、评测与反馈闭环
- 前后端服务化与部署

当前 README 先聚焦已可运行的 `RAGKnowledge`，后续会按模块分章节扩展。

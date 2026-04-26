# NightStarAgent

NightStarAgent 是一个完整 Agent 工程的主仓库。  
当前已落地并可运行的核心模块是 `RAGKnowledge`，其余能力（后端服务、前端应用、评测与运维）按阶段持续补齐。

本项目面向公司经营分析场景：将财报解读、业务周报、市场研报、会议纪要和图表类资料统一入库，  
通过 Agent 问答快速定位证据并输出结构化分析结论，帮助业务和管理层更高效地完成信息整合与决策支持。

---

## 一、项目定位

- **目标**：构建可工程化落地的 Agent 系统，而非单一脚本。
- **当前阶段**：优先完成 RAG 知识库构建与检索问答链路。
- **演进方向**：从“离线建库”扩展到“在线服务 + 前端交互 + 可观测与评测闭环”。

---

## 二、当前已实现模块（RAGKnowledge）

`RAGKnowledge` 已支持离线知识库构建与 Agentic RAG 问答主链路。

### 2.1 已实现能力

- 文档解析：`docx` / `md` / `pdf` / 图片
- 文本切分：默认切分 + Markdown 结构切分
- 向量编码：
  - 文本 dense 向量
  - 文本 sparse 向量（BM25）
  - 图像 / 页面向量
- 增强处理：OCR / Caption
- 向量入库：Qdrant 三集合（`text` / `image` / `page`）
- 在线问答：多轮子问题扩展、并行检索、摘要、重排、LLM/VLM 回答

### 2.2 核心流程

1. 文档解析，产出 `md/images/pages`
2. 文本切分与向量化
3. 多模态数据写入 Qdrant
4. 检索、重排与回答生成

---

## 三、快速开始（RAG 模块）

### 3.1 配置向量库

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

建议在 `.env` 中至少设置：

```env
QDRANT_API_KEY=
SILICONFLOW_API=
API_TIMEOUT=60
CHUNK_TYPE=markdown
```

### 3.2 运行方式

```bash
python -m RAGKnowledge.run
```

日志将写入 `logs/run_yyyyMMdd_HHmmss.log`。

### 3.3 清空集合

```bash
python -m RAGKnowledge.clear_db
```

---

## 四、项目结构（当前）

### 4.1 RAG 目录

- `RAGKnowledge/run.py`：RAG 运行入口（建库/问答）
- `RAGKnowledge/file_processer.py`：离线处理主流程
- `RAGKnowledge/retrieval/`：多路召回
- `RAGKnowledge/rerank/`：重排逻辑
- `RAGKnowledge/agent.py`：Agentic RAG 调度
- `RAGKnowledge/vector_db/`：向量库抽象与 Qdrant 实现
- `RAGKnowledge/config/`：配置加载

### 4.2 详细文档

- `RAGKnowledge/README.md`（RAG 模块完整说明）

---

## 五、后端模块（预留）

> 预留位置：后续补充 API 服务化、鉴权、任务编排与部署相关内容。

### 5.1 模块目标

- 对外提供统一 API（问答、检索、文档入库、任务状态）
- 提供异步任务能力（大文件解析、批量建库）
- 支持配置中心、限流、审计日志与可观测

### 5.2 预留目录建议

```text
backend/
|-- app/
|-- api/
|-- services/
|-- workers/
|-- tests/
```

### 5.3 预留文档小节

- API 设计规范（OpenAPI/接口契约）
- 服务部署（Docker/K8s）
- 日志与监控（Tracing/Metrics）

---

## 六、前端模块（预留）

> 预留位置：后续补充 Web 控制台与知识库运营界面。

### 6.1 模块目标

- 提供对话工作台（多轮问答、引用证据展示）
- 提供知识库管理（上传、解析进度、检索调试）
- 提供评测与反馈入口（人工标注、问题回流）

### 6.2 预留目录建议

```text
frontend/
|-- src/
|-- components/
|-- pages/
|-- services/
|-- tests/
```

### 6.3 预留文档小节

- 技术栈说明（框架、状态管理、构建方式）
- 页面路由与信息架构
- 前后端接口对接约定

---

## 七、版本路线（预留）

- **阶段 1（当前）**：RAG 离线建库 + Agentic RAG 主流程
- **阶段 2**：后端 API 服务化 + 任务调度
- **阶段 3**：前端工作台 + 可观测与评测闭环

---

## 八、参考项目

- [joyagent-jdgenie](https://github.com/jd-opensource/joyagent-jdgenie)

说明：本项目在工程思路上参考了上述开源项目，并结合当前业务目标做了结构化适配。

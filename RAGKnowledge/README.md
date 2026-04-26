# RAGKnowledge

`RAGKnowledge` 是当前项目中的 RAG 模块，负责“离线建库 + 在线检索问答”两部分能力。  
目标是把多类型文档处理为可检索的多模态知识库，并通过 Agentic RAG 进行回答生成。

> 说明：本模块在设计思路与工程实现上，明确参考了开源项目 [joyagent-jdgenie](https://github.com/jd-opensource/joyagent-jdgenie)，并结合当前项目结构做了适配与扩展。

---

## 1. 能力概览

当前已实现：

- 文档解析：`DOCX / Markdown / PDF / 图片`
- 文本切分：默认切分 + Markdown 标题切分
- 向量编码：
  - 文本稠密向量（dense）
  - 文本稀疏向量（BM25 sparse）
  - 图像/页面向量
- 图像增强：OCR / Caption
- 向量入库：Qdrant 三个集合（text / image / page）
- 在线问答：Agentic RAG（子问题扩展、并行检索、摘要、重排、LLM/VLM 作答）

---

## 2. 快速开始

### 2.1 环境准备

至少确认以下依赖与服务可用：

- Python 环境（建议 `agent` conda 环境）
- Qdrant 服务
- `.env` 中可用的模型 API Key（如 `SILICONFLOW_API`）

### 2.2 常用运行方式

```bash
python -m RAGKnowledge.run
```

---

## 3. 运行入口说明（`run.py`）

`RAGKnowledge/run.py` 中常用函数：

- `show_db_info()`：查看当前集合列表
- `create_db()`：创建配置中的集合
- `clear_db()`：删除配置中的集合
- `process_documents()`：执行离线文档处理并写入向量库
- `agent()`：运行一次示例问答（Agentic RAG）

当前 `python -m RAGKnowledge.run` 默认执行的是 `agent()`。  
如果你要先建库，请在 `run.py` 中切换调用顺序（例如先 `create_db()`，再 `process_documents()`）。

默认数据路径（当前代码）：

- 输入目录：`RAGKnowledge/dataset/Chart-MRAG/files`
- 工作目录：`RAGKnowledge/dataset/Chart-MRAG/knowledge/<文件名去后缀>/`

---

## 4. 配置说明

### 4.1 主配置文件

`RAGKnowledge/config/config.yaml`（示例）：

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
    port: 6334
    api_key: ${QDRANT_API_KEY}
```

### 4.2 配置加载规则

- 由 `config/config.py` 读取并缓存
- 自动加载项目根目录 `.env`
- 支持占位符：
  - `${VAR}`
  - `${VAR:default}`

---

## 5. 离线建库流程

离线主流程由 `file_processer.py` 组织，典型步骤：

1. 解析原始文件为标准中间产物（文本、图片、页面）
2. 文本切分为 chunk（可选子块增强）
3. 对文本与图像计算向量（dense / sparse / image）
4. 生成 OCR / Caption 等增强文本
5. 写入 Qdrant（text / image / page 集合）

中间产物目录结构：

```text
work_dir/
|-- <file_name>.md
|-- images/
|-- pages/
```

---

## 6. 文本切分策略

实现位置：`utils/text_splitter.py`

- 默认切分器：
  - `CHUNK_SIZE` 默认 500
  - `CHUNK_OVERLAP` 默认 100
  - 先句级切片，再组装 chunk
- Markdown 切分器：
  - 先按标题层级切分
  - 再按长度合并
  - 超长块回退默认切分器

建议在 `.env` 显式设置：

```env
CHUNK_TYPE=markdown
```

### 子块增强（Sub-chunk Enhance）

可通过环境变量开启：

```env
ENABLE_SUB_CHUNK_ENHANCE=true
SUB_CHUNK_SIZE=100
```

作用：提升短问题、局部事实类查询的命中率。

---

## 7. 在线检索与回答链路

核心文件：`agent.py`

链路概览：

1. 检查是否需要检索（可早退到 LLM/VLM 直答）
2. 子问题扩展（首轮可结合图片描述）
3. 四路并行召回：
   - 文本 dense
   - 文本 sparse（BM25）
   - text->image
   - text->page
4. 子问题级摘要与“是否可答”判断
5. 多轮累积后合并去重
6. 文本 rerank
7. 依据证据类型选择 LLM 或 VLM 生成最终答案

---

## 8. 向量与 Payload 设计

### 8.1 集合分工

- `text_collection`：文本 chunk（dense + sparse）
- `image_collection`：图片相关向量
- `page_collection`：页面图向量

### 8.2 Payload 特点

- 文本类：`text` + 元信息（如 `kb_id / file_id / chunk_type`）
- 图像/页面类：路径、ID、类型等元数据
- 不存原始二进制文件本体，文件内容由路径索引到本地产物

---

## 9. 常见问题

### 9.1 集合不存在（NOT_FOUND）

现象：

- `Collection 'xxx' doesn't exist`

处理：

- 入库前先创建集合（`create_db()`）

### 9.2 文档页面图未生成

可能原因：

- `docx2pdf` / `pdf2image` 等工具链未安装

影响：

- 仅影响页面图相关流程，不阻断文本主流程

### 9.3 模型请求超时或阻塞

可在 `.env` 中设置超时，例如：

```env
API_TIMEOUT=60
```

---

## 10. 常用脚本与模块

- `run.py`：运行入口（建库/清库/问答）
- `clear_db.py`：清空配置中的集合
- `file_processer.py`：离线建库主流程
- `retrieval/`：检索层（文本/图像/路由）
- `rerank/`：重排层
- `agent.py`：Agentic RAG 调度层
- `vector_db/`：向量库抽象与 Qdrant 实现
- `config/config.py`：配置加载与缓存

---

## 11. 参考开源项目

- [joyagent-jdgenie](https://github.com/jd-opensource/joyagent-jdgenie)

当前实现基于该项目的部分 RAG 思路进行落地，主要包括：

- Agentic RAG 的多阶段处理思路（查询判断、子问题扩展、检索、摘要、作答）。
- 检索与重排结合的回答生成流程。
- 面向工程可用性的模块拆分方式（检索层、重排层、提示词层、调度层）。

# RAGKnowledge

本项目用于将多类型文档离线处理后写入 Qdrant，形成可检索的多模态知识库。  
当前主流程覆盖：

- 文档解析（DOCX / Markdown / PDF / 图片）
- 文本切分（默认切分 + Markdown 标题切分）
- 向量编码（文本 dense + BM25 sparse + 图像 embedding）
- OCR / Caption 文本增强
- 向量入库（文本、图片、页面三路集合）

---

## 1. 快速运行

### 1.1 命令行运行

```bash
python -m RAGKnowledge.run
```

### 1.2 双击脚本运行（Windows）

使用 `RAGKnowledge/run_click.bat`：
- 自动切到项目目录
- 尝试激活 `conda` 的 `agent` 环境
- 运行时日志同时打印并写入 `logs/run_yyyyMMdd_HHmmss.log`

---

## 2. 当前运行入口说明

`RAGKnowledge/run.py` 提供以下函数：

- `show_db_info()`：打印当前集合列表
- `create_db()`：创建配置中的三个集合并打印信息
- `clear_db()`：按配置删除三个集合并打印信息
- `main()`：遍历数据目录，执行文档处理并入库

默认处理目录（当前代码）：

- 输入：`RAGKnowledge/dataset/Chart-MRAG/files`
- 输出工作目录：`RAGKnowledge/dataset/Chart-MRAG/knowledge/<文件名去后缀>/`

---

## 3. 配置说明（`config/config.yaml`）

当前主要配置项：

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

说明：
- 配置由 `config/config.py` 读取，支持 `${ENV_VAR}` 占位符。
- 会自动加载项目根目录 `.env`。
- `VectorDB` 为单例，首次初始化后会缓存配置。

---

## 4. 文档解析产物约定

解析统一落到同一工作目录：

```text
work_dir/
|-- <file_name>.md
|-- images/
|-- pages/
```

### 4.1 DOCX

- 提取文本、表格、内嵌图到 Markdown + images
- 尝试将每页转图到 pages（`docx2pdf/libreoffice/unoconv`）
- 页面转换失败不阻断主流程

### 4.2 Markdown

- 复制到标准 md 路径
- 下载 HTTP/HTTPS 图片到 `images/` 并改写链接
- 尝试生成页面预览图（`imgkit` 或 `playwright`）

### 4.3 PDF

- `fitz` 提取文本和内嵌图
- `pdfplumber` 提取表格
- 尝试 `pdf2image` 生成 `pages/`

### 4.4 图片

- Markdown 写空
- 原图复制到 `pages/`

---

## 5. 文本切分策略（`utils/text_splitter.py`）

### 5.1 默认策略

- `CHUNK_SIZE` 默认 500
- `CHUNK_OVERLAP` 默认 100
- 先句级切片（URL/引号保护）再组装 chunk

### 5.2 Markdown 策略

- 先按 `#`~`######` 标题切分
- 再按大小合并
- 超大块回退默认切分器

注意：`get_text_splitter()` 当前依赖 `CHUNK_TYPE`，若环境变量未设置，建议设置为：

```env
CHUNK_TYPE=default
```

### 5.3 子块增强（Sub-chunk Enhance）

`file_processer.py` 的 `_process_text()` 支持可选“子块增强”：

- 主块切分后，可再进行一次更细粒度切分
- 子块也会生成 dense + sparse 向量并入文本集合
- 适合提升短问句、局部事实的召回命中

相关环境变量：

```env
ENABLE_SUB_CHUNK_ENHANCE=true
SUB_CHUNK_SIZE=100
```

说明：
- `ENABLE_SUB_CHUNK_ENHANCE` 开启后生效
- 子块切分器固定使用 `default`，并设置 `chunk_overlap=0`
- 主块与子块会共同存在于文本集合中，检索时统一参与召回
- 当前实现中子块 payload 的 `text` 字段沿用父块文本，向量来自子块文本（用于保留上下文）

---

## 6. 向量入库与载荷（Payload）

### 6.1 三路集合

- 文本集合：`text_collection`
- 图像集合：`image_collection`
- 页面集合：`page_collection`

### 6.2 入库内容

- `text`：文本 chunk 的 dense + sparse 向量
- `image/page`：图像向量
- `ocr_text/caption`：作为文本再入文本集合

### 6.3 Payload 特点

- 文本类通常包含 `text` + 各种业务字段（`kb_id/file_id/chunk_type/...`）
- 图像/页面类包含路径、ID、类型等元数据，不存图片二进制

---

## 7. Qdrant 连接与常见问题

### 7.1 集合不存在（NOT_FOUND）

现象：
- `Collection 'text/image/page' doesn't exist`

原因：
- 未创建集合就开始写入

建议：
- 先执行 `create_db()` 或确保 `create_all_collections()` 在入库前执行

### 7.2 DOCX 页面图未生成

日志提示：
- `docx2pdf 或 pdf2image 未安装`

说明：
- 仅影响 `pages/` 产物，不影响文本和图片抽取主流程

### 7.3 VLM/OCR 请求长时间阻塞

可通过 `.env` 设置：

```env
API_TIMEOUT=60
```

---

## 8. 清库脚本

`RAGKnowledge/clear_db.py`：按当前配置删除 `text/image/page` 集合。

```bash
python -m RAGKnowledge.clear_db
```

---

## 9. 关键模块速览

- `run.py`：主入口与管理函数
- `file_processer.py`：离线处理主流程
- `utils/file_paser.py`：文档解析
- `utils/text_splitter.py`：文本切分
- `vector_db/db.py`：向量库统一接口（单例）
- `vector_db/qdrant_db.py`：Qdrant 具体实现
- `config/config.py`：配置读取与缓存

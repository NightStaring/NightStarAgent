# RAGKnowledge 解析与切分原则

本文档用于总结当前代码中：
- 不同文件类型的解析原则（`utils/file_paser.py`）
- 不同文档切分策略的原则（`utils/text_splitter.py`）

---

## 一、统一处理约定

无论输入是什么文件类型，解析阶段都尽量统一产出到同一工作目录结构：

```text
work_dir/
|-- <file_name>.md        # 解析后的文本主文件（统一为 markdown）
|-- images/               # 从文档中抽取出的图片
|-- pages/                # 页面级预览图（整页截图）
```

后续离线处理（见 `file_processer.py`）会基于这三类产物分别做：
- 文本切分与向量化
- 图片向量化 + OCR/Caption 文本化
- 页面图向量化 + OCR/Caption 文本化

---

## 二、不同文件类型的解析原则

### 1) DOCX（`DocxDocumentParser`）

核心原则：**结构化提取 + 尽量保留语义层级**。

- 先处理 Word 修订痕迹（Track Changes），尽量保留“最终可见内容”。
- 段落文本写入 markdown：
  - 英文标题样式（Heading 1~6）映射为 `#` 到 `######`
  - 中文常见层级（如“一、”“（一）”“1.”）也会尝试映射为标题级别
  - 普通段落按正文写入
- 表格转换为 markdown table 语法。
- 内嵌图片抽取到 `images/`，并在 markdown 插入图片引用。
- 页面预览图优先通过“docx -> pdf -> page images”生成，按以下顺序尝试：
  - `docx2pdf`（Windows）
  - `LibreOffice`
  - `unoconv`

### 2) Markdown（`MarkdownDocumentParser`）

核心原则：**尽量保持原文，同时把远程资源本地化**。

- 若源 md 不在 `work_dir`，会复制到标准位置 `<file_name>.md`。
- 扫描 markdown 中的 HTTP/HTTPS 图片链接并下载到 `images/`。
- 下载成功后，把文内图片链接替换为本地 `images/...` 路径。
- 页面预览优先尝试：
  - `markdown2 + imgkit`
  - `markdown2 + playwright`

### 3) PDF（`PdfParser`）

核心原则：**本地直读优先，文本/表格/图片三路提取**。

- 使用 `fitz` 提取文本与内嵌图片（图片写入 `images/`）。
- 使用 `pdfplumber` 提取表格并追加到 markdown 文本。
- 再使用 `pdf2image` 生成每页预览图到 `pages/`。
- 页面预览失败不会阻断主流程（主流程以文本抽取成功为先）。

### 4) 图片文件（`ImageParser`）

核心原则：**把图片作为页面输入，文本留空**。

- markdown 文件写空字符串。
- 原图复制到 `pages/`，供后续 OCR、Caption、图像向量化使用。
- 当前支持后缀：`.png/.jpg/.jpeg`。

### 5) 文件类型路由

统一入口 `get_document_parser(file_extension)`：
- `.docx` -> `DocxDocumentParser`
- `.md` -> `MarkdownDocumentParser`
- `.pdf` -> `PdfParser`
- `.png/.jpg/.jpeg` -> `ImageParser`
- 其他后缀直接抛错（不支持）

---

## 三、文档切分原则（`utils/text_splitter.py`）

### 1) 切分目标

切分的核心目标是：**在不破坏语义连续性的前提下，生成适合检索与向量化的 chunk**。

默认受以下参数控制（可由环境变量覆盖）：
- `CHUNK_SIZE`（默认 500）
- `CHUNK_OVERLAP`（默认 100）

### 2) 策略优先级

`text_split_to_chunks(...)` 的执行优先级：

1. 若传入 `separators`，直接按用户分隔符切分（可选保留分隔符）。
2. 否则使用默认策略：
   - 先句级切片（`SplitSentence`）
   - 再按 `chunk_size/chunk_overlap` 聚合为 chunk

### 3) 句级切片原则（`SplitSentence`）

为避免“错误断句”，句切分时有额外规则：

- 使用中文句号、感叹号、换行等做粗切分。
- 对 URL 做保护，尽量不在链接中间切断。
- 对中英文引号做配对处理，尽量把引号与其内容保留在同一片段。

### 4) chunk 组装原则（默认切分器）

采用“指针滑动 + 左右拼接”思路：

- 向左回看，控制重叠区域（`chunk_overlap`）
- 向右累加，直到接近块大小上限（`chunk_size`）
- 对生成块做清洗（去空白、去空块）

这一策略兼顾了：
- 相邻块的语义衔接（通过 overlap）
- 单块长度可控（利于向量模型与检索效率）

### 5) Markdown 专用切分（`MarkdownDocumentSplitter`）

核心原则：**先结构、后长度**。

- 先按 `#` 到 `######` 标题层级拆分。
- 尽可能在不超过 `chunk_size` 的情况下合并相邻块。
- 对超大块回退到默认切分器做二次切分。
- 生成 chunk 时会补回标题上下文，降低“离开章节语境后语义丢失”的问题。

### 6) 切分器工厂（`get_text_splitter`）

- `chunk_type=default` -> `DefaultDocumentSplitter`
- `chunk_type=markdown` -> `MarkdownDocumentSplitter`
- 未识别类型 -> 抛错

---

## 四、与离线入库流程的衔接

在 `file_processer.py` 中，解析与切分后的数据会进入三路索引：

- 文本路：`text` chunk -> dense 向量 + BM25 稀疏向量
- 图片路：图片 -> 图像向量；同时生成 OCR/Caption 并走文本向量路
- 页面路：页面图 -> 图像向量；同时生成 OCR/Caption 并走文本向量路

并支持可选子块增强（`ENABLE_SUB_CHUNK_ENHANCE` + `SUB_CHUNK_SIZE`），用于提高细粒度召回命中率。

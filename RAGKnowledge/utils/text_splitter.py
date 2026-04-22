import os
import re
import time
import traceback
from abc import ABC, abstractmethod
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

from .logger_utils import logger

# URL 相关字符集合：用于在句子切分时尽量将完整 URL 作为一个片段保留。
url_allowed_chars = "-_.~"
url_reserve_chars = "!*‘();:@&=+$,/?#[]"
url_unsafe_chars = """'"<>#%{}|\^`~"""
url_match_pattern = "(https?://[A-Za-z0-9{}]+)".format(
    "".join(["\{}".format(i) for i in url_allowed_chars + url_reserve_chars + url_unsafe_chars]))

default_separators = ["。", "!", "！", "\n\n", "\r\n"]


# 句子切分器（面向中英文混排、URL、引号场景）。
class SplitSentence(object):
    """
    先按粗粒度标点切片，再通过引号与 URL 规则进行二次合并。

    目标：
    - 尽量避免在 URL 中间切断；
    - 尽量避免将引号与其内部文本拆开；
    - 为后续 chunk 组装提供更自然的句级切片。
    """

    def __init__(self):
        self.separators = default_separators
        # 注意：这里同时包含 URL 正则与分隔符正则。
        separators_all = "{}|##|###|(\n\n)|(\r\n)|{}".format(url_match_pattern, "|".join(
            ["(\{})".format(char) for char in self.separators]))
        self.puncs_coarse_ptn = re.compile(separators_all)
        # self.separators = f"([。！!?？])|{url_match_pattern}|(\n\n)|(\r\n)"
        # self.puncs_coarse_ptn = re.compile(self.separators)
        self.puncs_coarse = {"。", "!", "！", "\n\n", "\r\n", "\r", "\n"}
        self.front_quote_list = {"“", "‘", "[", "《", "(", "（", "{", "「", "【"}
        self.back_quote_list = {"”", "’", "]", "》", ")", "）", "}", "」", "】"}

    def __call__(self, text):
        """返回句子切片列表。"""
        final_sentences = []
        tmp_list = self.puncs_coarse_ptn.split(text)
        quote_flag = False
        for sen in tmp_list:
            if not sen:  # ''或None
                continue
            if sen in self.puncs_coarse | self.front_quote_list | self.back_quote_list:
                if len(final_sentences) == 0:  # 即文本起始字符是标点
                    if sen in self.front_quote_list:  # 起始字符是前引号
                        quote_flag = True
                    final_sentences.append(sen)
                    continue
                # 以下确保当前标点前必然有文本且非空字符串
                # 前引号较为特殊，其后的一句需要与前引号合并，而不与其前一句合并
                if sen in self.front_quote_list:
                    if final_sentences[-1][-1] in self.puncs_coarse:
                        # 前引号前有标点如句号，引号等：另起一句，与此合并
                        final_sentences.append(sen)
                    else:
                        # 前引号之前无任何终止标点，与前一句合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    quote_flag = True
                else:  # 普通,非前引号，则与前一句合并
                    final_sentences[-1] = final_sentences[-1] + sen
                continue
            if len(final_sentences) == 0:  # 起始句且非标点
                final_sentences.append(sen)
                continue
            if quote_flag:  # 当前句子之前有前引号，须与前引号合并
                final_sentences[-1] = final_sentences[-1] + sen
                quote_flag = False
            else:
                if final_sentences[-1][-1] in self.back_quote_list:
                    # 此句之前是后引号，需要考察有无其他终止符，用来判断是否和前句合并
                    if len(final_sentences[-1]) <= 1:
                        # 前句仅一个字符。后引号，则合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    else:  # 前句有多个字符
                        if final_sentences[-1][-2] in self.puncs_coarse:
                            # 有句号，则需要另起一句
                            final_sentences.append(sen)
                        else:  # 前句无句号，则需要与前句合并
                            final_sentences[-1] = final_sentences[-1] + sen
                elif sen.startswith("http"):
                    final_sentences[-1] = final_sentences[-1] + sen
                else:
                    final_sentences.append(sen)
        return final_sentences


# 使用用户自定义分隔符切分文本。
def split_by_user_separators(text, separators, keep_separators):
    """
    按用户指定分隔符切分。

    Args:
        text: 原始文本
        separators: 分隔符列表
        keep_separators: 是否保留分隔符并并入前一段
    """
    if keep_separators:
        # sep_pattern = "|".join(["(\{})".format(sep) for sep in separators])
        sep_pattern = "|".join(f"({re.escape(sep)})" for sep in separators)
    else:
        sep_pattern = "|".join(re.escape(sep) for sep in separators)
        # sep_pattern = "|".join(["\{}".format(sep) for sep in separators])
    puncs_ptn = re.compile(sep_pattern)
    tmp_chunks_list = puncs_ptn.split(text)
    if keep_separators:
        chunks_list = []
        merge_flag = False
        for chunk in tmp_chunks_list:
            if not chunk:
                continue
            if merge_flag:
                chunks_list[-1] = chunks_list[-1] + chunk
                merge_flag = False
            else:
                if chunk in separators:
                    merge_flag = True
                else:
                    for sep in separators:
                        if re.match("^{}$".format(sep), chunk):
                            merge_flag = True
                            break
                chunks_list.append(chunk)
    else:
        chunks_list = [chunk for chunk in tmp_chunks_list if chunk and chunk not in separators]
    return chunks_list


# 文本主切分函数：将长文本切成可用于检索/向量化的 chunks。
def text_split_to_chunks(
        request_id="",
        chunk_size=None,
        chunk_overlap=None,
        name="",
        text="",
        separators=None,
        keep_separators=True
) -> list[str]:
    """
    将文本切分为 chunk 列表。

    策略优先级：
    1) 若传入 separators，直接使用用户分隔符；
    2) 否则先句级切片，再按 chunk_size/chunk_overlap 聚合。
    """
    if chunk_size is None:
        # 未显式传参时，从环境变量读取默认值。
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

    try:
        if not name and not text:
            logger.warning(f"{request_id} 标题和正文都为空，返回空列表")
            return []
        if not text:
            logger.warning(f"{request_id} 正文为空，返回标题chunk")
            return [name]
        if len(text) < chunk_size and (separators is None or len(separators) == 0):
            logger.info(f"{request_id}  文档长度小于 {chunk_size} ，不用切分")
            return [text]
        if separators is not None:
            logger.warning(f"{request_id} 直接按用户自定义分割符 {separators} 切分")
            chunks_list = split_by_user_separators(text, separators, keep_separators)
            for index, chunk in enumerate(chunks_list):
                if name is not None and name != "":
                    # 为每个 chunk 增加标题上下文，提升检索可解释性。
                    chunks_list[index] = name + "\n" + chunk
            logger.info(f"{request_id} 共计生成{len(chunks_list)}个chunk")
            return chunks_list
        t0 = time.time()
        # 文本分割
        logger.info(f"{request_id} 正文总字数:{len(text)}")
        # 用改进句子分割方法分割
        split_sentence = SplitSentence()
        text_slices_list = split_sentence(text)
        logger.info(f"{request_id} 按标点分割切片数:{len(text_slices_list)}")
        # 下面是“指针滑动 + 左右拼接”的 chunk 组装过程：
        # - 左侧回看：用于满足 overlap
        # - 右侧前进：直到接近 chunk_size
        chunks_list_temp = []  # 用于存储切分chunk列表
        idx = 0  # 当前切片索引指针
        idx_slice_added = False  # 指针所在切片是否已生成chunk
        chunk_left_slices_len = 0  # 当前chunk在指针左边的切片总长度
        chunk_left_slices_content = []  # 当前chunk在指针左边的切片内容
        chunk_right_slices_len = 0  # 当前chunk在指针及右边的切片总长度
        chunk_right_slices_content = []  # 当前chunk在指针及右边的切片内容
        # 从左往右遍历切片，切分chunk
        while idx < len(text_slices_list):
            logger.info(f"{request_id} 当前指针索引:{idx}")
            chunk_left_slices_len = 0  # 当前chunk在指针左边的切片总长度
            chunk_left_slices_content = []  # 当前chunk在指针左边的切片内容
            all_left_slices_content = text_slices_list[:idx][::-1]  # 指针左边的所有切片内容
            # 从右往左遍历指针左边的切片，定位chunk起始切片
            for slice in all_left_slices_content:
                chunk_left_slices_len_extend = chunk_left_slices_len + len(slice)
                # 存在重叠且累计长度超过chunk_overlap，确定chunk在指针左边的切片
                # if (chunk_left_slices_len_extend > chunk_overlap) and chunk_left_slices_len > 0:
                if chunk_left_slices_len_extend > chunk_overlap * 1.1:
                    logger.info(
                        f"{request_id} chunk包含指针左边切片数:{len(chunk_left_slices_content)} 左边切片长度:{chunk_left_slices_len}")
                    break
                # 累计长度未超过chunk_overlap，继续向左遍历
                else:
                    chunk_left_slices_len += len(slice)
                    chunk_left_slices_content = [slice] + chunk_left_slices_content
            chunk_right_slices_len = 0  # 当前chunk在指针及右边的切片总长度
            chunk_right_slices_content = []  # 当前chunk在指针及右边的切片内容
            all_right_slices_content = text_slices_list[idx:]  # 指针及右边的所有切片内容
            # 从左往右遍历指针及右边的切片，定位chunk终止切片
            for idx_tmp, slice in enumerate(all_right_slices_content):
                chunk_right_slices_len_extend = chunk_left_slices_len + chunk_right_slices_len + len(slice)
                # 累计长度超过chunk_size，确定chunk在指针右边的切片
                if chunk_right_slices_len_extend > chunk_size * 1.1 and chunk_right_slices_len > 0:
                    logger.info(
                        f"{request_id} chunk包含指针及右边切片数:{len(chunk_right_slices_content)} 右边切片长度:{chunk_right_slices_len}")
                    chunk_content = "".join(chunk_left_slices_content + chunk_right_slices_content)
                    chunks_list_temp.append(chunk_content)
                    logger.info(
                        f"{request_id} 生成第{len(chunks_list_temp)}个chunk 包含切片数:{len(chunk_left_slices_content) + len(chunk_right_slices_content)} 字数:{len(chunk_content)}")
                    idx_slice_added = True
                    if idx_tmp == 0:
                        idx += 1
                    break
                # 累计长度未超过chunk_size，继续向右遍历
                else:
                    chunk_right_slices_len += len(slice)
                    chunk_right_slices_content.append(slice)
                    idx += 1
                    idx_slice_added = False
        # 最后一个chunk长度未超过chunk_size，补充到列表
        if not idx_slice_added:
            logger.info(
                f"{request_id} chunk右边切片数:{len(chunk_right_slices_content)} 右边切片长度:{chunk_right_slices_len}")
            chunk_content = "".join(chunk_left_slices_content + chunk_right_slices_content)
            chunks_list_temp.append(chunk_content)
            logger.info(
                f"{request_id} 生成第{len(chunks_list_temp)}个chunk 包含切片数:{len(chunk_left_slices_content) + len(chunk_right_slices_content)} 字数:{len(chunk_content)}")
        # chunks 后处理：去空白、补充 name 前缀（若未带上）。
        chunks_list = []
        for chunk in chunks_list_temp:
            chunk = chunk.strip(" \n\t\r")
            if not chunk:
                continue
            if chunk.startswith(name):
                chunks_list.append(chunk)
            else:
                chunks_list.append(f"{name}-{chunk}")
        t1 = time.time()
        logger.info(f"{request_id} 共计生成{len(chunks_list)}个chunks，用时{round(t1 - t0, 3)}秒")
        return chunks_list
    except Exception as e:
        error_msg = traceback.format_exc().replace("\n", "\\n")
        logger.error(f"{request_id} 生成chunks异常：{error_msg}")
        return []


# 抽象基类，定义文本切分器接口
class BaseDocumentSplitter(ABC):
    """文本切分器抽象基类。"""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """输入原始文本，返回切分后的文本块列表。"""
        pass


# 默认切分器：调用 `text_split_to_chunks` 的通用实现。
class DefaultDocumentSplitter(BaseDocumentSplitter):
    """默认切分器：调用 `text_split_to_chunks` 的通用实现。"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self._chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 500))
        self._chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 100))

    def split(self, text: str):
        return text_split_to_chunks(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            name="",
            text=text,
            separators=None,
            keep_separators=True
        )


# Markdown 专用切分器。
class MarkdownDocumentSplitter(BaseDocumentSplitter):
    """
    Markdown 专用切分器。

    流程：
    1. 按标题层级先拆分；
    2. 在不超过 `chunk_size` 的前提下尽量合并相邻块；
    3. 对超大块回退到默认切分器二次切分。
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self._chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 500))
        self._chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 100))

    def split(self, text: str):
        headers = [('#', 'Header 1'),
                   ('##', 'Header 2'),
                   ('###', 'Header 3'),
                   ('####', 'Header 4'),
                   ('#####', 'Header 5'),
                   ('######', 'Header 6')]

        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        docs = md_splitter.split_text(text)

        def build_context(chunk_text: str, chunk_metadata: dict):
            """
            将标题上下文拼接回正文，保留层级语义。
            """
            context = ""
            for k, v in headers:
                if v in chunk_metadata:
                    context += f"{k} {chunk_metadata[v]}\n"
            context += chunk_text
            return context

        # 合并阶段：尽可能拼接相邻块，直到接近 chunk_size。
        merged_chunks = []
        current_chunk = ""
        current_metadata = {}

        for doc in docs:
            # 获取文档内容和元数据
            content = build_context(doc.page_content, doc.metadata)
            # print("Merge content: ", content)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}

            # 计算合并后的长度
            potential_length = len(current_chunk) + len(content)

            # 如果当前块为空，直接添加
            if not current_chunk:
                current_chunk = content
                current_metadata = metadata.copy()
            # 如果合并后不超过chunk_size，则合并
            elif potential_length <= self._chunk_size:
                # 添加适当的分隔符
                separator = "\n\n" if current_chunk and not current_chunk.endswith('\n') else ""
                current_chunk += separator + content

            else:
                # 当前块已满，保存并开始新块
                merged_chunks.append({
                    'content': current_chunk,
                    'metadata': current_metadata
                })

                if len(doc.page_content) > self._chunk_size:
                    text_splitter = DefaultDocumentSplitter(
                    )
                    texts = text_splitter.split(doc.page_content)
                    for chunk_text in texts:
                        merged_chunks.append({
                            'content': chunk_text,
                            'metadata': metadata.copy()
                        })
                    current_chunk = ""
                    current_metadata = {}
                else:
                    current_chunk = content
                    current_metadata = metadata.copy()

        # 添加最后一个块
        if current_chunk:
            merged_chunks.append({
                'content': current_chunk,
                'metadata': current_metadata
            })

        # 返回文本列表（包含标题上下文）。
        return [build_context(chunk['content'], chunk['metadata']) for chunk in merged_chunks]


# 根据类型工厂化创建切分器实例。
def get_text_splitter(chunk_type: str = None, chunk_size: int = None, chunk_overlap: int = None):
    """根据类型工厂化创建切分器实例。"""
    if chunk_type is None:
        chunk_type = os.getenv("CHUNK_TYPE")
    if chunk_type.lower() == "default":
        return DefaultDocumentSplitter(chunk_size, chunk_overlap)
    elif chunk_type.lower() == "markdown":
        return MarkdownDocumentSplitter(chunk_size, chunk_overlap)
    else:
        raise ValueError(f"不支持的文本切分器类型: {chunk_type}")

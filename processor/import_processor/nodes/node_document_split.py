# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.path_config import OUTPUT_DIR
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.markdown_utils import inline_image_spans


class NodeDocumentSplit(BaseNode):
    """
    节点: 文档切分 (node_document_split)
    将长文档切分成小的 Chunks (切片) 以便检索。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：加载并标准化输入数据
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：按MD标题进行初次切分
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # 步骤3：无标题场景兜底处理
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：Chunk精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)

        # 步骤5：输出文档切分统计信息
        self._step_5_print_stats(lines_count, sections)

        # 步骤6：Chunk结果本地JSON备份
        self._step_6_backup(state, sections)

        # 写入状态字典
        state["chunks"] = sections
        return state


    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """

        try:
            # 拼接备份文件路径：优先使用 state 中的输出目录，未提供时回退到项目根目录 output/
            backup_root = Path(state.get("file_dir") or OUTPUT_DIR)
            backup_path = backup_root / state.get("file_title") / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    #开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    #开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2             # 格式化缩进，便于阅读
                )
            self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            self.logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        content = section.get("content")

        # 1. 长度未超限，无需切分，直接返回
        # max_content_length = 2000
        if len(content) <= self.config.max_content_length:
            return [section]

        # 2. 切割文本
        # 章节标题
        title = section.get("title")
        # 完整标题路径占用每个子块的长度预算。
        prefix = section.get("_heading_prefix", "")

        # 计算可用正文的最大长度
        available_len = self.config.max_content_length - len(prefix)

        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分")
            return [section]


        body = content.removeprefix(prefix)

        # 对正文进行切割
        # langchain扽文本切割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len, # 切片长度（扣除标题）
            chunk_overlap=0, # 无重叠：按标题切分后语义完整，无需重叠
            separators=[
                "\n\n",
                "\n",
                "。",
                "？",
                "！",
                "；",
                ".",
                "?",
                "!",
                ";",
                " ",
                "",
            ],
            strip_whitespace=False,
        )

        sub_sections = []
        chunks = self._split_body_with_images(body, splitter, available_len)
        for idx, chunk in enumerate((chunk for chunk in chunks if chunk.strip()), start=1):
            full_text = prefix + chunk
            new_title = f"{title}-{idx}"

            sub_sections.append(
                {
                    "parent_title": title,
                    "title": new_title,
                    "content": full_text,
                    "part": idx,
                    "file_title": section.get("file_title")
                }
            )

        return sub_sections

    def _split_body_with_images(
        self, body: str, splitter: RecursiveCharacterTextSplitter, available_len: int
    ) -> Iterator[str]:
        spans = list(inline_image_spans(body))
        if not spans:
            yield from splitter.split_text(body)
            return

        def parts() -> Iterator[str]:
            cursor = 0
            for start, end in spans:
                yield from splitter.split_text(body[cursor:start])
                yield body[start:end]
                cursor = end
            yield from splitter.split_text(body[cursor:])

        current = ""
        for part in parts():
            if current and len(current) + len(part) > available_len:
                yield current
                current = ""
            if len(part) > available_len:
                self.logger.warning("图片引用长度 %d 超过正文预算 %d，独立保留", len(part), available_len)
            current += part
        if current:
            yield current


    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
       【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
       执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
       :param sections: 步骤3处理后的章节列表
       :return: 长度适中、低碎片化的最终Chunk列表
       """
        # 每个原始小节内先长切，再合并相邻短块。
        final_sections = []
        for section in sections:
            sub_sections = self._split_long_section(section)
            prefix = section.get("_heading_prefix", "")
            final_sections.extend(self._merge_short_sections(sub_sections, prefix))

        # 阶段3：父标题兜底
        # 兜底规则：无parent_title则用自身title，title也无则填空字符串
        for sec in final_sections:
            sec.pop("_heading_prefix", None)
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections


    def _merge_short_sections(self, sections: List[Dict[str,str]], prefix: str) -> List[Dict[str,str]]:
        if len(sections) < 2:
            return sections


        #合并后的结果
        merged_sections = []
        # 保存当前要合并的chunk
        current_chunk = None

        for section in sections:
            #获取第一个section赋值给current_chunk
            if current_chunk is None:
                current_chunk = section
                continue

            #段落是否过短
            # min_content_length = 500
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            next_body = section["content"].removeprefix(prefix)
            merged_length = len(current_chunk["content"]) + len(next_body)
            if is_current_short and merged_length <= self.config.max_content_length:
                current_chunk["content"] += next_body
            else:
                # 不满足合并条件：将当前块加入结果，切换为新的待合并块
                merged_sections.append(current_chunk)
                current_chunk = section
        # 循环结束后，将最后一个待合并块加入结果
        if current_chunk is not None:
            merged_sections.append(current_chunk)

        if len(merged_sections) != len(sections):
            for index, section in enumerate(merged_sections, start=1):
                section["part"] = index
                section["title"] = f"{section['parent_title']}-{index}"

        self.logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
        return merged_sections


    def _step_3_handle_no_title(self, content: str, sections: List[Dict[str, str]], title_count: int,
                                file_title: str) -> List[Dict[str, str]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count == 0 and sections:

            # 将content交给LLM，总结一个标题出来
            # 提示词：根据content和file_title总结一个段落标题，不要有空格，不要换行，控制在20个字符以内

            # 无标题处理
            return [{
                "title": "无标题",
                "content": content.strip("\n"),
                "file_title": file_title
            }]

        return sections

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # 捕获 # 的数量，后续用作标题栈的层级。
        title_pattern = r'^ {0,3}(#{1,6})[ \t]+\S'

        # 2、初始化需要的数据
        lines = content.split("\n")
        sections = []  # 章节列表
        levels = []
        current_level = 0
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        code_block_start_marker = ""

        # 3、定义内部函数组装sections列表
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            body = "\n".join(current_lines).strip("\n")
            if not current_title and not body.strip():
                return
            sections.append({
                "title": current_title,
                "content": body,
                "file_title": file_title,
            })
            levels.append(current_level)

        # 4、逐行遍历，识别标题和普通行以及代码快
        for line in lines:
            # 去空格
            striped_line = line.strip()

            # 允许围栏带语言名，用同类且足够长的标记闭合。
            fence = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
            if code_block_start_marker:
                if fence and fence[1].startswith(code_block_start_marker) and not fence[2].strip():
                    code_block_start_marker = ""
            elif fence and (fence[1][0] == "~" or "`" not in fence[2]):
                code_block_start_marker = fence[1]
            elif heading := re.match(title_pattern, line):
                #遇到标题行则先将上一个片段写入section
                _flush_section()
                current_title = striped_line
                current_level = len(heading[1])
                current_lines = []
                title_count += 1
                continue

            current_lines.append(line)
        # 为了处理最后一个章节，_flush_section中只有遇到下一个标题是才会触发保存
        _flush_section()

        # 标题进入路径，有正文的小节才生成块。
        title_stack = []
        with_body = []
        for section, level in zip(sections, self._restore_flat_levels(sections, levels)):
            if level:
                while title_stack and title_stack[-1][0] >= level:
                    title_stack.pop()
                title_stack.append((level, section["title"]))
            if section["content"].strip():
                prefix = "\n".join(title for _, title in title_stack) + "\n\n" if title_stack else ""
                section["_heading_prefix"] = prefix
                section["content"] = prefix + section["content"]
                with_body.append(section)

        return with_body, title_count, len(lines)

    def _restore_flat_levels(self, sections: List[Dict[str, str]], levels: List[int]) -> List[int]:
        """为同级且存在编号父子关系的标题补充层级。"""
        if len(set(levels) - {0}) != 1:
            return levels

        numbers = []
        for section in sections:
            match = re.match(r"^#{1,6}[ \t]+(\d+(?:\.\d+)+)[ \t]+\S", section["title"])
            numbers.append(tuple(match[1].split(".")) if match else ())

        seen = set()
        for number in numbers:
            if any(number[:depth] in seen for depth in range(1, len(number))):
                break
            if number:
                seen.add(number)
        else:
            return levels

        first = int(levels[0] == 0)
        # 编号章节前紧邻的无正文首标题作为根标题。
        root = int(
            first + 1 < len(sections)
            and not numbers[first]
            and not sections[first]["content"].strip()
            and bool(numbers[first + 1])
        )
        ancestors = []
        local_level = plain_depth = 0
        previous_plain_empty = False
        for index in range(first, len(sections)):
            section, number = sections[index], numbers[index]
            if number:
                while ancestors and (
                    len(number) <= len(ancestors[-1]) or number[:len(ancestors[-1])] != ancestors[-1]
                ):
                    ancestors.pop()
                ancestors.append(number)
                levels[index] = root + len(ancestors)
                local_level = plain_depth = 0
                previous_plain_empty = False
            elif not ancestors:
                levels[index] = 1
            elif re.match(r"^#{1,6}[ \t]+\d+\.[ \t]+\S", section["title"]):
                local_level, plain_depth = 1, 0
                previous_plain_empty = False
                levels[index] = root + len(ancestors) + local_level
            else:
                plain_depth = plain_depth + 1 if previous_plain_empty else 1
                levels[index] = root + len(ancestors) + local_level + plain_depth
                previous_plain_empty = not section["content"].strip()
        return levels


    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题（无内容则返回None,None）
        """

        # 中间节点也可以只做轻量级的防御性判断，不做完整的磁盘级别的检查 => 前置校验 + 信任状态

        # 1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title

if __name__ == "__main__":
    setup_logging()

    demo_title = "H3C LA2608室内无线网关 用户手册-6W100-整本手册"
    md_path = OUTPUT_DIR / demo_title / f"{demo_title}_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "task_id": str(uuid.uuid4()),
        "md_path": str(md_path),
        "md_content": md_content,
        "file_title": demo_title
    }

    # 执行核心处理流程
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))

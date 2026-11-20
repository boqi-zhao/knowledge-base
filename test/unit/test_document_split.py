import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from processor.import_processor.import_config import ImportConfig
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit


class DocumentSplitTests(unittest.TestCase):
    def split(self, markdown, max_length=2000):
        node = NodeDocumentSplit(ImportConfig(max_content_length=max_length))
        with TemporaryDirectory() as directory:
            (Path(directory) / "手册").mkdir()
            return node({
                "task_id": "document-split-test",
                "file_dir": directory,
                "file_title": "手册",
                "md_content": markdown,
            })["chunks"]

    def test_emits_body_with_complete_heading_stack(self):
        chunks = self.split(
            "# 手册\n\n## 网络\n\n### 有线\n\n设置网关。\n\n"
            "### 无线\n\n设置 SSID。\n\n## 安全\n\n启用防火墙。"
        )
        self.assertEqual([chunk["content"] for chunk in chunks], [
            "# 手册\n## 网络\n### 有线\n\n设置网关。",
            "# 手册\n## 网络\n### 无线\n\n设置 SSID。",
            "# 手册\n## 安全\n\n启用防火墙。",
        ])

    def test_flat_numbered_headings_keep_their_own_ancestors(self):
        chunks = self.split(
            "## 功能管理\n\n## 4.1 网络\n\n## 4.1.1 有线\n\n"
            "## 页面向导\n\n设置网关。\n\n## 4.1.2 无线\n\n设置 SSID。\n\n"
            "## 4.10 安全\n\n启用防火墙。"
        )
        self.assertEqual([chunk["content"] for chunk in chunks], [
            "## 功能管理\n## 4.1 网络\n## 4.1.1 有线\n## 页面向导\n\n设置网关。",
            "## 功能管理\n## 4.1 网络\n## 4.1.2 无线\n\n设置 SSID。",
            "## 功能管理\n## 4.10 安全\n\n启用防火墙。",
        ])

    def test_each_long_section_part_keeps_path_and_fits_budget(self):
        prefix = "# 手册\n## 网络\n### 有线\n\n"
        body = "起始" + "记录设备运行状态" * 35 + "结束"
        chunks = self.split(prefix + body, max_length=65)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["content"].startswith(prefix) for chunk in chunks))
        self.assertTrue(all(len(chunk["content"]) <= 65 for chunk in chunks))
        self.assertEqual("".join(chunk["content"][len(prefix):] for chunk in chunks), body)
        self.assertEqual([chunk["part"] for chunk in chunks], list(range(1, len(chunks) + 1)))

    def test_moves_complete_image_to_next_chunk_when_remaining_space_is_too_small(self):
        prefix = "# 手册\n## 网络\n\n"
        image = r"![DHCP [地址池\]]](https://example.invalid/dhcp(1).png)"
        opening = "检查设备当前的网络与网关配置"
        ending = "点击保存"
        limit = len(prefix) + len(image) + len(ending)
        chunks = self.split(prefix + opening + image + ending, max_length=limit)
        self.assertEqual([chunk["content"] for chunk in chunks], [
            prefix + opening,
            prefix + image + ending,
        ])

    def test_keeps_image_in_current_chunk_when_it_fits_exactly(self):
        prefix = "# 手册\n\n"
        image = "![配置截图](https://example.invalid/config.png)"
        opening = "打开配置页面。"
        ending = "点击保存。"
        limit = len(prefix) + len(opening) + len(image)
        chunks = self.split(prefix + opening + image + ending, max_length=limit)
        self.assertEqual([chunk["content"] for chunk in chunks], [
            prefix + opening + image,
            prefix + ending,
        ])

    def test_keeps_an_oversize_image_in_its_own_chunk(self):
        prefix = "# 手册\n\n"
        image = "![配置截图](https://example.invalid/config.png)"
        limit = len(prefix) + len(image) - 1
        with self.assertLogs("import.node_document_split", level="WARNING"):
            chunks = self.split(prefix + "配置前。" + image + "配置后。", max_length=limit)
        self.assertEqual([chunk["content"] for chunk in chunks], [
            prefix + "配置前。",
            prefix + image,
            prefix + "配置后。",
        ])

    def test_fenced_code_keeps_comment_with_its_section(self):
        chunks = self.split("# 配置\n\n```sh\n# 注释\n  iface=lan0\n````\n\n## 验证\n\n检查接口。")
        self.assertEqual([chunk["content"] for chunk in chunks], [
            "# 配置\n\n```sh\n# 注释\n  iface=lan0\n````",
            "# 配置\n## 验证\n\n检查接口。",
        ])

    def test_preserves_untitled_body_and_preamble(self):
        chunks = self.split("开机后检查电源。")
        self.assertEqual(chunks[0]["title"], "无标题")
        self.assertEqual(chunks[0]["content"], "开机后检查电源。")
        chunks = self.split("操作说明。\n\n# 配置\n\n调整参数。")
        self.assertEqual([chunk["content"] for chunk in chunks], ["操作说明。", "# 配置\n\n调整参数。"])

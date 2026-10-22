# processor/import_processor/nodes/node_entry.py
import json
import logging
from pathlib import Path

from config.path_config import DOC_DIR
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        # 1.从state中获取文件的绝对路径
        import_file_path = state.get("import_file_path")
        # 判断路径是否为空
        if not import_file_path:
            raise StateFieldError(
                field_name="import_file_path",
                message="文件路径不能为空",
                expected_type=str)

        #2.将import_file_path转换为Path对象
        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件{import_file_path_obj.name}不存在")

        # 3.判断文件类型
        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(message=f"不支持的文件类型{import_file_path_obj.suffix}")

        #4. 获取文件名作为标题
        state["file_title"] = import_file_path_obj.stem

        return state


if __name__ == "__main__":
    # 激活日志的全局配置
    setup_logging()

    # 初始化图状态（示例文件位于项目根目录 doc/ 下）
    demo_file = DOC_DIR / "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"
    init_state = {
        "task_id": "task_test_001",
        "import_file_path": str(demo_file)
    }

    # 创建节点对象
    node_entry = NodeEntry()
    # 执行节点的单元测试
    result = node_entry(init_state)
    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from processor.import_processor.nodes import node_item_name_recognition


class ItemNameRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.node = node_item_name_recognition.NodeItemNameRecognition()
        self.llm = Mock()
        self.client = self.enterContext(patch.object(
            node_item_name_recognition, "get_llm_client", return_value=self.llm
        ))
        self.embed = self.enterContext(patch.object(
            self.node, "_step_5_generate_vectors", return_value=([0.1], {1: 0.5})
        ))
        self.save = self.enterContext(patch.object(self.node, "_step_6_save_to_milvus"))

    def recognize(self, chunks, *responses):
        self.llm.invoke.side_effect = [
            response if isinstance(response, Exception)
            else SimpleNamespace(content=json.dumps(response, ensure_ascii=False))
            for response in responses
        ]
        return self.node({
            "task_id": "item-name-recognition-test",
            "file_title": "设备告警处理说明",
            "chunks": chunks,
        })

    def test_scans_late_chunks_and_sends_only_grounded_quotes_to_aggregation(self):
        example = "以下以 NX-300 为例说明告警配置。"
        subject = "本手册适用于 NX-200 工业网关。"
        invented = "本手册适用于 NX-999 打印机。"
        state = self.recognize(
            [
                {"title": "封面", "content": f"![{invented} [截图]](image(1).png)"},
                {"content": example},
                {"content": "请检查电源。"},
                {"title": "适用设备", "content": subject},
                {"content": example},
            ],
            {"quote": invented}, {"quote": example}, {"quote": None},
            {"quote": subject}, {"quote": example},
            {"status": "identified", "item_name": "NX-200 工业网关", "evidence_chunk": 4},
        )

        self.client.assert_called_once_with(json_mode=True)
        self.assertEqual(self.llm.invoke.call_count, 6)
        first_prompt = self.llm.invoke.call_args_list[0].args[0][-1].content
        self.assertNotIn(invented, first_prompt)
        self.assertNotIn("image(1).png", first_prompt)
        aggregation_prompt = self.llm.invoke.call_args.args[0][-1].content
        document = json.loads(aggregation_prompt.split("文档数据：\n", 1)[1])
        self.assertEqual(document["evidence"], [
            {"chunk_index": 2, "quote": example},
            {"chunk_index": 4, "quote": subject},
            {"chunk_index": 5, "quote": example},
        ])
        self.assertEqual(state["item_name_status"], "identified")
        self.assertEqual(state["item_name"], "NX-200工业网关")
        self.assertEqual([chunk["item_name"] for chunk in state["chunks"]], [state["item_name"]] * 5)
        self.embed.assert_called_once_with(state["item_name"])
        self.save.assert_called_once_with(state, state["file_title"], state["item_name"], [0.1], {1: 0.5})

    def test_keeps_file_title_when_no_evidence_is_found(self):
        state = self.recognize([{"content": "请检查电源。"}], {"quote": None})

        self.llm.invoke.assert_called_once()
        self.assertEqual(state["item_name_status"], "unconfirmed")
        self.assertEqual(state["item_name"], state["file_title"])
        self.assertEqual(state["chunks"][0]["item_name"], state["file_title"])

    def test_keeps_file_title_when_document_has_multiple_subjects(self):
        quote = "本手册适用于 NX-200 和 NX-300 工业网关。"
        state = self.recognize([{"content": quote}], {"quote": quote}, {"status": "multiple"})

        self.assertEqual(state["item_name_status"], "multiple")
        self.assertEqual(state["item_name"], state["file_title"])

    def test_rejects_confirmation_that_cites_missing_evidence(self):
        quote = "本手册适用于 NX-200 工业网关。"
        state = self.recognize(
            [{"content": quote}], {"quote": quote},
            {"status": "identified", "item_name": "NX-200", "evidence_chunk": 7},
        )

        self.assertEqual(state["item_name_status"], "unconfirmed")
        self.assertEqual(state["item_name"], state["file_title"])

    def test_model_failure_does_not_confirm_from_partial_evidence(self):
        quote = "本手册适用于 NX-200 工业网关。"
        state = self.recognize(
            [{"content": quote}, {"content": "请检查电源。"}],
            {"quote": quote}, ConnectionError("model unavailable"),
        )

        self.assertEqual(self.llm.invoke.call_count, 2)
        self.assertEqual(state["item_name_status"], "unconfirmed")
        self.assertEqual(state["item_name"], state["file_title"])

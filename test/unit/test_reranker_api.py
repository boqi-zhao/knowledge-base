"""验证百炼重排调用及分数与文档的对应关系，不发送真实请求。"""
import io
import json
import unittest
from contextlib import ExitStack
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from config.reranker_config import reranker_config
from utils.reranker_http_utils import rerank_documents


def response(results):
    return io.BytesIO(json.dumps({"output": {"results": results}}).encode())


class RerankerAPITests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in {
            "text_rerank_api_key": "rerank-test-secret",
            "text_rerank_model": "qwen3-rerank",
            "text_rerank_instruct": "针对查询检索答案",
            "text_rerank_api_base": "https://dashscope.aliyuncs.com/api/v1",
            "text_rerank_timeout": 60,
        }.items():
            stack.enter_context(patch.object(reranker_config, name, value))
        self.urlopen = stack.enter_context(patch("utils.reranker_http_utils.urlopen"))

    def test_restores_scores_to_original_document_order(self):
        self.urlopen.return_value = response([
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.02},
        ])
        docs = ["检查连接", "无关内容", "检查网络配置"]
        scores = rerank_documents("如何排查网络故障？", docs)
        self.assertEqual(scores, [0.7, 0.02, 0.95])
        self.assertEqual(docs, ["检查连接", "无关内容", "检查网络配置"])
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")
        self.assertEqual(request.get_header("Authorization"), "Bearer rerank-test-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3-rerank")
        self.assertEqual(payload["input"]["documents"], docs)
        self.assertEqual(payload["parameters"], {
            "top_n": 3, "return_documents": False, "instruct": "针对查询检索答案",
        })
        self.assertEqual(self.urlopen.call_args.kwargs["timeout"], 60)

    def test_empty_documents_do_not_call_api(self):
        self.assertEqual(rerank_documents("查询", []), [])
        self.urlopen.assert_not_called()

    def test_empty_instruction_uses_model_default(self):
        reranker_config.text_rerank_instruct = ""
        self.urlopen.return_value = response([{"index": 0, "relevance_score": 0}])
        self.assertEqual(rerank_documents("查询", ["文档"]), [0.0])
        payload = json.loads(self.urlopen.call_args.args[0].data)
        self.assertNotIn("instruct", payload["parameters"])

    def test_missing_key_fails_before_call(self):
        reranker_config.text_rerank_api_key = ""
        with self.assertRaisesRegex(ValueError, "TEXT_RERANK_API_KEY"):
            rerank_documents("查询", ["文档"])
        self.urlopen.assert_not_called()

    def test_invalid_input_is_not_filtered_or_sent(self):
        for query, docs in (
            ("", ["文档"]), (None, ["文档"]), ("查询", "文档"),
            ("查询", [None]), ("查询", ["  "]), ("查询", ["文档"] * 501),
        ):
            with self.subTest(query=query), self.assertRaises((TypeError, ValueError)):
                rerank_documents(query, docs)
        self.urlopen.assert_not_called()

    def test_incomplete_or_invalid_results_cannot_misalign_scores(self):
        for results in (
            [],
            [{"index": 0, "relevance_score": 0.9}],
            [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}],
            [{"index": 0, "relevance_score": 0.9}, {"index": 3, "relevance_score": 0.2}],
            [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": float("nan")}],
            [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 1.1}],
        ):
            with self.subTest(results=len(results)):
                self.urlopen.return_value = response(results)
                with self.assertRaises(RuntimeError):
                    rerank_documents("查询", ["文档一", "文档二"])

    def test_errors_do_not_expose_keys_in_logs(self):
        secret = reranker_config.text_rerank_api_key
        for error in (
            HTTPError("https://example.invalid", 401, secret, None, io.BytesIO(secret.encode())),
            URLError(secret), TimeoutError(secret),
        ):
            with self.subTest(error=type(error).__name__):
                self.urlopen.side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    rerank_documents("查询", ["文档"])
                self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, repr(reranker_config))

    def test_invalid_json_is_reported(self):
        self.urlopen.return_value = io.BytesIO(b"not JSON")
        with self.assertRaisesRegex(RuntimeError, "JSON"):
            rerank_documents("查询", ["文档"])


if __name__ == "__main__":
    unittest.main()

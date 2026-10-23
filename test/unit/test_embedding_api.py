"""百炼适配器的离线合同测试；不发送真实请求、不依赖本地模型或 Milvus。"""
import io
import json
import unittest
from contextlib import ExitStack
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from config.embedding_config import embedding_config
from utils.embedding_utils import EmbeddingAPIError, generate_embeddings


def embedding(index=0, value=0.25):
    return {
        "text_index": index,
        "embedding": [value] * 1024,
        "sparse_embedding": [{"index": 102897, "token": "无线", "value": 1.1865}],
    }


def response(entries):
    return io.BytesIO(json.dumps({"output": {"embeddings": entries}}).encode())


class EmbeddingAPITests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in {
            "api_key": "test-secret-key",
            "api_base": "https://dashscope.aliyuncs.com/api/v1",
            "model": "text-embedding-v4",
            "dimension": 1024,
            "timeout": 60,
        }.items():
            stack.enter_context(patch.object(embedding_config, name, value))
        self.urlopen = stack.enter_context(patch("utils.embedding_utils.urlopen"))

    def test_preserves_dual_vector_contract_and_input_order(self):
        self.urlopen.return_value = response([embedding(1, 0.5), embedding(0, 0.25)])
        result = generate_embeddings(["无线网关", "网络配置"])
        self.assertEqual(result["dense"][0], [0.25] * 1024)
        self.assertEqual(result["dense"][1], [0.5] * 1024)
        self.assertEqual(result["sparse"], [{102897: 1.1865}, {102897: 1.1865}])
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding")
        payload = json.loads(request.data)
        self.assertEqual(payload["parameters"]["output_type"], "dense&sparse")
        self.assertEqual(payload["parameters"]["dimension"], 1024)
        self.assertEqual(payload["model"], "text-embedding-v4")
        self.assertEqual(self.urlopen.call_args.kwargs["timeout"], 60)

    def test_batches_more_than_ten_texts_without_losing_alignment(self):
        batch_sizes = []

        def reply(request, timeout):
            batch = json.loads(request.data)["input"]["texts"]
            batch_sizes.append(len(batch))
            return response([embedding(i, float(text)) for i, text in reversed(list(enumerate(batch)))])

        self.urlopen.side_effect = reply
        result = generate_embeddings([str(i) for i in range(23)])
        self.assertEqual(batch_sizes, [10, 10, 3])
        self.assertEqual([v[0] for v in result["dense"]], list(range(23)))
        self.assertEqual(len(result["sparse"]), 23)

    def test_empty_input_does_not_call_api(self):
        self.assertEqual(generate_embeddings([]), {"dense": [], "sparse": []})
        self.urlopen.assert_not_called()

    def test_invalid_inputs_fail_before_api_call(self):
        for texts in ("网关", [""], ["  "], [None]):
            with self.subTest(texts=texts), self.assertRaises((TypeError, ValueError)):
                generate_embeddings(texts)
        self.urlopen.assert_not_called()

    def test_missing_key_fails_without_exposing_other_credentials(self):
        embedding_config.api_key = ""
        with self.assertRaisesRegex(ValueError, "EMBEDDING_API_KEY"):
            generate_embeddings(["网关"])
        self.urlopen.assert_not_called()

    def test_dimension_must_match_existing_business_schema(self):
        embedding_config.dimension = 1536
        with self.assertRaisesRegex(ValueError, "1024"):
            generate_embeddings(["网关"])
        self.urlopen.assert_not_called()

    def test_rejects_dense_only_response_instead_of_faking_sparse_vectors(self):
        entry = embedding()
        del entry["sparse_embedding"]
        self.urlopen.return_value = response([entry])
        with self.assertRaisesRegex(EmbeddingAPIError, "稀疏向量"):
            generate_embeddings(["网关"])

    def test_rejects_incomplete_or_malformed_vectors(self):
        bad_dimension = embedding()
        bad_dimension["embedding"] = [0.1]
        bad_number = embedding()
        bad_number["embedding"][0] = float("nan")
        bad_sparse = embedding()
        bad_sparse["sparse_embedding"][0]["index"] = "无线"
        for entries, texts in (
            ([], ["网关"]),
            ([embedding(0), embedding(0)], ["网关", "网络"]),
            ([embedding(3)], ["网关"]),
            ([bad_dimension], ["网关"]),
            ([bad_number], ["网关"]),
            ([bad_sparse], ["网关"]),
        ):
            with self.subTest(entries=len(entries), texts=len(texts)):
                self.urlopen.return_value = response(entries)
                with self.assertRaises(EmbeddingAPIError):
                    generate_embeddings(texts)

    def test_http_and_connection_errors_do_not_leak_credentials(self):
        secret = embedding_config.api_key
        for error in (
            HTTPError("https://example.invalid", 401, secret, None, io.BytesIO(secret.encode())),
            URLError(secret),
            TimeoutError(secret),
        ):
            with self.subTest(error=type(error).__name__):
                self.urlopen.side_effect = error
                with self.assertRaises(EmbeddingAPIError) as raised:
                    generate_embeddings(["网关"])
                self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, repr(embedding_config))

    def test_invalid_json_is_reported_as_api_error(self):
        self.urlopen.return_value = io.BytesIO(b"not JSON")
        with self.assertRaisesRegex(EmbeddingAPIError, "JSON"):
            generate_embeddings(["网关"])


if __name__ == "__main__":
    unittest.main()

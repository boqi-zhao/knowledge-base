import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from processor.import_processor.nodes import node_import_milvus


class NodeImportMilvusTests(unittest.TestCase):
    def test_insert_preserves_full_titles_and_limits_fixed_fields_to_utf8_bytes(self):
        long_title = "章" * 38
        long_parent = "a" * 98 + "📄后"
        exact_title = "a" + "节" * 33
        common = {
            "file_title": "设备手册",
            "item_name": "设备",
            "content": "配置步骤",
            "dense_vector": [0.1, 0.2],
            "sparse_vector": {1: 0.5},
        }
        chunks = [
            {**common, "title": long_title, "parent_title": long_parent},
            {**common, "title": exact_title, "parent_title": "配置", "part": 2},
        ]
        client = Mock()
        client.has_collection.return_value = True
        client.insert.return_value = {"ids": [101, 102]}

        with patch.object(node_import_milvus, "get_milvus_client", return_value=client):
            result = node_import_milvus.NodeImportMilvus()({
                "task_id": str(uuid4()),
                "chunks": chunks,
            })

        client.insert.assert_called_once()
        rows = client.insert.call_args.kwargs["data"]
        self.assertEqual(rows[0], {
            **common,
            "title": "章" * 33,
            "parent_title": "a" * 98,
            "full_title": long_title,
            "full_parent_title": long_parent,
            "part": 0,
        })
        self.assertEqual(rows[1], {
            **common, "title": exact_title, "parent_title": "配置", "part": 2,
        })
        self.assertEqual(result["chunks"][0], {
            **common, "title": long_title, "parent_title": long_parent, "part": 0, "chunk_id": 101,
        })
        self.assertEqual(result["chunks"][1]["chunk_id"], 102)

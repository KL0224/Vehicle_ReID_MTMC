# milvus_manager.py
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import numpy as np
from typing import List, Tuple, Optional
import threading
from log_utils import setup_logger

logger = setup_logger(__name__, "milvus.log")


class MilvusManager:
    def __init__(
            self,
            host: str = "localhost",
            port: str = "19531",
            collection_name: str = "vehicle_reid_s2",
            init_collection: bool = False,
            drop_existing: bool = False,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None

        self._lock = threading.Lock()

        self._connect()

        if init_collection:
            self._init_collection_driver(drop_existing)
        else:
            self._load_collection_executor()

        self.next_global_id = self._fetch_max_global_id() + 1
        logger.info(f"[Milvus] Next Global ID will start from: {self.next_global_id}")

    def _connect(self):
        if not connections.has_connection("default"):
            try:
                connections.connect(alias="default", host=self.host, port=self.port)
                logger.info(f"[Milvus] Connected to {self.host}:{self.port}")
            except Exception as e:
                logger.exception("[Milvus] Failed to connect")
                raise

    def _init_collection_driver(self, drop_existing: bool):
        if utility.has_collection(self.collection_name) and drop_existing:
            utility.drop_collection(self.collection_name)
            logger.warning(f"[DRIVER] Dropped collection {self.collection_name}")

        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="global_id", dtype=DataType.INT64),
                FieldSchema(name="camera_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="local_track_id", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),
                FieldSchema(name="frame_start", dtype=DataType.INT64),
                FieldSchema(name="frame_end", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields=fields, description="Vehicle ReID")
            self.collection = Collection(name=self.collection_name, schema=schema)

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"[DRIVER] Created collection {self.collection_name}")
        else:
            self.collection = Collection(self.collection_name)

        self.collection.load()

    def _load_collection_executor(self):
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _fetch_max_global_id(self) -> int:
        try:
            if self.collection.num_entities == 0:
                return -1

            res = self.collection.query(
                expr="global_id >= 0",
                output_fields=["global_id"],
                limit=1,
                order_by="global_id desc"
            )

            if res:
                return res[0]["global_id"]
            return -1
        except Exception as e:
            logger.error(f"Error fetching max ID: {e}")
            return -1

    def get_new_global_id(self) -> int:
        with self._lock:
            new_id = self.next_global_id
            self.next_global_id += 1
            return new_id

    def insert_embedding(
            self,
            camera_id: str,
            local_track_id: int,
            embedding: np.ndarray,
            frame_start: int,
            frame_end: int,
            global_id: int,
    ):
        """Insert embedding vào Milvus"""

        if frame_start > frame_end:
            frame_start, frame_end = frame_end, frame_start

        data = [
            [global_id],
            [camera_id],
            [local_track_id],
            [embedding.tolist()],
            [int(frame_start)],
            [int(frame_end)],
        ]
        self.collection.insert(data)

    def search_embedding(
            self,
            embedding: np.ndarray,
            expr: str,  # ĐỔI: Nhận filter expression thay vì camera_id
            top_k: int = 1,
            threshold: float = 0.7,
    ) -> List[Tuple[int, float, int, int]]:
        """
        Tìm kiếm embedding trong Milvus với filter expression.

        Args:
            embedding: Vector embedding cần tìm
            expr: Milvus filter expression (VD: 'camera_id == "1" && frame_id >= 100')
            top_k: Số kết quả trả về
            threshold: Ngưỡng similarity

        Returns:
            List[(global_id, similarity_score, frame_start, frame_end)]
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,  # Dùng expr từ tham số
            output_fields=["global_id", "frame_start", "frame_end"],
        )

        matches = []
        for hits in results:
            for hit in hits:
                # COSINE metric
                similarity = hit.distance
                if similarity >= threshold:
                    gid = int(hit.entity.get("global_id"))
                    fs = int(hit.entity.get("frame_start"))
                    fe = int(hit.entity.get("frame_end"))
                    matches.append((gid, float(similarity), fs, fe))

        return matches

    def flush(self):
        """Flush data to ensure all inserts are persisted"""
        try:
            if self.collection:
                self.collection.flush()
                logger.info("Milvus collection flushed successfully")
        except Exception as e:
            logger.warning(f"Flush failed (có thể không cần thiết): {e}")

    def close(self):
        connections.disconnect("default")


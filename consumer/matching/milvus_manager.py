# milvus_manager.py
# Spark-safe Milvus manager for pymilvus 2.6.5

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
from utils import setup_logger

logger = setup_logger(__name__, "milvus.log")


class MilvusManager:
    """
    MilvusManager – Spark SAFE (pymilvus 2.6.5)

    DRIVER:
        - init_collection=True
        - drop_existing=True/False
        - Chạy 1 lần duy nhất

    EXECUTOR:
        - init_collection=False
        - Chỉ connect + load + search + insert
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19531",
        collection_name: str = "vehicle_reid",
        init_collection: bool = False,
        drop_existing: bool = False,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name

        self.collection: Optional[Collection] = None
        self._insert_lock = threading.Lock()

        self._connect()

        # Driver / Executor separation
        if init_collection:
            self._init_collection_driver(drop_existing)
        else:
            self._load_collection_executor()

    # ==================================================
    # CONNECTION
    # ==================================================
    def _connect(self):
        if connections.has_connection("default"):
            return

        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            logger.info(f"[Milvus] Connected to {self.host}:{self.port}")
        except Exception as e:
            logger.exception("[Milvus] Failed to connect")
            raise RuntimeError("Cannot connect to Milvus") from e

    # ==================================================
    # DRIVER ONLY
    # ==================================================
    def _init_collection_driver(self, drop_existing: bool):
        """
        DRIVER ONLY
        """
        if utility.has_collection(self.collection_name):
            if drop_existing:
                logger.warning(
                    f"[DRIVER] Drop collection {self.collection_name}"
                )
                utility.drop_collection(self.collection_name)
            else:
                logger.info(
                    f"[DRIVER] Use existing collection {self.collection_name}"
                )
                self.collection = Collection(self.collection_name)
                self._load_collection()
                return

        logger.info(f"[DRIVER] Create collection {self.collection_name}")

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(name="global_id", dtype=DataType.INT64),
            FieldSchema(
                name="camera_id",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(name="local_track_id", dtype=DataType.INT64),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=2048,
            ),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Vehicle ReID Embeddings",
        )

        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        self._load_collection()

        logger.info(f"[DRIVER] Collection {self.collection_name} ready")

    # ==================================================
    # EXECUTOR ONLY
    # ==================================================
    def _load_collection_executor(self):
        if not utility.has_collection(self.collection_name):
            raise RuntimeError(
                f"[Executor] Collection {self.collection_name} NOT FOUND. "
                f"Run driver init first."
            )

        self.collection = Collection(self.collection_name)
        self._load_collection()

    # ==================================================
    # LOAD COLLECTION (SAFE, IDPOTENT)
    # ==================================================
    def _load_collection(self):
        """
        Gọi bao nhiêu lần cũng được:
        - Nếu đã load → NO-OP
        - Nếu chưa load → load 1 lần
        """
        try:
            self.collection.load()
            logger.info(f"[Milvus] Collection {self.collection_name} loaded")
        except Exception as e:
            logger.exception("[Milvus] Load collection failed")
            raise

    # ==================================================
    # INSERT
    # ==================================================
    def insert_embedding(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp: int,
        global_id: int,
    ) -> int:
        """
        global_id phải được sinh bên ngoài (Spark-safe)
        """
        data = [
            [global_id],
            [camera_id],
            [local_track_id],
            [embedding.tolist()],
            [timestamp],
        ]

        with self._insert_lock:
            self.collection.insert(data)

        return global_id

    # ==================================================
    # SEARCH
    # ==================================================
    def search_embedding(
        self,
        embedding: np.ndarray,
        camera_id: str,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Tuple[int, float]]:

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        expr = f'camera_id != "{camera_id}"'

        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["global_id"],
        )

        matches = []
        for hits in results:
            for hit in hits:
                similarity = 1.0 - hit.distance
                if similarity >= threshold:
                    matches.append(
                        (hit.entity.get("global_id"), similarity)
                    )

        return matches

    # ==================================================
    # QUERY
    # ==================================================
    def get_global_id_mapping(
        self,
        camera_id: str,
        local_track_id: int,
    ) -> Optional[int]:

        expr = (
            f'camera_id == "{camera_id}" && '
            f'local_track_id == {local_track_id}'
        )

        results = self.collection.query(
            expr=expr,
            output_fields=["global_id"],
            limit=1,
        )

        return results[0]["global_id"] if results else None

    # ==================================================
    # CLEANUP
    # ==================================================
    def close(self):
        try:
            connections.disconnect("default")
        except Exception:
            pass

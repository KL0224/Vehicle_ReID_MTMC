# reid_matcher.py
import time
import uuid
import numpy as np
from typing import Dict, Tuple, Optional
from .milvus_manager import MilvusManager
from utils import setup_logger
import random

logger = setup_logger(__name__, "reid.log")


class ReIDMatcher:
    """
    Spark-safe ReID Matcher

    - Không sinh global_id trong Milvus
    - Chống insert trùng
    - Cache có TTL
    - Không giữ state vô hạn
    """

    def __init__(
        self,
        milvus: MilvusManager,
        similarity_threshold: float = 0.7,
        cache_ttl: int = 300,  # seconds
    ):
        self.milvus = milvus
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl

        # cache_key -> (global_id, last_seen_ts)
        self._local_cache: Dict[str, Tuple[int, int]] = {}

    # ----------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------
    def match_or_create(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
    ) -> int:
        """
        Match embedding -> global_id
        Nếu không match → tạo global_id mới

        global_id do matcher sinh (Spark-safe)
        """

        now = int(time.time())
        cache_key = f"{camera_id}:{local_track_id}"

        # 1️Cleanup cache cũ
        self._cleanup_cache(now)

        # Cache hit
        cached = self._local_cache.get(cache_key)
        if cached:
            global_id, _ = cached
            return global_id

        # Search Milvus
        matches = self.milvus.search_embedding(
            embedding=embedding,
            camera_id=camera_id,
            top_k=5,
            threshold=self.similarity_threshold,
        )

        if matches:
            global_id, score = matches[0]
            logger.info(
                f"[ReID] MATCH cam={camera_id} "
                f"track={local_track_id} -> global={global_id} "
                f"score={score:.3f}"
            )
        else:
            # Tạo global_id mới (Spark-safe)
            global_id = self._generate_global_id()
            logger.info(
                f"[ReID] NEW OBJECT cam={camera_id} "
                f"track={local_track_id} -> global={global_id}"
            )

        # Insert embedding (CHỈ 1 lần)
        self.milvus.insert_embedding(
            camera_id=camera_id,
            local_track_id=local_track_id,
            embedding=embedding,
            timestamp=now,
            global_id=global_id,
        )

        # Update cache
        self._local_cache[cache_key] = (global_id, now)

        return global_id

    # ----------------------------------------------------
    # INTERNAL
    # ----------------------------------------------------
    def _cleanup_cache(self, now: int):
        """Xóa cache quá TTL"""
        expired_keys = [
            k
            for k, (_, ts) in self._local_cache.items()
            if now - ts > self.cache_ttl
        ]
        for k in expired_keys:
            del self._local_cache[k]

    def _generate_global_id(self) -> int:
        """
        Sinh global_id Spark-safe
        """
        return random.randint(1, 2 ** 63 - 1)

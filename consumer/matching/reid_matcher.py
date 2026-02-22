import numpy as np
from typing import Dict, Tuple, List, Optional
from .milvus_manager import MilvusManager
from log_utils import setup_logger


logger = setup_logger(__name__, "reid.log")


class ReIDMatcher:
    # Simplified 1-way flow time windows (seconds)
    # Cam1 → Cam2
    # Cam1 → Cam3
    # Cam2 → Cam3 (overlap)
    DT_MIN_SEC = {
        ("1", "2"): 0.5,
        ("1", "3"): 0.0,
        ("2", "3"): -3.0,
    }

    DT_MAX_SEC = {
        ("1", "2"): 4.0,
        ("1", "3"): 4.0,
        ("2", "3"): 3.0,
    }

    def __init__(
        self,
        milvus: MilvusManager,
        fps: int = 10,
        similarity_threshold: float = 0.6,
        top_k: int = 5,
        alpha_time: float = 0.15,
        active_lock_frames: int = 5,
    ):
        self.milvus = milvus
        self.fps = fps
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.alpha_time = alpha_time

        # Block using the same global\_id in 2 cameras if their last frames are too close
        self.active_lock_frames = active_lock_frames

        # Track active global\_ids: global\_id -> (camera\_id, last\_frame\_id)
        self.active_global: Dict[int, Tuple[str, int]] = {}

        # seconds -> frames
        self.dt_min_frames = {(k[0], k[1]): int(v * fps) for k, v in self.DT_MIN_SEC.items()}
        self.dt_max_frames = {(k[0], k[1]): int(v * fps) for k, v in self.DT_MAX_SEC.items()}

        logger.info(f"ReID Matcher initialized with FPS={fps}, top_k={top_k}, alpha_time={alpha_time}")

    def _build_filter_expr(self, from_camera: str, to_camera: str, current_start: int, current_end: int) -> str:
        """
                Candidate track interval \([cand_start,cand_end]\) must intersect the expected time window
                relative to current interval \([current_start,current_end]\).

                We use current\_start to set the window for when the object could have been seen in `from_camera`
                before appearing in `to_camera`.
                """
        key = (from_camera, to_camera)
        if key not in self.dt_min_frames:
            return ""

        if current_start > current_end:
            current_start, current_end = current_end, current_start

        t_min = self.dt_min_frames[key]
        t_max = self.dt_max_frames[key]

        w_start = current_start - t_max
        w_end = current_start - t_min
        if w_start > w_end:
            w_start, w_end = w_end, w_start

        # interval overlap: cand_start <= w_end AND cand_end >= w_start
        return (
            f'(camera_id == "{from_camera}" && '
            f'frame_start <= {int(w_end)} && frame_end >= {int(w_start)})'
        )

    def _expected_dt_frames(self, from_camera: str, to_camera: str) -> int:
        key = (from_camera, to_camera)
        mn = self.dt_min_frames.get(key, 0)
        mx = self.dt_max_frames.get(key, 0)
        return int((mn + mx) / 2)

    def _is_parallel_active_block(self, gid: int, current_camera: str, current_frame_id: int) -> bool:
        """
        Trả về True nếu gid đang hoạt động ở camera khác quá gần về thời gian.
        Đặc biệt: Cho phép hoạt động song song nếu là cặp Cam 2 và Cam 3 (Overlap).
        """
        if gid not in self.active_global:
            return False

        active_cam, active_frame = self.active_global[gid]

        # Nếu cùng một camera thì không block (logic cũ)
        if active_cam == current_camera:
            return False

        # --- ĐIỀU CHỈNH MỚI CHO VÙNG OVERLAP ---
        # Danh sách các cặp camera được phép nhìn thấy nhau cùng lúc
        overlap_pairs = {("2", "3"), ("3", "2")}
        if (active_cam, current_camera) in overlap_pairs:
            # Trong vùng overlap, không chặn cho dù thời gian có trùng nhau hoàn toàn
            return False
        # ---------------------------------------

        # Đối với các cặp khác (ví dụ 1-2, 1-3), vẫn giữ nguyên quy tắc khóa thời gian
        return abs(current_frame_id - active_frame) <= self.active_lock_frames

    def _update_active(self, gid: int, camera_id: str, frame_id: int) -> None:
        self.active_global[gid] = (camera_id, frame_id)

    @staticmethod
    def _closest_dt_to_interval(current_start: int, current_end: int, cand_start: int, cand_end: int) -> int:
        """
        Compute a robust dt between two intervals, using the closest valid boundary relationship.
        Intended for transitions where `cand` happens before `current` typically.
        """
        if current_start > current_end:
            current_start, current_end = current_end, current_start
        if cand_start > cand_end:
            cand_start, cand_end = cand_end, cand_start

        # dt options (positive is typical): current_start - cand_end, current_end - cand_start
        d1 = current_start - cand_end
        d2 = current_end - cand_start
        # pick the one with smaller absolute magnitude (more stable vs interval length)
        return d1 if abs(d1) <= abs(d2) else d2

    def _pick_best_with_time_penalty(
            self,
            matches: List[Tuple[int, float, int, int]],
            current_camera: str,
            current_start: int,
            current_end: int,
            from_camera: str,
            to_camera: str,
            alpha: Optional[float] = None,
    ) -> Optional[Tuple[int, float, float]]:
        """
        matches: List of (global\_id, similarity, cand\_frame\_start, cand\_frame\_end)
        Returns (best\_gid, best\_final\_score, best\_similarity)
        """
        if not matches:
            return None

        expected_dt = self._expected_dt_frames(from_camera, to_camera)
        alpha = self.alpha_time if alpha is None else alpha

        best_gid: Optional[int] = None
        best_final = -1e18
        best_sim = -1e18

        # choose a single frame to enforce active lock, use current\_start
        current_anchor = int(min(current_start, current_end))

        for gid, sim, cand_start, cand_end in matches:
            if self._is_parallel_active_block(gid, current_camera=current_camera, current_frame_id=current_anchor):
                continue

            dt = self._closest_dt_to_interval(current_start, current_end, cand_start, cand_end)
            time_penalty = abs(dt - expected_dt) / max(1, expected_dt)
            final_score = sim - alpha * time_penalty

            if final_score > best_final:
                best_final = final_score
                best_gid = gid
                best_sim = sim

        if best_gid is None:
            return None
        return best_gid, best_final, best_sim

    def _search_candidates(
            self,
            embedding: np.ndarray,
            filter_expr: str,
            threshold: float,
    ) -> List[Tuple[int, float, int, int]]:
        if not filter_expr:
            return []
        return self.milvus.search_embedding(
            embedding=embedding,
            expr=filter_expr,
            top_k=self.top_k,
            threshold=threshold,
        )

    def match_and_update(
            self,
            camera_id: str,
            local_track_id: int,
            embedding: np.ndarray,
            frame_start: int,
            frame_end: int,
    ) -> int:
        if frame_start > frame_end:
            frame_start, frame_end = frame_end, frame_start

        global_id = -1
        current_anchor = int(frame_start)

        if camera_id == "1":
            global_id = self.milvus.get_new_global_id()
            logger.info(f"[CAM1 NEW] Local:{local_track_id} → Global:{global_id}")

        elif camera_id == "2":
            filter_expr = self._build_filter_expr("1", "2", frame_start, frame_end)
            threshold_cross_view = self.similarity_threshold * 0.4
            matches = self._search_candidates(embedding=embedding, filter_expr=filter_expr,
                                              threshold=threshold_cross_view)

            picked = self._pick_best_with_time_penalty(
                matches=matches,
                current_camera=camera_id,
                current_start=frame_start,
                current_end=frame_end,
                from_camera="1",
                to_camera="2",
            )

            if picked:
                global_id, final_score, sim = picked
                logger.info(
                    f"[CAM2 MATCHED] Local:{local_track_id} → Global:{global_id} "
                    f"(from Cam1, sim={sim:.3f}, final={final_score:.3f})"
                )
            else:
                global_id = self.milvus.get_new_global_id()
                logger.info(f"[CAM2 NEW] Local:{local_track_id} → Global:{global_id} (no usable match in Cam1)")

        elif camera_id == "3":
            filter_expr_cam1 = self._build_filter_expr("1", "3", frame_start, frame_end)
            matches_cam1 = self._search_candidates(embedding=embedding, filter_expr=filter_expr_cam1,
                                                   threshold=self.similarity_threshold)
            picked_cam1 = self._pick_best_with_time_penalty(
                matches=matches_cam1,
                current_camera=camera_id,
                current_start=frame_start,
                current_end=frame_end,
                from_camera="1",
                to_camera="3",
            )

            picked_cam2 = None
            if not picked_cam1:
                filter_expr_cam2 = self._build_filter_expr("2", "3", frame_start, frame_end)
                threshold_overlap = self.similarity_threshold * 0.4
                matches_cam2 = self._search_candidates(
                    embedding=embedding,
                    filter_expr=filter_expr_cam2,
                    threshold=threshold_overlap,
                )
                picked_cam2 = self._pick_best_with_time_penalty(
                    matches=matches_cam2,
                    current_camera=camera_id,
                    current_start=frame_start,
                    current_end=frame_end,
                    from_camera="2",
                    to_camera="3",
                )

            picked = picked_cam1 or picked_cam2
            if picked:
                global_id, final_score, sim = picked
                source = "Cam1" if picked_cam1 else "Cam2"
                logger.info(
                    f"[CAM3 MATCHED] Local:{local_track_id} → Global:{global_id} "
                    f"(from {source}, sim={sim:.3f}, final={final_score:.3f})"
                )
            else:
                global_id = self.milvus.get_new_global_id()
                logger.info(f"[CAM3 NEW] Local:{local_track_id} → Global:{global_id} (no usable match in Cam1/Cam2)")

        else:
            logger.error(f"Unknown camera_id: {camera_id}")
            global_id = self.milvus.get_new_global_id()

        self.milvus.insert_embedding(
            camera_id=camera_id,
            local_track_id=local_track_id,
            embedding=embedding,
            frame_start=frame_start,
            frame_end=frame_end,
            global_id=global_id,
        )

        # keep the active lock anchored at track start
        self._update_active(global_id, camera_id, current_anchor)
        return global_id
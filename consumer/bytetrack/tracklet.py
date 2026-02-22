from pathlib import Path
import numpy as np
from collections import defaultdict
import os
import json
import cv2
from log_utils import setup_logger

logger = setup_logger(__name__, "tracklet.log")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder để xử lý NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class Tracklet:
    """Tạo tracklet để lưu trư các frame object cho reid"""
    def __init__(self, max_frame_per_track = 30, min_conf = 0.4, output_dir="tracking_output"):
        self.max_frame_per_track = max_frame_per_track
        self.min_conf = min_conf
        self.crops = defaultdict(list) # Lưu top K crop cho reid
        self.metadata = defaultdict(lambda: {
            "track_id": None,
            "frame_id": [],
            "bbox": [],
            "conf": [],
            "global_id": None,
            "is_active": True,
            "state_history": [], # Lịch sử trạng thái của đối tượng,
        })

        self.global_ids = {}
        self.embeddings = {}  # Lưu {track_id: embedding_vector}
        self.prev_tracks_state = {}  # Lưu trạng thái ở frame trước

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_blur_score(self, image):
        """Tính độ nét của ảnh bằng Laplacian Variance"""
        if image is None or image.size == 0: return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def should_reid(self, track_id: int, min_frames: int = 5, min_crops: int = 1, min_rep_conf: float = 0.3) -> bool:
        """
        Quality gate: only do ReID for tracks that are long enough and have usable crops.
        This reduces fragmentation -> new global_id explosions.
        """
        meta = self.metadata.get(track_id)
        if not meta:
            return False

        num_frames = len(meta.get("frame_id", []))
        if num_frames < min_frames:
            return False

        crops = self.crops.get(track_id, [])
        if len(crops) < min_crops:
            return False

        confs = meta.get("conf", [])
        if not confs:
            return False

        rep_conf = float(np.max(confs))
        if rep_conf < float(min_rep_conf):
            return False

        return True

    def update(self, track_id, frame, frame_id, bbox, conf):
        x1, y1, x2, y2 = map(float, bbox)
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))

        w_obj = x2 - x1
        h_obj = y2 - y1
        if w_obj <= 0 or h_obj <= 0 or w_obj * h_obj < 1200:
            return

        # Always store metadata (for MOT/JSON and for ReID gating)
        meta = self.metadata[track_id]
        if meta["track_id"] is None:
            meta["track_id"] = int(track_id)
        meta["frame_id"].append(int(frame_id))
        meta["bbox"].append([float(x1), float(y1), float(x2), float(y2)])
        meta["conf"].append(float(conf))

        # Crop only for ReID candidates
        crop_obj = frame[y1:y2, x1:x2]
        if crop_obj is None or crop_obj.size == 0:
            return

        area = w_obj * h_obj
        self.crops[track_id].append(
            {
                "crop": crop_obj,
                "frame_id": int(frame_id),
                "conf": float(conf),
                "area": float(area),
            }
        )

        self.crops[track_id].sort(key=lambda x: (x["area"], x["conf"]), reverse=True)
        if len(self.crops[track_id]) > self.max_frame_per_track:
            self.crops[track_id] = self.crops[track_id][: self.max_frame_per_track]

    def get_top_k_crops(self, track_id, k=5):
        """Lấy ra K crops tốt nhất cho REID"""
        if track_id not in self.crops:
            return []
        sorted_crops = sorted(self.crops[track_id], key=lambda x: x["conf"], reverse=True)
        return [c["crop"] for c in sorted_crops[:k]]

    def get_reid_payload(self, track_id: int, top_k: int = 5):
        """
        Trả về dữ liệu cần cho ReID worker
        """
        return {
            "track_id": track_id,
            "crops": self.get_top_k_crops(track_id, k=top_k),
            "metadata": self.metadata.get(track_id)
        }

    def get_track_frame_range(self, track_id: int) -> tuple[int, int]:
        """
        Return `(frame_start, frame_end)` for the track.
        Returns `(-1, -1)` if unavailable.
        """
        meta = self.metadata.get(track_id)
        if not meta:
            return -1, -1

        frame_ids = meta.get("frame_id", [])
        if not frame_ids:
            return -1, -1

        frame_start = int(frame_ids[0])
        frame_end = int(frame_ids[-1])
        if frame_start > frame_end:
            frame_start, frame_end = frame_end, frame_start
        return frame_start, frame_end

    def make_removed_tracks(self, current_tracks_state):
        """
        Phát hiện tracks đã chuyển sang trạng thái Removed
        Args:
            + current_tracks_state: Dict {track_id: state}
                                 state = 0 (Tracked), 1 (Lost), 2 (Removed)

        Returns:
            List track_id đã Removed (cần extract embedding)
        """
        removed_tracks = []

        # So sánh với frame trước để tìm tracks mới chuyển sang Removed
        for track_id, current_state in current_tracks_state.items():
            prev_state = self.prev_tracks_state.get(track_id, None)

            # Lưu lịch sử trạng thái
            if track_id in self.metadata:
                state_name = {0: "Tracked", 1: "Lost", 2: "Removed"}.get(current_state, "Unknown")
                self.metadata[track_id]["state_history"].append(state_name)

            # Chỉ xử lý khi track chuyển từ Lost → Removed hoặc Tracked → Removed
            if current_state == 2:  # Removed
                if prev_state != 2:  # Chưa phải Removed ở frame trước
                    if track_id in self.metadata and self.metadata[track_id]['is_active']:
                        self.metadata[track_id]['is_active'] = False
                        removed_tracks.append(track_id)

                        state_msg = f"{prev_state if prev_state else 'Unknown'} → Removed"
                        logger.info(f"Track {track_id}: {state_msg} (có {len(self.crops.get(track_id, []))} crops)")

        # Cập nhật trạng thái cho frame tiếp theo
        self.prev_tracks_state = current_tracks_state.copy()

        return removed_tracks

    def save_embeddings(self, track_id, embedding):
        """Lưu embedding cho track_id"""
        if embedding is None:
            logger.warning(f"Embedding của {track_id} NULL")
            return

        self.embeddings[track_id] = embedding
        logger.info(f"Track {track_id}: đã lưu embedding shape {embedding.shape}")

    def save_global_id(self, track_id: int, global_id: int):
        """Lưu global_id cho track"""
        if track_id in self.metadata:
            self.metadata[track_id]["global_id"] = global_id

        self.global_ids[track_id] = global_id
        logger.info(f"Track {track_id} --> global id {global_id}")

    def remove_track_data(self, track_id):
        """Xoá crops và metadata để giải phóng bộ nhớ"""

        # Xóa crop
        if track_id in self.crops:
            num_crops = len(self.crops[track_id])
            total_pixels = sum(c['crop'].size for c in self.crops[track_id])
            memory_mb = (total_pixels * 3) / (1024 * 1024)
            del self.crops[track_id]
            logger.info(f"Track {track_id}: clear {num_crops} crops (~{memory_mb:.2f} MB)")

        # Xóa embeddings
        if track_id in self.embeddings:
            del self.embeddings[track_id]

        # Xóa trạng thái cũ
        if track_id in self.prev_tracks_state:
            del self.prev_tracks_state[track_id]

    def export_to_json(self, cam_id, file_name=None):
        """Lưu trữ track ra file json"""

        if not file_name:
            file_path = os.path.join(self.output_dir, f"camera_{cam_id}.json")
        else:
            file_path = os.path.join(self.output_dir, f"{file_name}.json")

        export_data = {
            'cam_id': cam_id,
            'tracks': []
        }

        for track_id, metadata in self.metadata.items():
            if not metadata["frame_id"]:
                continue

            track_data = {
                'track_id': int(metadata['track_id']),
                'total_frames': len(metadata['frame_id']),
                'detections': []
            }

            for frame_id, bbox, conf in zip(
                    metadata['frame_id'],
                    metadata['bbox'],
                    metadata['conf']
            ):
                track_data['detections'].append({
                    'frame_id': int(frame_id),
                    'bbox': [float(x) for x in bbox],
                    'confidence': float(conf)
                })

            export_data['tracks'].append(track_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"Đã lưu file {file_path} thành công")

    def export_to_crops(self, cam_id, track_id, output_crops=None, k: int = 5):
        """Lưu tối đa *k* crops tốt nhất của *track_id* ra ảnh, dùng cho debug/REID."""
        if not output_crops:
            output_crops = Path(self.output_dir) / f"camera{cam_id}" / "crops"
        else:
            output_crops = Path(output_crops)
        output_crops.mkdir(parents=True, exist_ok=True)

        if track_id not in self.crops:
            logger.warning(f"Track #{track_id} không có crops!")
            return []

        if not self.crops[track_id]:
            logger.warning(f"Track #{track_id} có crops rỗng!")
            return []

        # Lấy k crops có conf cao nhất
        crops_sorted = sorted(self.crops[track_id], key=lambda x: x["conf"], reverse=True)
        crops_sorted = crops_sorted[:k]

        save_files = []
        for i, crop_data in enumerate(crops_sorted):
            crop = crop_data['crop']
            conf = crop_data['conf']
            frame_id = crop_data['frame_id']

            file_name = output_crops / f"track_{track_id}.jpg"

            if crop is None or crop.size == 0:
                logger.warning(f"Crop #{i} cho track {track_id} bị rỗng!")
                continue

            success = cv2.imwrite(str(file_name), crop)
            if success:
                save_files.append(file_name)
            else:
                logger.warning(f"Failed to save: {file_name}")

        logger.info(f"Đã lưu {len(save_files)} crop ảnh cho track {track_id}!")
        return save_files

    def export_embeddings(self, cam_id, file_name=None):
        """Export embeddings ra file .npz"""

        if not self.embeddings:
            logger.warning(f"Không có embeddings nào để export")
            return

        if not file_name:
            file_path = os.path.join(self.output_dir, f"camera_{cam_id}_embeddings.npz")
        else:
            file_path = os.path.join(self.output_dir, f"{file_name}.npz")

        np.savez(
            file_path,
            track_ids = np.array(list(self.embeddings.keys())),
            embeddings = np.array(list(self.embeddings.values())),
        )

        logger.info(f"Đã lưu {len(self.embeddings)} embeddinsg tại {file_path}")

    def export_mot(self, cam_id, file_name=None, track_ids=None):
        """
        Export kết quả tracking sang định dạng MOT:
        frame, id, x, y, w, h, conf, -1, -1, -1
        Trong đó bbox đang lưu dạng [x1, y1, x2, y2].
        """
        if not file_name:
            file_path = os.path.join(self.output_dir, f"camera_{cam_id}.txt")
        else:
            file_path = os.path.join(self.output_dir, f"{file_name}.txt")

        # Đọc các dòng đã có trong file (nếu tồn tại)
        existing_lines = set()
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_lines = set(line.strip() for line in f if line.strip())

        new_lines = []

        # Chọn ids có trước hoặc lấy toàn bộ trong metadata
        target_ids = track_ids if track_ids is not None else list(self.metadata.keys())

        for track_id in target_ids:
            if track_id not in self.metadata:
                continue

            meta = self.metadata[track_id]
            if not meta["frame_id"]:
                continue

            frame_ids = meta["frame_id"]
            bboxes = meta["bbox"]
            confs = meta["conf"]
            global_id = meta.get("global_id", None)

            if global_id is None:
                continue

            tid = int(global_id)

            for frame_id, bbox, conf in zip(frame_ids, bboxes, confs):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

                x, y, w, h = float(x1), float(y1), float(w), float(h)
                conf = float(conf)
                fid = int(frame_id)

                # MOT: frame, id, x, y, w, h, conf, -1, -1, -1
                line = f"{fid},{tid},{int(x)},{int(y)},{int(w)},{int(h)},{conf:.4f},-1,-1,-1"

                # Chỉ thêm nếu chưa tồn tại
                if line not in existing_lines:
                    new_lines.append(line)
                    existing_lines.add(line)

        if not new_lines and not existing_lines:
            return file_path

        # Gộp tất cả, sort theo frame_id và track_id, ghi đè file
        all_lines = list(existing_lines)
        all_lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))

        with open(file_path, 'w', encoding='utf-8') as f:
            for line in all_lines:
                f.write(line + '\n')

        logger.info(f"Đã export {len(all_lines)} detections (unique) sang file MOT: {file_path}")
        return file_path
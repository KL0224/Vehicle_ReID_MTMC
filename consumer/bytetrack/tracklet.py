from pathlib import Path
import numpy as np
from collections import defaultdict
import os
import json
import cv2
from utils import setup_logger

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
    def __init__(self, max_frame_per_track = 30, min_conf = 0.5, output_dir="tracking_output"):
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

    def __getstate__(self):
        """Serialize cho Spark checkpoint"""
        return {
            'crops': {k: v for k, v in self.crops.items()},
            'metadata': {k: dict(v) for k, v in self.metadata.items()},
            'prev_tracks_state': dict(self.prev_tracks_state),
            'embeddings': dict(self.embeddings),
            'global_ids': dict(self.global_ids),
            'output_dir': self.output_dir,
            'max_frame_per_track': self.max_frame_per_track,
            'min_conf': self.min_conf,
        }

    def __setstate__(self, state):
        """Restore từ checkpoint"""
        # Khôi phục lại defaultdict behavior
        self.crops = defaultdict(list, state['crops'])
        self.metadata = defaultdict(
            lambda: {
                "track_id": None,
                "frame_id": [],
                "bbox": [],
                "conf": [],
                "global_id": None,
                "is_active": True,
                "state_history": [],
            },
            state['metadata']
        )

        self.prev_tracks_state = state['prev_tracks_state']
        self.embeddings = state.get('embeddings', {})
        self.global_ids = state.get('global_ids', {})
        self.output_dir = state['output_dir']
        self.max_frame_per_track = state['max_frame_per_track']
        self.min_conf = state['min_conf']

    def update(self, track_id, frame, frame_id, bbox, conf):
        """
        Nhận track_id từ bytetrack, lưu metadata và crops
        :param track_id: id của đối tượng
        :param frame: frame chứa đối tượng đó để crop
        :param frame_id: frame_id chứa đối tượng
        :param bbox: danh sách bbox của đối tượng
        :param conf: danh sách conf của đối tượng
        """

        if conf < self.min_conf:
            return

        x1, y1, x2, y2 = map(float, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Bbox không hợp lệ cho track {track_id}: ({x1},{y1},{x2},{y2})")
            return

        crop_obj = frame[y1:y2, x1:x2]

        # Kiểm tra crop có hợp lệ không
        if crop_obj is None or crop_obj.size == 0:
            logger.warning(f"Crop rỗng cho track {track_id} tại frame {frame_id}")
            return

        # crop_h, crop_w = crop_obj.shape[:2]
        # if crop_h < 20 or crop_w < 20:
        #     logger.warning(f"Crop quá nhỏ: {crop_h}, {crop_w}")
        #     return
        # aspect_ratio = crop_w / crop_h
        # if aspect_ratio > 5  or aspect_ratio < 5:
        #     logger.warning(f"Aspect ratio bất thường: {aspect_ratio:.2f}")
        #     return

        # Lưu crop cho reid
        self.crops[track_id].append(
            {
                "crop": crop_obj,
                "frame_id": int(frame_id),
                "conf": float(conf) # Dùng cho sorted
            }
        )

        # Chỉ lấy crop của max_frame_per_track
        if len(self.crops[track_id]) > self.max_frame_per_track:
            self.crops[track_id].sort(key=lambda crop: crop["conf"], reverse=True)
            self.crops[track_id] = self.crops[track_id][:self.max_frame_per_track]

        # Lưu metadata
        meta = self.metadata[track_id]
        if meta["track_id"] is None:
            meta["track_id"] = int(track_id)

        meta["frame_id"].append(int(frame_id))
        meta["bbox"].append([float(x1), float(y1), float(x2), float(y2)])
        meta["conf"].append(float(conf))

    def get_top_k_crops(self, track_id, k=5):
        """Lấy ra K crops tốt nhất cho REID"""
        if track_id not in self.crops:
            return []
        sorted_crops = sorted(self.crops[track_id], key=lambda x: x["conf"], reverse=True)
        return [c["crop"] for c in sorted_crops[:k]]

    def get_metadata(self, track_id):
        return self.metadata.get(track_id)

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

    def clear_crops(self, track_id):
        """Xoá crops và metadata để tránh track_id tái sử dụng bị dính lịch sử cũ"""
        if track_id in self.crops:
            num_crops = len(self.crops[track_id])
            total_pixels = sum(c['crop'].size for c in self.crops[track_id])
            memory_mb = (total_pixels * 3) / (1024 * 1024)
            del self.crops[track_id]
            logger.info(f"Track {track_id}: clear {num_crops} crops (~{memory_mb:.2f} MB)")

        # Xoá luôn metadata và embedding để lần sau coi như track mới
        if track_id in self.metadata:
            del self.metadata[track_id]
        if track_id in self.embeddings:
            del self.embeddings[track_id]
        if track_id in self.global_ids:
            del self.global_ids[track_id]

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

            file_name = output_crops / f"track{track_id}_frame{frame_id}_conf{conf:.3f}.jpg"

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

    def export_mot(self, cam_id, file_name=None):
        """
        Export kết quả tracking sang định dạng MOT:
        frame, id, x, y, w, h, conf, -1, -1, -1
        Trong đó bbox đang lưu dạng [x1, y1, x2, y2].
        """
        if not file_name:
            file_path = os.path.join(self.output_dir, f"camera_{cam_id}.txt")
        else:
            file_path = os.path.join(self.output_dir, f"{file_name}.txt")

        lines = []

        for track_id, meta in self.metadata.items():
            # Bỏ qua nếu track không có detection
            if not meta["frame_id"]:
                continue

            frame_ids = meta["frame_id"]
            bboxes = meta["bbox"]
            confs = meta["conf"]

            for frame_id, bbox, conf in zip(frame_ids, bboxes, confs):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

                # Đảm bảo số thực
                x, y, w, h = float(x1), float(y1), float(w), float(h)
                conf = float(conf)
                tid = int(track_id)
                fid = int(frame_id)

                # MOT: frame, id, x, y, w, h, conf, -1, -1, -1
                line = f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                lines.append(line)

        # MOT thường yêu cầu sort theo frame, rồi theo id
        lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.info(f"Đã export {len(lines)} detections sang file MOT: {file_path}")
        return file_path











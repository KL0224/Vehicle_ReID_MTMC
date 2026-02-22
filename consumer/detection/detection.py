from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict

# COCO class IDs cho xe (vehicle)
VEHICLE_CLASS_IDS = {
    2: "car",
    5: "bus",

    7: "truck",
}


def load_model_detector(model_path):
    if not os.path.exists(model_path):
        print(f"Không tồn tại model tại đường dẫn {model_path}")
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Đã load model thành công lên {device}")
    return model

def filter_boxes(
        boxes: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray = None,
        track_ids: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lọc chỉ giữ lại các bbox là xe (vehicle classes theo COCO).

    Args:
        boxes: Array bbox [N, 4] format (x1, y1, x2, y2)
        class_ids: Array class ID [N]
        scores: Array confidence scores [N] (optional)
        track_ids: Array track ID [N] (optional)

    Returns:
        Tuple (filtered_boxes, filtered_class_ids, filtered_scores, filtered_track_ids)
    """
    if len(boxes) == 0:
        empty = np.array([])
        return empty, empty, empty, empty

    # Tạo mask cho vehicle classes
    mask = np.isin(class_ids, list(VEHICLE_CLASS_IDS.keys()))

    filtered_boxes = boxes[mask]
    filtered_class_ids = class_ids[mask]
    filtered_scores = scores[mask] if scores is not None else np.array([])
    filtered_track_ids = track_ids[mask] if track_ids is not None else np.array([])

    return filtered_boxes, filtered_class_ids, filtered_scores, filtered_track_ids


class ZoneManager:
    """Quản lý zone masks cho từng camera"""

    def __init__(self, zone_dir: str = "data/zones"):
        """
        Args:
            zone_dir: Thư mục chứa ảnh zone mask
                      Tên file: 0.png, 1.png, 2.png (tương ứng camera_id)
        """
        self.zone_dir = zone_dir
        self.zone_masks: Dict[str, np.ndarray] = {}
        self._load_zones()

    def _load_zones(self):
        """Load tất cả zone masks từ thư mục"""
        if not os.path.exists(self.zone_dir):
            print(f"[ZoneManager] Thư mục zone không tồn tại: {self.zone_dir}")
            return

        # Tìm tất cả file ảnh trong thư mục
        for filename in os.listdir(self.zone_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                camera_id = os.path.splitext(filename)[0]  # "0.png" -> "0"
                filepath = os.path.join(self.zone_dir, filename)

                # Load ảnh zone (grayscale)
                mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Chuyển về binary mask (0 hoặc 255 -> 0 hoặc 1)
                    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                    self.zone_masks[camera_id] = binary_mask
                    print(f"[ZoneManager] Loaded zone cho camera {camera_id}: {mask.shape}")
                else:
                    print(f"[ZoneManager] Không thể load zone: {filepath}")

    def is_in_zone(self, camera_id: str, cx: int, cy: int) -> bool:
        """
        Kiểm tra điểm (cx, cy) có nằm trong zone không.

        Args:
            camera_id: ID camera ("0", "1", "2")
            cx, cy: Tọa độ tâm của bbox

        Returns:
            True nếu nằm trong zone, False nếu không
        """
        if camera_id not in self.zone_masks:
            return True  # Không có zone -> cho qua tất cả

        mask = self.zone_masks[camera_id]
        h, w = mask.shape

        # Kiểm tra bounds
        if not (0 <= cx < w and 0 <= cy < h):
            return False

        return mask[cy, cx] == 1

    def filter_by_zone(
            self,
            camera_id: str,
            boxes: np.ndarray,
            class_ids: np.ndarray = None,
            scores: np.ndarray = None,
            track_ids: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Lọc các bbox có tâm nằm trong zone.

        Args:
            camera_id: ID camera
            boxes: Array bbox [N, 4] format (x1, y1, x2, y2)
            class_ids: Array class ID [N] (optional)
            scores: Array confidence scores [N] (optional)
            track_ids: Array track ID [N] (optional)

        Returns:
            Tuple (filtered_boxes, filtered_class_ids, filtered_scores, filtered_track_ids)
        """
        if len(boxes) == 0:
            empty = np.array([])
            return empty, empty, empty, empty

        if camera_id not in self.zone_masks:
            # Không có zone -> trả về tất cả
            return (
                boxes,
                class_ids if class_ids is not None else np.array([]),
                scores if scores is not None else np.array([]),
                track_ids if track_ids is not None else np.array([]),
            )

        # Tính tâm của mỗi bbox
        centers_x = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
        centers_y = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int)

        # Tạo mask cho các bbox nằm trong zone
        mask = self.zone_masks[camera_id]
        h, w = mask.shape

        in_zone_mask = np.zeros(len(boxes), dtype=bool)
        for i, (cx, cy) in enumerate(zip(centers_x, centers_y)):
            if 0 <= cx < w and 0 <= cy < h:
                in_zone_mask[i] = mask[cy, cx] == 1

        filtered_boxes = boxes[in_zone_mask]
        filtered_class_ids = class_ids[in_zone_mask] if class_ids is not None else np.array([])
        filtered_scores = scores[in_zone_mask] if scores is not None else np.array([])
        filtered_track_ids = track_ids[in_zone_mask] if track_ids is not None else np.array([])

        return filtered_boxes, filtered_class_ids, filtered_scores, filtered_track_ids


# Singleton instance để dùng chung
_zone_manager: ZoneManager = None


def get_zone_manager(zone_dir: str = "data/zones") -> ZoneManager:
    """Lấy singleton ZoneManager instance"""
    global _zone_manager
    if _zone_manager is None:
        _zone_manager = ZoneManager(zone_dir)
    return _zone_manager
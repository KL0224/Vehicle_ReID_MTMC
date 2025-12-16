import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import logging
sys.path.insert(0, str(Path(__file__).parent))
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.engine import DefaultPredictor
from fast_reid.fastreid.data.transforms import build_transforms
from utils import setup_logger

logger = setup_logger(__name__, "extractor.log")

class EmbeddingExtractor:
    """
    Trích xuất embedding vector từ crops sử dụng fast-reid model.

    Flow:
    1. Load pretrained model (SBS hoặc AGW)
    2. Nhận top-K crops từ tracklet
    3. Trích xuất embedding và tính trung bình
    4. Trả về vector 2048-dim.
    """

    def __init__(self, config_file, weights_path, device="cuda"):
        """
        Khởi tạo Extractor.
        :param config_file: đường dẫn file config yaml của fast-reid model
        :param weights_path: Đường dẫn tới file model weights (.pth)
        :param device: "cuda" hoặc "cpu"
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.freeze() # lock config

        # Build predictor
        self.predictor = DefaultPredictor(self.cfg)
        self.model = self.predictor.model

        # Build transform
        self.transform = build_transforms(cfg=self.cfg, is_train=False)

        logger.info(f"[INFO] Loaded REID model weights in {config_file} ...")
        logger.info(f"Device: {self.device}")

    def processes_image(self, image):
        """
        Tiền xử lý dữ liệu theo chuẩn fast-reid:
            + Resize về 256.
            + Normalize với ImageNet mean/std
        :param image: BGR image từ cv2.imread
        :return: tensor (1, 3, , H, W)
        """

        # Convert sang RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Apply transform
        img_tensor = self.transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def extract_single_image(self, image_path_or_array):
        """
        Trích xuất embedding từ 1 ảnh
        :param image_path_or_array: dường dẫn ảnh hoặc numpy array
        :return: embedding numpy array shape (D, )
        """

        # Load image
        if isinstance(image_path_or_array, (str, Path)):
            image = cv2.imread(str(image_path_or_array))
        else:
            image = image_path_or_array

        if image is None:
            raise ValueError(f"Không thể mở ảnh")

        # Preprocessor
        img_tensor = self.processes_image(image)

        # Extract embedding
        with torch.no_grad():
            feat = self.model(img_tensor)

        # Convert to numpy
        embedding = feat.cpu().numpy().flatten()

        # L2 norm
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def extract_from_crops(self, crops, top_k=5, method="mean"):
        """
        Trích xuất embedding của nhiều crops của cùng 1 đối tượng
        :param crops: Danh sách crops dưới dạng numpy array
        :param top_k: Số lượng crops tốt nhất
        :param method: "mean" or "max"
        :return: embedding numpy array shape (D, ) và metadata chứa thông tin bổ sung
        """

        if not crops:
            raise ValueError("Danh sách crops rỗng")

        # Lấy top K - crops
        selected_crops = crops[:min(top_k, len(crops))]

        embeddings = []
        for idx, crop_array in enumerate(selected_crops):
            try:
                embed = self.extract_single_image(crop_array)
                embeddings.append(embed)

            except Exception as e:
                logger.error(f"Bỏ qua crop {idx} vì {e}")
                continue

        embeddings = np.stack(embeddings, axis=0)
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Không có embedding nào cả")

        if method == "mean":
            final_embedding = np.mean(embeddings, axis=0)
        elif method == "max":
            final_embedding = np.max(embeddings, axis=0)
        else:
            raise ValueError(f"Method không hợp lệ: {method}")

        # Chuẩn hóa L2
        final_embedding = final_embedding / np.linalg.norm(final_embedding)

        metadata = {
            "num_crops": len(selected_crops),
            "aggregation_method": method,
            "dimension": final_embedding.shape[0],
        }

        return final_embedding, metadata




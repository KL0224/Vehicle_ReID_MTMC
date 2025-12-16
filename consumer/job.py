import os
import time
import cv2
import yaml
import numpy as np
from confluent_kafka import Consumer, KafkaError
import ast
import threading
from detection import load_model_detector
from matching  import ReIDMatcher, MilvusManager
import pathlib
import supervision as sv
from bytetrack import Tracklet
from queue import Queue
import sys
import warnings
from utils import setup_logger

logger = setup_logger(__name__, "job.log")

# Tắt warnings
warnings.filterwarnings('ignore', category=UserWarning, module="fastreid")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Setup paths (chỉ thêm 1 lần)
BASE_DIR = pathlib.Path(__file__).parent.parent
FASTREID_PATH = BASE_DIR / "consumer" / "reid" / "fast_reid"

if "fastreid" not in sys.modules:
    if str(FASTREID_PATH) not in sys.path:
        sys.path.insert(0, str(FASTREID_PATH))

# Import reid sau cùng (để tránh conflict)
try:
    from reid import EmbeddingExtractor
    logger.info("Reid module imported successfully")
except Exception as e:
    logger.error(f"Error importing reid: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cấu hình đường dẫn
BASE_DIR = pathlib.Path(__file__).parent.parent  # K:\Python\Vehicle
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
REID_CONFIG = BASE_DIR / "consumer" /"reid" / "fast_reid" / "configs" / "VeRi" / "sbs_R50-ibn.yml"
REID_WEIGHTS = BASE_DIR / "models" / "veri_sbs_R50-ibn.pth"

model_lock = threading.Lock()
camera_windows = {}
export_interval = 100

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def embedding_worker(embedding_queue, extractor, tracklet_dict, reid_matcher):
    """Worker thread xử lý embedding extraction"""
    logger.info("Worker thread bắt đầu...")

    while True:
        try:
            cam_id, track_id = embedding_queue.get()
            tracklet = tracklet_dict.get(cam_id)

            if not tracklet:
                logger.warning(f"Không tìm thâ tracklet nào cho camera {cam_id}")
                embedding_queue.task_done()
                continue

            crops = tracklet.get_top_k_crops(track_id, k=5)

            if not crops:
                logger.warning(f"Track {track_id} (cam {cam_id}) không có crops")
                embedding_queue.task_done()
                continue

            logger.info(f"Tiến hành extract embedding: cam {cam_id}, track {track_id}, len {len(crops)} crops")
            try:
                embedding, metadata = extractor.extract_from_crops(crops, top_k=5, method="mean")
                tracklet.save_embeddings(track_id, embedding)

                # Matching
                global_id = reid_matcher.match_or_create(
                    camera_id=cam_id,
                    local_track_id=track_id,
                    embedding=embedding
                )

                # Lưu global_id vào tracklet
                tracklet.save_global_id(track_id, global_id)

                tracklet.clear_crops(track_id)

                logger.info(f"Camera {cam_id}, track {track_id} extract embedding: {len(embedding)}")
            except Exception as e:
                logger.error(f"Lỗi extract embedding Track {track_id}: {e}")

            embedding_queue.task_done()

        except Exception as e:
            logger.error(f"Embedding worker error: {e}")

def consume_and_detect_per_camera(camera_id, topic, group_id, model, stats, embedding_queue, tracklet_dict):
    """Mỗi camera có consumer + worker riêng"""

    # Tạo consumer
    conf = {
        "bootstrap.servers": "localhost:9092",
        "group.id": f"{group_id}_{camera_id}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 300000,
        "session.timeout.ms": 45000,
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    logger.info(f"Camera {camera_id} started - Topic: {topic}")

    # Tạo Tracklet
    byte_tracker = sv.ByteTrack()

    tracklet = Tracklet(
        max_frame_per_track=10,
        min_conf=0.5,
        output_dir=f"tracking_results/camera{camera_id}",
    )

    tracklet_dict[camera_id] = tracklet

    processed = 0  # Đếm số frame xử lý
    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Camera {camera_id} error: {msg.error()}")
                    break

            # Decode frame
            frame_bytes = msg.value()
            metadata = {}
            if msg.headers():
                for key, value in msg.headers():
                    if key == "meta":
                        try:
                            metadata = ast.literal_eval(value.decode())
                        except Exception as e:
                            print(f"Camera {camera_id} error: {value}")

            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_id = metadata.get('frame_id', 'N/A')
            logger.info(f"Camera {camera_id} frame_id: {frame_id}")

            if frame is None:
                continue

            # YOLOv8 detection
            with model_lock:
                results = model(frame, verbose=False)
            num_objects = len(results[0].boxes)

            # Tracking by bytetrack
            detections = sv.Detections.from_ultralytics(results[0])
            detections = byte_tracker.update_with_detections(detections)

            current_tracks_state = {}

            # Tracked tracks
            for track in byte_tracker.tracked_tracks:
                track_id = track.external_track_id
                if track_id >= 0:
                    current_tracks_state[track_id] = 0

            # Lost tracks
            for track in byte_tracker.lost_tracks:
                track_id = track.external_track_id
                if track_id >= 0:
                    current_tracks_state[track_id] = 1

            # Removed tracks
            for track in byte_tracker.removed_tracks:
                track_id = track.external_track_id
                if track_id >= 0:
                    current_tracks_state[track_id] = 2

            # Lưu metadata và crops
            for track_id, bbox, conf in zip(detections.tracker_id, detections.xyxy, detections.confidence):
                if track_id is not None and track_id >= 0:
                    tracklet.update(
                        track_id=track_id,
                        frame=frame,
                        frame_id=frame_id,
                        bbox=bbox,
                        conf=conf,
                    )

            # Phát hiện tracklet removed và đẩy vào queue
            removed_tracks = tracklet.make_removed_tracks(current_tracks_state)
            for track_id in removed_tracks:
                logger.info(f"Camera {camera_id} removed track {track_id} và chuyển vào queue")
                embedding_queue.put((camera_id, track_id))

            processed += 1
            stats[camera_id] = processed

            # Export JSON định kỳ
            if processed % export_interval == 0:
                logger.info("Exporting...")
                logger.info(f"Exporting to json")
                tracklet.export_to_json(camera_id)
                logger.info(f"Exporting embeddings")
                tracklet.export_embeddings(camera_id)
                logger.info(f"[Camera {camera_id}] Exporting crops at frame {processed}...")
                for track_id in tracklet.crops.keys():
                    saved_files = tracklet.export_to_crops(camera_id, track_id)
                    if saved_files:
                        logger.info(f"[Camera {camera_id}] Saved {len(saved_files)} crops for track #{track_id}")

            if processed % 30 == 0:
                logger.info(f"Camera {camera_id} | Frame: {frame_id} | "
                      f"Objects: {num_objects} | Total: {processed}")

    except KeyboardInterrupt:
        print(f"\nCamera {camera_id} stopped by user")

    finally:
        print(f"Saving final results for camera {camera_id}...")

        # Export JSON
        tracklet.export_to_json(camera_id)

        # Export crops cho TẤT CẢ tracks
        logger.info(f"[Camera {camera_id}] Exporting crops for {len(tracklet.crops)} tracks...")

        total_saved = 0
        for track_id in tracklet.crops.keys():
            saved_files = tracklet.export_to_crops(camera_id, track_id)
            total_saved += len(saved_files)

        logger.info(f"[Camera {camera_id}] Total crops saved: {total_saved}")

        consumer.close()
        logger.info(f"Camera {camera_id} stopped. Processed: {processed} frames")

def main():
    config = load_config()

    # Lấy danh sách cameras và topics
    cameras = []
    for topic_key in config["kafka"]["topics"]:
        if topic_key.startswith("cam_stream_"):
            cam_id = topic_key.split("_")[-1]
            topic_name = config["kafka"]["topics"][topic_key]["name"]
            cameras.append((cam_id, topic_name))

    timestamp = int(time.time())
    group_id = f"reid_multicamera_group_{timestamp}"
    logger.info(f"Dùng group_id {group_id}")

    logger.info(f"Khởi động hệ thống ReID Multi-Camera")
    logger.info(f"Số cameras: {len(cameras)}\n")

    # Load YOLOv8 model (chia sẻ giữa các threads)
    logger.info("Load model YOLO...")
    model = load_model_detector(MODEL_PATH)
    logger.info("Đã load model YOLO")

    # Load model reid
    logger.info("Load model REID...")
    extractor = EmbeddingExtractor(
        config_file=str(REID_CONFIG),
        weights_path=str(REID_WEIGHTS),
        device="cuda",
    )

    logger.info("Đã load model REID!")

    # Khởi tạo milvus và matcher
    logger.info("Đang khởi tạo database và matcher")
    milvus_manager = MilvusManager(
        host="localhost",
        port="19531",
        collection_name="vehicle_reid"
    )

    reid_matcher = ReIDMatcher(
        milvus_manager=milvus_manager,
        similarity_threshold=0.7,
        time_window=300
    )

    logger.info("Đã khởi tạo milvus và reid_matcher thành công")

    # Statistics dictionary
    stats = {}
    embedding_queue = Queue()
    tracklet_dict = {}

    worker_thread = threading.Thread(
        target=embedding_worker,
        args=(embedding_queue, extractor, tracklet_dict, reid_matcher),
        daemon=True,
        name="embedding_worker",
    )
    worker_thread.start()

    # Khởi động 1 thread cho MỖI camera
    camera_threads = []
    for cam_id, topic in cameras:
        t = threading.Thread(
            target=consume_and_detect_per_camera,
            args=(cam_id, topic, group_id, model, stats, embedding_queue, tracklet_dict),
            name=f"Camera-{cam_id}"
        )
        t.daemon = True
        t.start()
        camera_threads.append(t)

    # Chờ tất cả camera threads
    try:
        for t in camera_threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("\nHệ thống dừng bởi người dùng")
        logger.info("Đợi embedding extraction hoàn thành...")
        embedding_queue.join()

    milvus_manager.flush()
    milvus_manager.close()

    cv2.destroyAllWindows()
    logger.info(f"\nHoàn thành! Cameras processed: {stats}")

if __name__ == "__main__":
    main()
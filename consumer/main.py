import os
import time
import cv2
import yaml
import numpy as np
from confluent_kafka import Consumer, KafkaError
import ast
import threading
from detection import load_model_detector, filter_boxes, get_zone_manager
from matching import ReIDMatcher, MilvusManager
import pathlib
import supervision as sv
from bytetrack import Tracklet
from queue import Queue
import sys
import warnings
from log_utils import setup_logger

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
BASE_DIR = pathlib.Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
MODEL_PATH = BASE_DIR / "models" / "yolov8x.pt"
REID_CONFIG = BASE_DIR / "consumer" / "reid" / "fast_reid" / "configs" / "VeRi" / "sbs_R50-ibn.yml"
REID_WEIGHTS = BASE_DIR / "models" / "veri_sbs_R50-ibn.pth"
ZONE_DIR = BASE_DIR / "data" / "zones"
model_lock = threading.Lock()


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
                logger.warning(f"Không tìm thấy tracklet nào cho camera {cam_id}")
                embedding_queue.task_done()
                continue

            crops = tracklet.get_top_k_crops(track_id, k=1)

            if not crops or len(crops) < 1:
                logger.warning(f"Track {track_id} (cam {cam_id}) không có crops")
                embedding_queue.task_done()
                continue

            logger.info(f"Tiến hành extract embedding: cam {cam_id}, track {track_id}, len {len(crops)} crops")
            try:
                embedding, metadata = extractor.extract_from_crops(crops, top_k=5, method="mean")
                tracklet.save_embeddings(track_id, embedding)

                frame_start, frame_end = tracklet.get_track_frame_range(track_id)
                logger.info(f"Dùng frame range [{frame_start},{frame_end}] để matching")

                # Matching
                global_id = reid_matcher.match_and_update(
                    camera_id=cam_id,
                    local_track_id=track_id,
                    embedding=embedding,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )

                # Lưu global_id vào tracklet
                tracklet.save_global_id(track_id, global_id)

                saved_files = tracklet.export_to_crops(cam_id, track_id)
                if saved_files:
                    logger.info(f"[Worker] Cam {cam_id} saved {len(saved_files)} crops for track #{track_id}")

                tracklet.remove_track_data(track_id)
                logger.info(f"Camera {cam_id}, track {track_id} -> global_id: {global_id}")
            except Exception as e:
                logger.error(f"Lỗi extract embedding Track {track_id}: {e}")

            embedding_queue.task_done()

        except Exception as e:
            logger.error(f"Embedding worker error: {e}")
            try:
                embedding_queue.task_done()
            except Exception:
                pass


def consume_and_detect_per_camera(camera_id, topic, group_id, model, zone_manager, stats, embedding_queue, tracklet_dict):
    """Mỗi camera có consumer + worker riêng"""

    # Tạo consumer
    conf = {
        "bootstrap.servers": "localhost:9092",
        "group.id": f"{group_id}_{camera_id}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 300000,
        "session.timeout.ms": 45000,
        "fetch.min.bytes": 1,
        "fetch.wait.max.ms": 100,
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    logger.info(f"Camera {camera_id} started - Topic: {topic}")

    # Tạo ByteTrack và Tracklet
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.40,
        minimum_matching_threshold=0.70,
        lost_track_buffer=90,
        frame_rate=10,
        minimum_consecutive_frames=3
    )

    tracklet = Tracklet(
        max_frame_per_track=10,
        min_conf=0.5,
        output_dir=f"tracking_results/camera{camera_id}",
    )

    tracklet_dict[camera_id] = tracklet

    processed = 0
    poll_timeout_count = 0
    max_timeout = 30
    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                poll_timeout_count += 1
                if poll_timeout_count % 100 == 0:
                    logger.warning(f"Camera {camera_id}: {poll_timeout_count} poll timeouts, processed: {processed}")

                if poll_timeout_count >= max_timeout:
                    logger.info(
                        f"Camera {camera_id}: idle {max_timeout:.0f}s (no messages), stopping to export results"
                    )
                    break
                continue

            poll_timeout_count = 0
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"Camera {camera_id}: Reached end of partition, processed: {processed}")
                    break  # Hoặc continue tùy logic
                else:
                    logger.error(f"Camera {camera_id} error: {msg.error()}")
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

            if frame is None:
                continue

            # === BƯỚC 1: YOLOv8 Detection ===
            with model_lock:
                results = model(frame, verbose=False, conf=0.15)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            scores = results[0].boxes.conf.cpu().numpy()

            # === BƯỚC 2: Lọc chỉ giữ xe (vehicle classes) ===
            boxes, class_ids, scores, _ = filter_boxes(boxes, class_ids, scores)

            # === BƯỚC 3: Lọc theo zone ===
            boxes, class_ids, scores, _ = zone_manager.filter_by_zone(
                camera_id=camera_id,
                boxes=boxes,
                class_ids=class_ids,
                scores=scores,
            )

            num_objects = len(boxes)

            # === BƯỚC 4: Tracking với ByteTrack ===
            if len(boxes) > 0:
                # Tạo Detections thủ công từ filtered data
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=scores,
                    class_id=class_ids,
                )
                detections = byte_tracker.update_with_detections(detections)
            else:
                # Không có detection nào sau filter
                detections = sv.Detections.empty()
                byte_tracker.update_with_detections(detections)

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
            if detections.tracker_id is not None:
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
                if tracklet.should_reid(track_id):
                    logger.info(f"Camera {camera_id} removed track {track_id} and go to queue")
                    embedding_queue.put((camera_id, track_id))
                else:
                    logger.info(f"Skip ReID track {track_id} (low quality)")
                    tracklet.remove_track_data(track_id)

            processed += 1
            stats[camera_id] = processed

            if processed % 30 == 0:
                logger.info(f"Camera {camera_id} | Frame: {frame_id} | "
                            f"Objects: {num_objects} | Total: {processed}")

    except KeyboardInterrupt:
        print(f"\nCamera {camera_id} stopped by user")


    finally:
        print(f"Saving final results for camera {camera_id}...")
        tracklet.export_to_json(camera_id)
        tracklet.export_mot(camera_id)
        # Export crops còn lại (tracks chưa bị removed khi kết thúc)
        logger.info(f"[Camera {camera_id}] Exporting remaining crops for {len(tracklet.crops)} tracks...")
        total_saved = 0
        for track_id in list(tracklet.crops.keys()):
            saved_files = tracklet.export_to_crops(camera_id, track_id)
            total_saved += len(saved_files)
        logger.info(f"[Camera {camera_id}] Total remaining crops saved: {total_saved}")
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

    # Load YOLOv8 model
    logger.info("Load model YOLO...")
    model = load_model_detector(MODEL_PATH)
    logger.info("Đã load model YOLO")

    # Load ZoneManager
    logger.info("Load zone masks...")
    zone_manager = get_zone_manager(str(ZONE_DIR))
    logger.info(f"Đã load {len(zone_manager.zone_masks)} zone masks")

    # Load model reid
    logger.info("Load model REID...")
    extractor = EmbeddingExtractor(
        config_file=str(REID_CONFIG),
        weights_path=str(REID_WEIGHTS),
        device="cuda",
    )
    logger.info("Đã load model REID!")

    # Khởi tạo Milvus và ReIDMatcher
    logger.info("Đang khởi tạo database và matcher")
    milvus_manager = MilvusManager(
        host="localhost",
        port="19531",
        collection_name="vehicle_reid",
        init_collection=True,
        drop_existing=True,
    )

    reid_matcher = ReIDMatcher(
        milvus=milvus_manager,
        fps=10,
        similarity_threshold=0.6,
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
            args=(cam_id, topic, group_id, model, zone_manager, stats, embedding_queue, tracklet_dict),
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
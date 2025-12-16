import sys
import os
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
import pandas as pd
import supervision as sv
import cv2
import yaml
import atexit
from typing import Iterable, Tuple, Optional
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct, from_json, expr
from pyspark.sql.types import (
    StructType, StructField, StringType, BinaryType,
    LongType, IntegerType, FloatType, ArrayType, DoubleType
)
from pyspark.sql.streaming.state import GroupStateTimeout, GroupState
from utils import setup_logger
import warnings

# Tắt warnings
warnings.filterwarnings('ignore', category=UserWarning, module="fastreid")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

BASE_DIR = pathlib.Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
env_log_path = os.environ.get("LOG_PATH")
if env_log_path:
    log_path = pathlib.Path(env_log_path)
else:
    log_path = LOG_DIR / "spark_job_all.log"

logger = setup_logger(__name__, str(log_path))

# Cấu hình đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found path file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ========== GLOBAL VARIABLES ==========
# Models cho executor
_detector = None
_extractor = None
_matcher = None
# Milvus broadcast config (driver will broadcast a dict)
_milvus_broadcast = None
# Local singleton milvus manager per executor process
_milvus_local = None
_models_initialized = False

# ========== CLEANUP FUNCTION ==========
def cleanup_models():
    """
    Cleanup resources khi worker process shutdown
    """
    global _detector, _extractor, _matcher, _milvus_local
    _cleanup_lock = threading.Lock()
    with _cleanup_lock:
        try:
            # Giải phóng GPU memory
            if _detector is not None or _extractor is not None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("[Worker] GPU cache cleaned")
                except Exception as e:
                    logger.warning(f"GPU cleanup failed: {e}")

            # Đóng Milvus kết nối cục bộ nếu tồn tại
            try:
                if _milvus_local is not None:
                    _milvus_local.close()
                    logger.info("[Worker] Milvus connection closed")
            except Exception as e:
                logger.warning(f"Failed Milvus connection closed: {e}")

            # Reset global variables
            _detector = None
            _extractor = None
            _matcher = None
            _milvus_local = None

        except Exception as e:
            logger.error(f"[Worker] Cleanup failed: {e}")

atexit.register(cleanup_models)

# ========== LAZY LOAD MODELS (EXECUTOR) ==========
def get_or_create_models():
    """
    Khởi tạo models trong executor với singleton pattern thread-safe.
    """
    global _detector, _extractor, _matcher, _milvus_broadcast, _milvus_local
    global _models_initialized

    # Double-checked locking pattern
    if _models_initialized:
        logger.info("[Worker] get_or_create_models() - reuse existing models")
        return _detector, _extractor, _matcher, _milvus_local

    _init_lock = threading.Lock()

    with _init_lock:
        # Check lại sau khi acquire lock
        if _models_initialized:
            logger.info("[Worker] get_or_create_models() - models already initialized after lock")
            return _detector, _extractor, _matcher, _milvus_local

        try:
            from detection import load_model_detector
            from matching import ReIDMatcher, MilvusManager
            from reid import EmbeddingExtractor

            MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
            REID_CONFIG = BASE_DIR / "consumer" / "reid" / "fast_reid" / "configs" / "VeRi" / "sbs_R50-ibn.yml"
            REID_WEIGHTS = BASE_DIR / "models" / "veri_sbs_R50-ibn.pth"

            logger.info("[Worker]Loading detector & extractor models (SINGLETON)...")
            _detector = load_model_detector(MODEL_PATH)
            _extractor = EmbeddingExtractor(
                config_file=str(REID_CONFIG),
                weights_path=str(REID_WEIGHTS),
                device="cuda"
            )

            # Khởi tạo MilvusManager cục bộ
            milvus_conf = _milvus_broadcast.value if _milvus_broadcast is not None else None
            if milvus_conf and _milvus_local is None:
                try:
                    _milvus_local = MilvusManager(
                        host=milvus_conf.get("host", "localhost"),
                        port=milvus_conf.get("port", "19531"),
                        collection_name=milvus_conf.get("collection_name", "vehicle_reid"),
                        init_collection=False,
                        drop_existing=False
                    )
                    logger.info("[Worker] Init MilvusManager SINGLETON successfully")
                except Exception as e:
                    logger.error(f"[Worker] Bug init MilvusManager: {e}")
                    _milvus_local = None

            _matcher = ReIDMatcher(
                milvus=_milvus_local,
                similarity_threshold=0.8,
                cache_ttl=100
            )

            _models_initialized = True
            logger.info("[Worker] Models loaded ONCE per executor")

        except Exception as e:
            logger.error(f"[Worker] Failed to load models: {e}")
            # Cleanup partial load to avoid memory leak
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[Worker] Partial GPU cleanup done after load failure")
            raise

    return _detector, _extractor, _matcher, _milvus_local


# ========== STATEFUL TRACKING FUNCTION ==========
def update_tracking_state(key: tuple, pdf_iter: Iterable[pd.DataFrame], state: GroupState):
    """
    Stateful tracking cho mỗi camera
    """
    from bytetrack import Tracklet
    import pickle

    camera_id = key[0]
    logger.info(f"[{camera_id}] update_tracking_state called")

    try:
        all_pdfs = list(pdf_iter)
        if not all_pdfs:
            logger.warning(f"[{camera_id}] Batch rỗng - không có data")
            yield pd.DataFrame(columns=["camera_id", "track_id", "frame_id",
                                        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                                        "confidence", "global_id", "state"])
            return

        pdf = pd.concat(all_pdfs, ignore_index=True)
        logger.info(f"[{camera_id}] Processing batch: {len(pdf)} frames")

    except Exception as e:
        logger.error(f"[{camera_id}] Lỗi consume iterator: {e}")
        yield pd.DataFrame(columns=["camera_id", "track_id", "frame_id",
                                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                                    "confidence", "global_id", "state"])
        return

    # LOAD MODELS (MILVUS TỪ BROADCAST -> init local trong get_or_create_models)
    detector, extractor, matcher, milvus = get_or_create_models()

    # Restore state
    if state.exists:
        state_data = state.get
        try:
            tracker = pickle.loads(state_data[0])
            tracklet = pickle.loads(state_data[1])
            frame_counter = state_data[2]
        except Exception:
            print("Tạo ByteTracker mới ")
            # Nếu deserialization fail -> khởi tạo lại
            tracker = sv.ByteTrack()
            tracklet = Tracklet(max_frame_per_track=10, min_conf=0.5,
                                output_dir=f"tracking_results/camera_{camera_id}")
            frame_counter = 0
    else:
        tracker = sv.ByteTrack()
        tracklet = Tracklet(
            max_frame_per_track=10,
            min_conf=0.5,
            output_dir=f"tracking_results/camera_{camera_id}"
        )
        frame_counter = 0

    results = []

    for idx, row in pdf.iterrows():
        frame_counter += 1
        frame_bytes = row['frame']
        frame_id = int(row['frame_id']) if row['frame_id'] is not None else frame_counter

        logger.debug(f"[{camera_id}] Processing frame {frame_id}")

        # Decode frame
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning(f"[{camera_id}] Frame {frame_id} decode failed")
            continue

        # YOLO Detection
        try:
            detections_raw = detector(frame, verbose=False)
            detections = sv.Detections.from_ultralytics(detections_raw[0])
        except Exception as e:
            logger.error(f"[{camera_id}] Detector failed: {e}")
            continue

        # ByteTrack Update
        detections = tracker.update_with_detections(detections)

        # Get Current Tracks State
        current_tracks_state = {}
        for track in tracker.tracked_tracks:
            if track.external_track_id >= 0:
                current_tracks_state[track.external_track_id] = 0

        for track in tracker.lost_tracks:
            if track.external_track_id >= 0:
                current_tracks_state[track.external_track_id] = 1

        for track in tracker.removed_tracks:
            if track.external_track_id >= 0:
                current_tracks_state[track.external_track_id] = 2

        # Update Tracklet
        for track_id, bbox, conf in zip(
                detections.tracker_id,
                detections.xyxy,
                detections.confidence
        ):
            if track_id < 0:
                continue

            tracklet.update(
                track_id=int(track_id),
                frame=frame,
                frame_id=int(frame_id),
                bbox=bbox,
                conf=float(conf)
            )

            results.append({
                "camera_id": camera_id,
                "track_id": int(track_id),
                "frame_id": int(frame_id),
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
                "confidence": float(conf),
                "global_id": -1,
                "state": "tracked"
            })

        # Check Removed Tracks → ReID
        removed_tracks = tracklet.make_removed_tracks(current_tracks_state)

        for r_track_id in removed_tracks:
            crops = tracklet.get_top_k_crops(r_track_id)
            if crops:
                try:
                    # Extract embedding
                    embedding, _ = extractor.extract_from_crops(crops, top_k=5)

                    # Match hoặc tạo global_id
                    if matcher is not None:
                        global_id = matcher.match_or_create(
                            camera_id=camera_id,
                            local_track_id=r_track_id,
                            embedding=embedding
                        )
                    else:
                        # Nếu không có milvus/matcher thì trả về local id as fallback
                        global_id = -1

                    # Save global_id
                    tracklet.save_global_id(r_track_id, global_id)

                    results.append({
                        "camera_id": camera_id,
                        "track_id": int(r_track_id),
                        "frame_id": int(frame_id),
                        "bbox_x1": 0.0,
                        "bbox_y1": 0.0,
                        "bbox_x2": 0.0,
                        "bbox_y2": 0.0,
                        "confidence": 0.0,
                        "global_id": int(global_id),
                        "state": "removed"
                    })

                    logger.info(f"REID MATCHED: Cam {camera_id} Track {r_track_id} → Global {global_id}")

                    # EXPORT CROPS CHO TRACK VỪA KẾT THÚC (DEBUG)
                    try:
                        tracklet.export_to_crops(
                            cam_id=camera_id,
                            track_id=r_track_id,
                            output_crops=None,  # dùng `output_dir` trong Tracklet
                            k=5,
                        )
                    except Exception as e:
                        logger.warning(f"[{camera_id}] Bug export crops for track {r_track_id}: {e}")

                except Exception as e:
                    logger.error(f"Error ReID: {e}")

            tracklet.clear_crops(r_track_id)

    # ---------- EXPORT EMBEDDINGS + MOT ĐỊNH KỲ ----------
    try:
        # Ví dụ: mỗi 300 frame sẽ export 1 lần
        if frame_counter % 300 == 0:
            try:
                tracklet.export_embeddings(cam_id=camera_id)
            except Exception as e:
                logger.warning(f"[{camera_id}] Bug export embeddings: {e}")

            try:
                tracklet.export_mot(cam_id=camera_id)
            except Exception as e:
                logger.warning(f"[{camera_id}] Bug export MOT: {e}")
    except Exception as e:
        logger.warning(f"[{camera_id}] Export failed: {e}")

    # Update state
    try:
        state.update((
            pickle.dumps(tracker),
            pickle.dumps(tracklet),
            frame_counter,
        ))
    except Exception as e:
        logger.error(f"[{camera_id}] Update state failed: {e}")

    state.setTimeoutDuration(600000)

    yield pd.DataFrame(results)

def write_outputs(batch_df, batch_id: int):
    """
    Single place to handle ALL outputs
    - Runs once per micro-batch
    - No duplicate execution
    """
    if batch_df.isEmpty():
        return

    # ---------- CONSOLE ----------
    cams = batch_df.select("camera_id").distinct().collect()
    print(f"Batch {batch_id} cameras:", [r.camera_id for r in cams])
    batch_df.show(20, truncate=False)

    # ---------- WRITE JSON ----------
    (
        batch_df
        .write
        .mode("append")
        .json("tracking_results/")
    )


# ========== MAIN FUNCTION ==========
def main():
    global _milvus_broadcast

    # Set pythonpath
    consumer_path = str(BASE_DIR / "consumer")
    if consumer_path not in sys.path:
        sys.path.insert(0, consumer_path)

    # ========== KHỞI TẠO MILVUS TRONG DRIVER ==========
    try:
        from matching import MilvusManager

        config = load_config(CONFIG_PATH)
        milvus_config = config["milvus"]

        logger.info("Init MilvusManager in driver...")
        milvus_manager = MilvusManager(
            host=milvus_config["host"],
            port=milvus_config["port"],
            collection_name=milvus_config["collection_name"],
            init_collection=True,
            drop_existing=True
        )
        logger.info("Created MilvusManager in driver")

    except Exception as e:
        logger.error(f"Init Milvus failed: {e}")
        raise

    # ========== KHỞI TẠO SPARK SESSION ==========
    logger.info("Khởi tạo spark session")
    spark = SparkSession.builder \
        .appName("VehicleReID_Streaming") \
        .config("spark.sql.shuffle.partitions", "3") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "3")  \
        .config("spark.executor.instances", "3")  \
        .config("spark.dynamicAllocation.enabled", "false")  \
        .config("spark.task.maxFailures", "1") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6") \
        .config("spark.executorEnv.PYTHONPATH", consumer_path) \
        .config("spark.executorEnv.LOG_PATH", str(log_path)) \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("Init spark session successfully")

    # BROADCAST MILVUS SANG EXECUTORS
    milvus_conf = {
        "host": milvus_config["host"],
        "port": milvus_config["port"],
        "collection_name": milvus_config["collection_name"]
    }
    _milvus_broadcast = spark.sparkContext.broadcast(milvus_conf)
    logger.info("Broadcasted MilvusManager to executors")

    # ========== ĐỌC KAFKA STREAM ==========
    logger.info("Spark read message from kafka server")
    kafka_config = config["kafka"]
    KAFKA_BOOTSTRAP_SERVERS = f"{kafka_config['servers']['host']}:{kafka_config['servers']['port']}"
    topics_name = [topic_data['name'] for topic_data in kafka_config['topics'].values() if
                   'cam_stream' in topic_data['name']]
    logger.info(f"Kafka Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"TOPICS: {topics_name}")

    # Tạo separate stream per topic for balanced read
    streams = []
    for topic in topics_name:
        df_topic = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .option("maxOffsetsPerTrigger", "10") \
            .option("includeHeaders", "true") \
            .load()
        streams.append(df_topic)

    # Union all streams
    df_kafka = streams[0]
    for s in streams[1:]:
        df_kafka = df_kafka.union(s)

    # ========== PARSE KAFKA MESSAGE ==========
    logger.info("Parse kafka message")
    metadata_schema = StructType([
        StructField("camera_id", StringType()),
        StructField("frame_id", IntegerType()),
        StructField("timestamp", DoubleType()),
        StructField("original_fps", DoubleType()),
        StructField("sent_fps", IntegerType())
    ])

    df_with_metadata = df_kafka.select(
        col("key").cast("string").alias("camera_id"),
        col("value").alias("frame"),
        expr("headers[0].value").cast("string").alias("metadata_bytes")
    )

    df_parsed = df_with_metadata.withColumn(
        "metadata",
        from_json(
            col("metadata_bytes"),
            metadata_schema
        )
    ).select(
        col("camera_id"),
        col("metadata.frame_id").alias("frame_id"),
        col("frame")
    )

    # ========== APPLY STATEFUL TRACKING ==========
    logger.info("Create output schema")
    output_schema = StructType([
        StructField("camera_id", StringType(), False),
        StructField("track_id", IntegerType(), False),
        StructField("frame_id", LongType(), False),
        StructField("bbox_x1", DoubleType(), False),
        StructField("bbox_y1", DoubleType(), False),
        StructField("bbox_x2", DoubleType(), False),
        StructField("bbox_y2", DoubleType(), False),
        StructField("confidence", DoubleType(), False),
        StructField("global_id", LongType(), False),
        StructField("state", StringType(), False)
    ])

    state_schema = StructType([
        StructField("tracker", BinaryType(), True),
        StructField("tracklet", BinaryType(), True),
        StructField("frame_counter", IntegerType(), True),
    ])

    logger.info("Processing")
    df_tracked = df_parsed \
        .groupBy("camera_id") \
        .applyInPandasWithState(
        func=update_tracking_state,
        outputStructType=output_schema,
        stateStructType=state_schema,
        outputMode="append",
        timeoutConf=GroupStateTimeout.ProcessingTimeTimeout
    )

    # ========== OUTPUT SINKS ==========
    query = (
        df_tracked.writeStream
        .outputMode("append")
        .foreachBatch(write_outputs)
        .option("checkpointLocation", "checkpoints/streaming")
        .trigger(processingTime="30 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
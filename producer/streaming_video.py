import os
import cv2
import time
import threading
import yaml
from confluent_kafka import Producer
from datetime import datetime

CONFIG_PATH = "config/config.yaml"
FPS_TARGET = 10  # FPS muốn gửi

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def delivery_report(err, msg):
    if err:
        print(f"Lỗi gửi message (camera={msg.key().decode()}): {err}")

def stream_video_sampling_real_time(video_path, camera_id, topic, fps_target=FPS_TARGET):
    """
    Stream video theo tốc độ FPS thực tế của video nhưng chỉ gửi các frame được lấy mẫu (sampling).
    """
    # Tạo Producer riêng biệt cho luồng này (Đảm bảo an toàn đa luồng)
    conf = {
        "bootstrap.servers": "localhost:9092",
        "linger.ms": 20,
        "compression.type": "snappy",
        'message.max.bytes': 10485760,
        "batch.size": 1000000,
        "queue.buffering.max.messages": 100000,
    }
    producer = Producer(conf)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không mở được video {video_path}")
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính toán frame_skip và sleep_time
    if original_fps <= 0:
        print(f"FPS của video {video_path} không hợp lệ ({original_fps}). Dừng luồng.")
        return 0

    if original_fps < fps_target:
        print(f"Cảnh báo: FPS gốc ({original_fps:.2f}) nhỏ hơn FPS mục tiêu ({fps_target}). Sẽ gửi mọi frame.")
        frame_skip = 1
    else:
        frame_skip = int(round(original_fps / fps_target))

    # Thời gian chờ cần thiết giữa các frame GỐC
    sleep_time_per_original_frame = 1.0 / original_fps

    print(f"Camera {camera_id} - Bắt đầu stream:")
    print(f"   - FPS Gốc: {original_fps:.2f}. FPS Mục tiêu: {fps_target}.")
    print(f"   - Lấy mẫu (Skip): {frame_skip}. Tổng frames: {total_frames}.")

    sent_count = 0
    frame_id = 0

    start_time_real = time.time()
    try:
        while True:
            # Thời điểm bắt đầu xử lý frame hiện tại
            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            is_sent = False

            if frame_id % frame_skip == 0:
                # FRAME NÀY ĐƯỢC CHỌN ĐỂ GỬI
                frame_timestamp = datetime.utcnow().timestamp()

                ret2, buffer = cv2.imencode(".jpg", frame)
                if ret2:
                    frame_bytes = buffer.tobytes()
                    metadata = {
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "timestamp": frame_timestamp,
                        "original_fps": original_fps,
                        "sent_fps": fps_target
                    }

                    try:
                        producer.produce(
                            topic=topic,
                            value=frame_bytes,
                            key=camera_id.encode(),
                            headers=[("meta", str(metadata).encode())],
                            callback=delivery_report
                        )
                        sent_count += 1
                        is_sent = True

                    except BufferError:
                        producer.poll(1)
                        producer.produce(
                            topic=topic,
                            value=frame_bytes,
                            key=camera_id.encode(),
                            headers=[("meta", str(metadata).encode())],
                            callback=delivery_report
                        )
                        sent_count += 1
                        is_sent = True

                    producer.poll(0)

                if is_sent and frame_id % int(original_fps) == 0:
                     elapsed_stream = time.time() - start_time_real
                     print(f"   -> Cam {camera_id}: Gửi Frame {frame_id}. Tốc độ thực: {elapsed_stream:.2f}s")

            process_time = time.time() - frame_start_time
            time_to_sleep = sleep_time_per_original_frame - process_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            frame_id += 1  # Luôn tăng frame_id sau mỗi frame đọc được
    except KeyboardInterrupt:
        print(f"Camera {camera_id} đã dừng stream bởi người dùng")
    except Exception as e:
        print(f"Camera {camera_id} gặp lỗi {e}")

    finally:
        cap.release()
        remaining = producer.flush(30)

    end_time_real = time.time()

    total_duration_video = total_frames / original_fps
    total_duration_sent = end_time_real - start_time_real

    print(f"\nCamera {camera_id} - HOÀN TẤT:")
    print(f"   - Tổng frames gửi: {sent_count}/{frame_id} (Tỷ lệ 1/{frame_skip})")
    print(f"   - Thời lượng video gốc: {total_duration_video:.2f} giây")
    print(f"   - Thời gian stream thực tế: {total_duration_sent:.2f} giây")
    print(f"   - Còn {remaining} message chưa gửi.")
    return sent_count

def main():
    config = load_config()
    cameras = config["producer"]["cameras_to_stream"]
    video_dir = config["producer"]["video_source_dir"]
    fps_send = FPS_TARGET
    time_start = time.time()
    threads = []
    for cam_id in cameras:
        video_path = os.path.join(video_dir, f"{cam_id}.mp4")
        topic_name = config["kafka"]["topics"][f"cam_stream_{cam_id[-1]}"]["name"]
        if not os.path.exists(video_path):
            print(f"Không tìm thấy video {video_path}")
            continue

        def wrapper(cam=cam_id, path=video_path, topic=topic_name):
            stream_video_sampling_real_time(path, cam, topic, fps_send)

        t = threading.Thread(target=wrapper, name=f"Thread-{cam_id}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    time_end = time.time()
    print(f"Tổng thời gian gửi frame là {time_end - time_start} second.")

if __name__ == "__main__":
    main()

import os
import yaml
from confluent_kafka.admin import AdminClient, NewTopic
import sys

def load_kafka_config(config_path: str = "config/config.yaml"):
    """Tải cấu hình từ file YAML với validation"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Validate cấu trúc config
    if 'kafka' not in config or 'topics' not in config['kafka']:
        raise ValueError("File config thiếu phần 'kafka.topics'")

    return config


def create_topic(config_path: str = "config/config.yaml"):
    """
    Kết nối tới Kafka server và tạo topics từ file config YAML
    """
    config = load_kafka_config(config_path)
    # Cấu hình kết nối tới Kafka server
    admin_conf = {
        "bootstrap.servers": "localhost:9092",
    }

    try:
        admin_client = AdminClient(admin_conf)
        # Test kết nối
        metadata = admin_client.list_topics(timeout=10)
        print(f"Kết nối Kafka thành công! Cluster có {len(metadata.topics)} topics.")
    except Exception as e:
        print(f"Không thể kết nối tới Kafka server: {e}")
        sys.exit(1)

    # Lấy thông tin topics từ file config
    topics_config = config['kafka']['topics']
    replication_factor = config['kafka']['topic_creation']['replication_factor']

    new_topics = []
    for topic_key, topic_info in topics_config.items():
        topic_name = topic_info['name']
        partitions = topic_info['partitions']

        new_topics.append(
            NewTopic(
                topic=topic_name,
                num_partitions=partitions,
                replication_factor=replication_factor
            )
        )
        print(f"Chuẩn bị tạo topic: {topic_name} ({partitions} partitions)")

    if not new_topics:
        print("Không có topic nào trong file config")
        return

    # Tạo topics
    print(f"\nĐang tạo {len(new_topics)} topics...")
    fs = admin_client.create_topics(new_topics, request_timeout=30)

    # Đợi kết quả và in ra màn hình
    success_count = 0
    for topic, future in fs.items():
        try:
            future.result()  # Block until complete
            print(f"Đã tạo topic '{topic}' thành công")
            success_count += 1
        except Exception as e:
            error_msg = str(e)
            if 'TOPIC_ALREADY_EXISTS' in error_msg:
                print(f"Topic '{topic}' đã tồn tại (bỏ qua)")
                success_count += 1
            else:
                print(f"Lỗi khi tạo topic '{topic}': {e}")

    print(f"\nKết quả: {success_count}/{len(new_topics)} topics đã sẵn sàng")

if __name__ == '__main__':
    create_topic()
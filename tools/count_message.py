from confluent_kafka import Consumer, TopicPartition

def count_messages(
    bootstrap_servers: str,
    topic: str,
    timeout: float = 5.0
) -> int:
    """
    Đếm tổng số message trong Kafka topic bằng offset
    (KHÔNG consume message)
    """

    conf = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": "__count_only__",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    }

    consumer = Consumer(conf)

    # Lấy metadata
    metadata = consumer.list_topics(topic, timeout=timeout)

    if topic not in metadata.topics:
        consumer.close()
        raise ValueError(f"Topic '{topic}' không tồn tại")

    partitions = metadata.topics[topic].partitions.keys()

    total_messages = 0

    for p in partitions:
        tp = TopicPartition(topic, p)

        # low = offset đầu, high = offset cuối + 1
        low, high = consumer.get_watermark_offsets(tp, timeout=timeout)

        total_messages += high - low

    consumer.close()
    return total_messages


if __name__ == "__main__":
    topic = "cam_stream_3"
    bootstrap_servers = "localhost:9092"

    total = count_messages(bootstrap_servers, topic)
    print(f"Topic '{topic}' có {total} messages")

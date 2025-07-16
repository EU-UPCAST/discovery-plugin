from confluent_kafka import Producer, Consumer
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'dev.upcast.maggioli-research.gr:9094',
    'security.protocol': 'SASL_PLAINTEXT',
    'sasl.mechanism': 'PLAIN',
    'sasl.username': 'user2',
    'sasl.password': 'b]e+w0}C'
}


class KafkaMessageProducer:
    def __init__(self):
        self.producer = Producer({
            **KAFKA_CONFIG,
            'client.id': 'python-producer'
        })

    def delivery_callback(self, err, msg):
        if err:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

    def send_message(self, topic, message, key=None):
        try:
            # Convert message to JSON string
            message_str = json.dumps(message)

            # Produce message
            self.producer.produce(
                topic,
                value=message_str.encode('utf-8'),
                key=key.encode('utf-8') if key else None,
                callback=self.delivery_callback
            )

            # Wait for any outstanding messages to be delivered
            self.producer.poll(0)

            # Flush to ensure message is sent
            self.producer.flush(timeout=5)
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False

    def close(self):
        try:
            # Wait for any outstanding messages to be delivered
            remaining = self.producer.flush(timeout=5)
            if remaining > 0:
                logger.warning(f"{remaining} messages were not delivered")
        except Exception as e:
            logger.error(f"Error while closing producer: {str(e)}")


class KafkaMessageConsumer:
    def __init__(self, topic, group_id="semih-test"):
        self.consumer = Consumer({
            **KAFKA_CONFIG,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        })
        self.topic = topic
        self.running = True
        self.consumer.subscribe([topic])

    def consume_messages(self, callback=None):
        try:
            while self.running:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                # Decode message value
                try:
                    value = json.loads(msg.value().decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {str(e)}")
                    continue

                logger.info(f"Received message: {value}")
                logger.info(f"Topic: {msg.topic()}, Partition: {msg.partition()}, Offset: {msg.offset()}")

                if callback:
                    try:
                        callback(value)
                    except Exception as e:
                        logger.error(f"Error in callback: {str(e)}")

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.close()

    def close(self):
        self.running = False
        try:
            self.consumer.close()
        except Exception as e:
            logger.error(f"Error while closing consumer: {str(e)}")


def example_usage():
    # Producer example
    producer = KafkaMessageProducer()
    message = {"key": "value", "timestamp": "2024-02-21T12:00:00"}
    producer.send_message("testTopic2", message, key="message-key")
    producer.close()

    # Consumer example
    def message_handler(msg):
        print(f"Processing message: {msg}")

    consumer = KafkaMessageConsumer("testTopic2")
    consumer.consume_messages(callback=message_handler)


if __name__ == "__main__":
    example_usage()
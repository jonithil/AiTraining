from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer
import json
import asyncio
import websockets

KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "anomaly_data"


admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)

existing_topics = admin_client.list_topics()
if TOPIC_NAME not in existing_topics:
    try:
        admin_client.create_topics([NewTopic(name=TOPIC_NAME, num_partitions=1, replication_factor=1)])
        print(f"Topic '{TOPIC_NAME}' created successfully.")
    except Exception as e:
        print(f"Error creating topic: {e}")
else:
    print(f"Topic '{TOPIC_NAME}' already exists.")

admin_client.close()

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    max_request_size=20 * 1024 * 1024  # 20MB limit
)


SERVER_IP = "192.168.1.222"
PORT = "7891"
WEBSOCKET_URL = f"ws://{SERVER_IP}:{PORT}"

async def fetch_data():
    async with websockets.connect(WEBSOCKET_URL, max_size=10 * 1024 * 1024) as websocket:
        while True:
            try:
                data = await websocket.recv()
                print(f"Raw Data: {data}")

                # Attempt to parse JSON
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    print("Received non-JSON data. Skipping...")
                    continue

                # Ensure message is a valid JSON object before sending
                if isinstance(message, dict):
                    producer.send(TOPIC_NAME, value=message).add_callback(
                    lambda metadata: print(f"Message sent to {metadata.topic} partition {metadata.partition}")
                        ).add_errback(
                            lambda error: print(f"Kafka send error: {error}")
                                                    )
                    producer.flush()

                    #print(f"Sent: {message}")
                else:
                    print("Ignored non-JSON object")

            except websockets.exceptions.ConnectionClosed as e:
                print(f"WebSocket connection closed: {e}")
                break
            except Exception as e:
                print(f"Unexpected Error: {e}")
                break

asyncio.run(fetch_data())

import asyncio
import websockets
from confluent_kafka import Producer

SERVER_IP = "192.168.1.222"
WEBSOCKET_PORT = "7893"
KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "docker_data"

conf = {
    "bootstrap.servers": KAFKA_SERVER,
    "message.max.bytes": 100 * 1024 * 1024,
    "linger.ms": 500,
    "batch.num.messages": 100
}
producer = Producer(conf)

WS_URI = f"ws://{SERVER_IP}:{WEBSOCKET_PORT}"

def delivery_report(err, msg):
    if err:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

async def handle_websocket():
    async with websockets.connect(WS_URI, max_size=100 * 1024 * 1024) as websocket:
        while True:
            try:
                message = await websocket.recv()
                print(f"Received Message ({len(message)} bytes): {message}")

                producer.produce(KAFKA_TOPIC, value=message, callback=delivery_report)
                producer.poll(0)  

            except websockets.exceptions.ConnectionClosed as e:
                print(f"WebSocket Connection Closed: {e}")
                await asyncio.sleep(5)  
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)  

async def main():
    while True:
        try:
            await handle_websocket()
        except Exception as e:
            print(f"WebSocket Connection Error: {e}")
            await asyncio.sleep(5)

asyncio.run(main())

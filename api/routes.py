import aiohttp
import asyncio
import time
import psutil
import pynvml
import random
import string
import logging
from typing import List

API_URL = "http://localhost:8000/embed"
TOTAL_ARTICLES = 1000
BATCH_SIZE = 16
CONCURRENT_REQUESTS = 8
CHUNK = False  # toggle for large inputs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/ubuntu/embed_server/api/embed_test.log"),
        logging.StreamHandler()
    ]
)

def random_paragraph(min_words=50, max_words=100):
    return " ".join(
        "".join(random.choices(string.ascii_lowercase + "     ", k=random.randint(4, 12)))
        for _ in range(random.randint(min_words, max_words))
    )

def generate_batches(total: int, batch_size: int) -> List[List[str]]:
    return [[random_paragraph() for _ in range(batch_size)] for _ in range(total // batch_size)]

def log_system_metrics():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gpu_info = "No GPU"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_info = f"GPU: {util.gpu}% | Mem: {mem_info.used / 1e6:.0f}/{mem_info.total / 1e6:.0f} MB"
        pynvml.nvmlShutdown()
    except Exception:
        pass

    logging.info(f"🧠 CPU: {cpu:.1f}% | RAM: {mem.percent:.1f}% | Disk: {disk.percent:.1f}% | {gpu_info}")

async def send_batch(session: aiohttp.ClientSession, idx: int, batch: List[str]):
    try:
        payload = {"texts": batch, "chunk": CHUNK}
        start = time.perf_counter()
        async with session.post(API_URL, json=payload, timeout=240) as resp:
            if resp.status != 200:
                logging.error(f"[{idx}] ❌ Status: {resp.status}")
                return
            await resp.json()
        latency = time.perf_counter() - start
        logging.info(f"[{idx}] ✅ Batch of {len(batch)} | Latency: {latency:.2f}s")
    except Exception as e:
        logging.exception(f"[{idx}] 🚨 Exception during request: {e}")

async def run_load_test():
    batches = generate_batches(TOTAL_ARTICLES, BATCH_SIZE)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(batches), CONCURRENT_REQUESTS):
            tasks = [
                send_batch(session, i + j, batch)
                for j, batch in enumerate(batches[i:i + CONCURRENT_REQUESTS])
            ]
            await asyncio.gather(*tasks)
            if i % (CONCURRENT_REQUESTS * 1) == 0:
                log_system_metrics()

if __name__ == "__main__":
    asyncio.run(run_load_test())

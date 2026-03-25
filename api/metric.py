from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter("embed_requests_total", "Total /embed requests")
REQUEST_LATENCY = Histogram("embed_request_latency_secs", "Latency of /embed requests")

BATCH_SIZE_GAUGE = Gauge("embed_batch_size", "Size of each processed batch")
BATCH_TOKENS_GAUGE = Gauge("embed_batch_tokens", "Total tokens in each batch")
BATCH_DURATION = Histogram("embed_batch_duration_secs", "Inference time per batch")


# Metrics
REQUEST_DURATION = Histogram('embeddings_request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('embeddings_active_requests', 'Active requests')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_mb', 'GPU memory usage in MB')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
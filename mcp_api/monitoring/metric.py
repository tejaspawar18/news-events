from prometheus_client import Counter, Histogram
import time
from contextlib import contextmanager

# ----------------------
# Metrics Definitions
# ----------------------

# Insert Documents
insert_success_total = Counter("insert_documents_success_total", "Total successful document insertions")
insert_failure_total = Counter("insert_documents_failure_total", "Total failed document insertions")
insert_latency = Histogram("insert_documents_latency_seconds", "Latency of insert_documents API")

# Fetch from Scylla
fetch_success_total = Counter("fetch_documents_success_total", "Total successful Scylla fetches")
fetch_failure_total = Counter("fetch_documents_failure_total", "Total failed Scylla fetches")
fetch_latency = Histogram("fetch_documents_latency_seconds", "Latency of fetch_documents API")

# Batch Search
batch_search_success_total = Counter("batch_search_success_total", "Total successful batch search operations")
batch_search_failure_total = Counter("batch_search_failure_total", "Total failed batch search operations")
batch_search_latency = Histogram("batch_search_latency_seconds", "Latency of batch field search API")

# Hybrid Search
hybrid_search_success_total = Counter("hybrid_search_success_total", "Total successful hybrid search operations")
hybrid_search_failure_total = Counter("hybrid_search_failure_total", "Total failed hybrid search operations")
hybrid_search_latency = Histogram("hybrid_search_latency_seconds", "Latency of hybrid search API")


# ----------------------
# Timing Utility
# ----------------------
@contextmanager
def observe_duration(histogram):
    start = time.monotonic()
    try:
        yield
    finally:
        histogram.observe(time.monotonic() - start)

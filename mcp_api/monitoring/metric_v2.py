# mcp_api/monitoring/enhanced_metrics.py

import time
from typing import List
from contextlib import contextmanager
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info, Enum
import psutil
import threading
from mcp_api.core.config import Config


class MetricsCollector:
    """Enhanced metrics collector for comprehensive observability"""
    
    def __init__(self, service_name: str = "mcp_api"):
        self.service_name = service_name
        
        # Service Info
        self.service_info = Info(
            'service_info',
            'Service information',
            ['service', 'version', 'environment']
        )
        
        # HTTP Request Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests by method, endpoint, and status',
            ['method', 'endpoint', 'status_code', 'service']
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'service'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0]
        )
        
        self.http_request_size_bytes = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint', 'service'],
            buckets=[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
        )
        
        self.http_response_size_bytes = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint', 'service'],
            buckets=[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
        )
        
        # Business Logic Metrics
        self.operation_duration_seconds = Histogram(
            'operation_duration_seconds',
            'Duration of business operations',
            ['operation', 'service', 'status'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0]
        )
        
        self.operations_total = Counter(
            'operations_total',
            'Total operations by type and status',
            ['operation', 'service', 'status', 'error_type']
        )
        
        # Document Processing Metrics
        self.documents_processed_total = Counter(
            'documents_processed_total',
            'Total documents processed',
            ['operation', 'service', 'status']
        )
        
        self.batch_size = Histogram(
            'batch_size',
            'Batch processing sizes',
            ['operation', 'service'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
        )
        
        self.embedding_generation_duration = Histogram(
            'embedding_generation_duration_seconds',
            'Time to generate embeddings',
            ['service', 'batch_size_range'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        )
        
        # Database Metrics
        self.database_operations_total = Counter(
            'database_operations_total',
            'Database operations by database and status',
            ['database', 'operation', 'service', 'status']
        )
        
        self.database_operation_duration = Histogram(
            'database_operation_duration_seconds',
            'Database operation duration',
            ['database', 'operation', 'service'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self.database_connection_pool_size = Gauge(
            'database_connection_pool_size',
            'Current database connection pool size',
            ['database', 'service']
        )
        
        self.database_connection_pool_active = Gauge(
            'database_connection_pool_active',
            'Active database connections',
            ['database', 'service']
        )
        
        # Search Metrics
        self.search_results_returned = Histogram(
            'search_results_returned',
            'Number of search results returned',
            ['search_type', 'service'],
            buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500]
        )
        
        self.search_query_complexity = Histogram(
            'search_query_complexity',
            'Search query complexity (length)',
            ['search_type', 'service'],
            buckets=[10, 25, 50, 100, 200, 500, 1000]
        )
        
        self.search_threshold_distribution = Histogram(
            'search_threshold_distribution',
            'Distribution of search thresholds used',
            ['search_type', 'service'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Vector Database Metrics
        self.vector_similarity_scores = Histogram(
            'vector_similarity_scores',
            'Distribution of vector similarity scores',
            ['service', 'field'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        self.qdrant_collection_size = Gauge(
            'qdrant_collection_size',
            'Number of vectors in Qdrant collection',
            ['collection', 'service']
        )
        
        # System Health Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            ['service']
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['service', 'type']  # type: used, available, total
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['service', 'type']  # type: used, free, total
        )
        
        # Application State
        self.application_state = Enum(
            'application_state',
            'Current application state',
            ['service'],
            states=['starting', 'running', 'degraded', 'stopping', 'stopped']
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['service', 'connection_type']
        )
        
        # Error Tracking
        self.errors_total = Counter(
            'errors_total',
            'Total errors by type and severity',
            ['service', 'error_type', 'severity', 'endpoint']
        )
        
        self.slow_queries_total = Counter(
            'slow_queries_total',
            'Total slow queries detected',
            ['service', 'endpoint', 'threshold']
        )
        
        # Custom Business Metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Cache operations by type and result',
            ['service', 'operation', 'result']  # result: hit, miss, error
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['service', 'cache_type']
        )
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_cpu_usage.labels(service=self.service_name).set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_memory_usage.labels(service=self.service_name, type='used').set(memory.used)
                    self.system_memory_usage.labels(service=self.service_name, type='available').set(memory.available)
                    self.system_memory_usage.labels(service=self.service_name, type='total').set(memory.total)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.system_disk_usage.labels(service=self.service_name, type='used').set(disk.used)
                    self.system_disk_usage.labels(service=self.service_name, type='free').set(disk.free)
                    self.system_disk_usage.labels(service=self.service_name, type='total').set(disk.total)
                    
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def set_service_info(self, version: str, environment: str):
        """Set service information"""
        self.service_info.labels(
            service=self.service_name,
            version=version,
            environment=environment
        ).info({'build_time': str(int(time.time()))})
    
    @contextmanager
    def observe_operation(self, operation: str, **labels):
        """Context manager to observe operation duration and status"""
        start_time = time.time()
        status = 'success'
        error_type = 'none'
        
        try:
            yield
        except Exception as e:
            status = 'error'
            error_type = type(e).__name__
            raise
        finally:
            duration = time.time() - start_time
            
            # Record operation metrics
            self.operation_duration_seconds.labels(
                operation=operation,
                service=self.service_name,
                status=status,
                **labels
            ).observe(duration)
            
            self.operations_total.labels(
                operation=operation,
                service=self.service_name,
                status=status,
                error_type=error_type
            ).inc()
    
    @contextmanager
    def observe_database_operation(self, database: str, operation: str):
        """Context manager to observe database operation"""
        start_time = time.time()
        status = 'success'
        
        try:
            yield
        except Exception:
            status = 'error'
            raise
        finally:
            duration = time.time() - start_time
            
            self.database_operation_duration.labels(
                database=database,
                operation=operation,
                service=self.service_name
            ).observe(duration)
            
            self.database_operations_total.labels(
                database=database,
                operation=operation,
                service=self.service_name,
                status=status
            ).inc()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            service=self.service_name
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
            service=self.service_name
        ).observe(duration)
        
        if request_size > 0:
            self.http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint,
                service=self.service_name
            ).observe(request_size)
        
        if response_size > 0:
            self.http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
                service=self.service_name
            ).observe(response_size)
    
    def record_document_processing(self, operation: str, count: int, status: str = 'success'):
        """Record document processing metrics"""
        self.documents_processed_total.labels(
            operation=operation,
            service=self.service_name,
            status=status
        ).inc(count)
        
        self.batch_size.labels(
            operation=operation,
            service=self.service_name
        ).observe(count)
    
    def record_search_metrics(self, search_type: str, query: str, results_count: int, 
                            threshold: float, similarity_scores: List[float] = None):
        """Record search-specific metrics"""
        self.search_results_returned.labels(
            search_type=search_type,
            service=self.service_name
        ).observe(results_count)
        
        self.search_query_complexity.labels(
            search_type=search_type,
            service=self.service_name
        ).observe(len(query))
        
        self.search_threshold_distribution.labels(
            search_type=search_type,
            service=self.service_name
        ).observe(threshold)
        
        # Record similarity score distribution
        if similarity_scores:
            for score in similarity_scores:
                self.vector_similarity_scores.labels(
                    service=self.service_name,
                    field=search_type
                ).observe(score)
    
    def record_embedding_generation(self, duration: float, batch_size: int):
        """Record embedding generation metrics"""
        # Categorize batch size
        if batch_size <= 10:
            size_range = 'small'
        elif batch_size <= 50:
            size_range = 'medium'
        elif batch_size <= 200:
            size_range = 'large'
        else:
            size_range = 'xlarge'
        
        self.embedding_generation_duration.labels(
            service=self.service_name,
            batch_size_range=size_range
        ).observe(duration)
    
    def record_error(self, error_type: str, severity: str = 'error', endpoint: str = 'unknown'):
        """Record error metrics"""
        self.errors_total.labels(
            service=self.service_name,
            error_type=error_type,
            severity=severity,
            endpoint=endpoint
        ).inc()
    
    def record_slow_query(self, endpoint: str, threshold: str):
        """Record slow query detection"""
        self.slow_queries_total.labels(
            service=self.service_name,
            endpoint=endpoint,
            threshold=threshold
        ).inc()
    
    def set_application_state(self, state: str):
        """Set current application state"""
        self.application_state.labels(service=self.service_name).state(state)
    
    def update_connection_count(self, connection_type: str, count: int):
        """Update active connection count"""
        self.active_connections.labels(
            service=self.service_name,
            connection_type=connection_type
        ).set(count)
    
    def update_qdrant_collection_size(self, collection_name: str, size: int):
        """Update Qdrant collection size"""
        self.qdrant_collection_size.labels(
            collection=collection_name,
            service=self.service_name
        ).set(size)


# Global metrics instance
metrics_class = MetricsCollector(service_name=Config.MONITORING)

# Initialize service info (you might want to get these from environment variables)
metrics_class.set_service_info(
    version="1.0.0",  # Get from environment or config
    environment="production"  # Get from environment or config
)

# Set initial application state
metrics_class.set_application_state("starting")


# Decorator for automatic metrics collection
def track_operation(operation_name: str, **labels):
    """Decorator to automatically track operation metrics"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics_class.observe_operation(operation_name, **labels):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics_class.observe_operation(operation_name, **labels):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


def track_database_operation(database: str, operation: str):
    """Decorator to track database operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with metrics_class.observe_database_operation(database, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator
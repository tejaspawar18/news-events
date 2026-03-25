import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import json
from contextvars import ContextVar
from mcp_api.services.logger import get_loki_logger
from mcp_api.core.config import Config
# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_ip_var: ContextVar[Optional[str]] = ContextVar('user_ip', default=None)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_entry['request_id'] = request_id
            
        user_ip = user_ip_var.get()
        if user_ip:
            log_entry['user_ip'] = user_ip
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

class ContextFilter(logging.Filter):
    """Add context variables to log records"""
    
    def filter(self, record):
        # Add extra fields from the record
        if hasattr(record, 'extra') and record.extra:
            record.extra_fields = record.extra
        return True

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup a logger with proper handlers and formatters"""
    
    # logger = logging.getLogger(name)
    # logger.setLevel(getattr(logging, level.upper()))
    
    logger = get_loki_logger(service=Config.MONITORING)
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    os.makedirs(f"{Config.LOG_DIR}", exist_ok=True)
    
    # Console handler with colored output for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler with JSON format for production
    file_handler = logging.handlers.RotatingFileHandler(
        f'{Config.LOG_DIR}/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        f'{Config.LOG_DIR}/error.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(JSONFormatter())
    error_handler.setLevel(logging.ERROR)
    
    # Add context filter
    context_filter = ContextFilter()
    
    # Add handlers and filters
    for handler in [console_handler, file_handler, error_handler]:
        handler.addFilter(context_filter)
        logger.addHandler(handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return setup_logger(name, os.getenv("LOG_LEVEL", "INFO"))

def setup_request_logging():
    """Setup request-specific logging"""
    # Setup access log
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)
    
    # Access log file handler
    access_handler = logging.handlers.RotatingFileHandler(
        f'{Config.LOG_DIR}/access.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    access_handler.setFormatter(JSONFormatter())
    access_logger.addHandler(access_handler)
    access_logger.propagate = False
    
    # Setup performance log
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    
    perf_handler = logging.handlers.RotatingFileHandler(
        f'{Config.LOG_DIR}/performance.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    perf_handler.setFormatter(JSONFormatter())
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False

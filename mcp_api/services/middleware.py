import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import logging
from mcp_api.services.logging_config import request_id_var, user_ip_var

# Get loggers
access_logger = logging.getLogger("access")
perf_logger = logging.getLogger("performance")
app_logger = logging.getLogger("middleware")

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to set request context variables"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Get client IP
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        # Set context variables
        request_id_var.set(request_id)
        user_ip_var.set(client_ip)
        
        # Add to request state for use in handlers
        request.state.request_id = request_id
        request.state.client_ip = client_ip
        
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request info
        request_id = getattr(request.state, 'request_id', 'unknown')
        client_ip = getattr(request.state, 'client_ip', 'unknown')
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        user_agent = request.headers.get("User-Agent", "")
        
        # Log request
        app_logger.info(f"Request started: {method} {path}", extra={
            "event": "request_started",
            "method": method,
            "path": path,
            "query_params": query_params,
            "user_agent": user_agent,
            "request_id": request_id,
            "client_ip": client_ip
        })
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Log exception
            app_logger.error(f"Request failed with exception: {str(e)}", extra={
                "event": "request_exception",
                "method": method,
                "path": path,
                "exception_type": type(e).__name__,
                "request_id": request_id,
                "client_ip": client_ip
            }, exc_info=True)
            
            # Re-raise the exception
            raise
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log access
        access_logger.info("Request completed", extra={
            "event": "request_completed",
            "method": method,
            "path": path,
            "status_code": status_code,
            "processing_time": round(process_time, 3),
            "request_id": request_id,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "query_params": query_params
        })
        
        # Log performance metrics
        if process_time > 1.0:  # Log slow requests
            perf_logger.warning(f"Slow request detected: {method} {path}", extra={
                "event": "slow_request",
                "method": method,
                "path": path,
                "processing_time": round(process_time, 3),
                "request_id": request_id,
                "client_ip": client_ip
            })
        
        # Add performance header
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        
        return response


import os
import logging
from logging.handlers import RotatingFileHandler
# from logging_loki import LokiHandler


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# def get_loki_logger(service, log_dir_=None, console=False):
    
#     logger = logging.getLogger(service)
#     logger.setLevel(logging.INFO)
#     logger.propagate = False  # Prevent propagation to root logger

#     if not logger.handlers:
#         loki_handler = LokiHandler(
#             url="http://172.31.21.66:3100/loki/api/v1/push",
#             tags={"application": "embedding"},
#             version="1",
#         )
#         logger.addHandler(loki_handler)
        
#         if not log_dir_:
#             current_file = os.path.abspath(__file__)
#             repo_root = current_file
#             print(repo_root)
#             print(os.path.basename(repo_root))
#             while repo_root and os.path.basename(repo_root) != "embed_server":
#                 repo_root = os.path.dirname(repo_root)
            
#             log_dir_ = os.path.join(repo_root, "logs")

#         os.makedirs(log_dir_, exist_ok=True)        
#         log_path = os.path.join(log_dir_, f"{service}.log")

#         log_file_handler = RotatingFileHandler(
#             log_path,
#             maxBytes=5_000_000,
#             backupCount= 5
#         )

#         logger_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#         log_file_handler.setFormatter(logger_format)
#         logger.addHandler(log_file_handler)

#         if console:
#             logger.addHandler(logging.StreamHandler())

#     return logger

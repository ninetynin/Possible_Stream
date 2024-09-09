import logging
import os

log_dir = "logging"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "logs.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name):
    return logging.getLogger(name)


def get_analysis_logger():
    analysis_log_path = os.path.join(log_dir, "analysis_logs.log")
    analysis_logger = logging.getLogger("analysis_logger")
    analysis_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(analysis_log_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not analysis_logger.handlers:
        analysis_logger.addHandler(file_handler)
    
    return analysis_logger
import os
import logging
import sys

def setup_logger(log_dir, exp_name):
  os.makedirs(log_dir, exist_ok=True)
  log_path = os.path.join(log_dir, f"{exp_name}.log")

  # Create a custom logger
  logger = logging.getLogger(exp_name)
  logger.setLevel(logging.INFO)

  if not logger.handlers:
      # Formatter: Timestamp | Level | Message
      formatter = logging.Formatter(
          '%(asctime)s | %(levelname)s | %(message)s', 
          datefmt='%Y-%m-%d %H:%M:%S'
      )

      # File Handler
      file_handler = logging.FileHandler(log_path)
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)

      # Console Handler
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(formatter)
      logger.addHandler(console_handler)

  return logger
import os
import logging
import sys
import argparse
import datetime

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

def parse_args():
   date_str = datetime.datetime.now().strftime('%Y-%m-%d')
   parser= argparse.ArgumentParser()

   parser.add_argument('--seed', type=int, default=Config.SEED)
   parser.add_argument('--exp_name', type=str, default=date_str)

   # RAG SETUP
   parser.add_argument('--chunk_limit', type=int, default=Config.CHUNK_LIMIT,
                       help="Max number of chunks to process")
   
   # RAG EXECUTION
   parser.add_argument('--query', type=str)

   # EVAL SETUP
   parser.add_argument('--eval_dataset', type=str)

   return parser.parse_args()

class Config:
   DATA_DIR = './data'
   SOURCES_DIR = './data/sources'
   LOG_DIR = './logs'
   RESULTS_DIR = './results'
   EVAL_DIR = './data/eval'

   SEED = 4

   CHUNK_LIMIT = None

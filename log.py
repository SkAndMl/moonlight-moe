import logging, os, time

if not os.path.exists("log"):
    os.mkdir("log")

fname = f"moe_{int(time.time())}"
logger = logging.getLogger(fname)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler(f"log/{fname}.log", mode='a') 
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
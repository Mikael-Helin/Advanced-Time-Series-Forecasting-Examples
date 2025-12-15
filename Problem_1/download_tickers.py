import pandas as pd
import yfinance as yf
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# -- Variables --

DATA_DIR = "../Data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
NYSE_FILE = os.path.join(DATA_DIR, "nyse_listed.csv")
START_DATE = "1980-01-01"
END_DATE = "2025-01-01"

if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
    
if not os.path.exists(NYSE_FILE):
    exit(NYSE_FILE + " not found")

df = pd.read_csv(NYSE_FILE)
tickers_list = df["ACT Symbol"].dropna().unique().tolist()

def download_ticker(ticker):
    # Random sleep to avoid IP bans (essential for bulk downloads)
    time.sleep(random.uniform(0.1, 0.5))

    try:
        # Simplest possible call - let yfinance handle the networking
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, threads=False, ignore_tz=True, auto_adjust=True, rounding=True)
        if df.empty:
            print("Empty data for " + ticker)
            return False
    except Exception:
        print("Failed to download " + ticker)
        return False
        
    try:
        path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        df.to_csv(path)
        return True
    except Exception:
        print("Failed to save " + ticker)
        return False

# Download all tickers
downloaded_tickers = []
for ticker in tickers_list:
    success = download_ticker(ticker)
    if success:
        downloaded_tickers.append(ticker)

print("List has " + str(len(tickers_list)) + " tickers")
print("Downloaded " + str(len(downloaded_tickers)) + " tickers")
    

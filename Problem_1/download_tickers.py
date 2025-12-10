import pandas as pd
import yfinance as yf
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Constants
DATA_DIR = '../Data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
NYSE_FILE = os.path.join(DATA_DIR, 'nyse_listed.csv')
START_DATE = '1980-01-01'

def setup():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

def get_tickers():
    if not os.path.exists(NYSE_FILE):
        return []
    df = pd.read_csv(NYSE_FILE)
    # Support both column names commonly found in NYSE lists
    col = 'ACT Symbol' if 'ACT Symbol' in df.columns else 'Symbol'
    return df[col].dropna().unique().tolist() if col in df.columns else []

def download_ticker(ticker):
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    
    if os.path.exists(path):
        return

    # Random sleep to avoid IP bans (essential for bulk downloads)
    time.sleep(random.uniform(0.1, 0.5))

    try:
        # Simplest possible call - let yfinance handle the networking
        df = yf.download(ticker, start=START_DATE, progress=False, threads=False)
        
        if not df.empty:
            # Flatten MultiIndex if present (fixes format issues)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Save only valid columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.to_csv(path)

    except Exception:
        pass # Skip errors, we will filter bad files in Step 2

def main():
    setup()
    tickers = get_tickers()
    random.shuffle(tickers) # Shuffle to distribute load
    
    print(f"Downloading {len(tickers)} tickers (Max 8 threads)...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(download_ticker, tickers), total=len(tickers)))

if __name__ == "__main__":
    main()
import pandas as pd
import os
import numpy as np

# --- Variables ---

DEBUG = True

DATA_DIR = "../Data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_TICKERS = os.path.join(DATA_DIR, "nyse_cleaned.csv")
OUTPUT_PARQUET = os.path.join(DATA_DIR, "nyse_cleaned.parquet")

# Date Constraints
# We filter data starting from Jan 1st
MEASUREMENTS_START_DATE = pd.Timestamp("2011-12-22")
MEASUREMENTS_END_DATE = pd.Timestamp("2024-12-31")

# Assume at least 220 trading dates per year, we have 14 years 
MIN_ROWS = 220 * 14

# --- Code ---

def debug_print(something):
    if DEBUG:
        print(something)

# A stock is valid if it starts ON or BEFORE MEASUREMENTS_START_DATE and ends ON or AFTER MEASUREMENTS_END_DATE
valid_tickers = []
valid_dfs = []
for ticker_file in os.listdir(RAW_DATA_DIR):
    if ticker_file.endswith(".csv"):
        # 1. Read CSV
        file_path = os.path.join(RAW_DATA_DIR, ticker_file)
        try:
            df = pd.read_csv(
                file_path, 
                skiprows=[0, 1, 2], 
                names=["Date", "Close", "High", "Low", "Open", "Volume"], 
                usecols=["Date", "Close"], # Load only what we need
                index_col=0, 
                parse_dates=True,
                date_format="%Y-%m-%d"
            )
        except Exception as e:
            print(f"Error reading {ticker_file}: {e}")
            continue

        # 2. Check Validity
        if df.empty: continue
        
        if df.index[0] > MEASUREMENTS_START_DATE:
            continue # Started too late (IPO after Jan 2012)
            
        if df.index[-1] < MEASUREMENTS_END_DATE:
            continue # Delisted before end of 2024

        # 3. Filter Date Range and Close Column
        ticker = ticker_file[:-4]
        df = df.loc[MEASUREMENTS_START_DATE:MEASUREMENTS_END_DATE, ["Close"]]

        if len(df) < MIN_ROWS:
            debug_print(f"Not enough data for {ticker}")
            continue # Not enough data


        # If daily volatility * sqrt(21) is too different from monthly volatility, do not add ticker
        daily_returns = df["Close"] / df["Close"].shift(1) - 1
        daily_vol = daily_returns.std()
        
        monthly_close = df["Close"].resample("ME").last()
        monthly_returns = monthly_close / monthly_close.shift(1) - 1
        monthly_vol = monthly_returns.std()

        if daily_vol * np.sqrt(21) < monthly_vol / 1.25:
            debug_print(f"Removing {ticker} daily volatility {daily_vol} is too different from monthly volatility {monthly_vol}")
            continue
        if daily_vol * np.sqrt(21) > monthly_vol * 1.25:
            debug_print(f"Removing {ticker} daily volatility {daily_vol} is too different from monthly volatility {monthly_vol}")
            continue
        if daily_vol * np.sqrt(21) > 0.6:
            debug_print(f"Removing {ticker} daily volatility {daily_vol} is too high")
            continue
        if monthly_vol > 0.6:
            debug_print(f"Removing {ticker} monthly volatility {monthly_vol} is too high")
            continue

        # 4. Prepare for Export
        clean_df = df[["Close"]].copy()
        clean_df["Ticker"] = ticker
        clean_df.index.name = "Date"
        valid_dfs.append(clean_df)
        valid_tickers.append(ticker)

# 5. Save
if valid_dfs:
    print("Saving valid tickers...")
    with open(CLEANED_TICKERS, "w") as f:
        f.write("\n".join(valid_tickers))
    print(f"Combining {len(valid_dfs)} valid tickers...")
    final_df = pd.concat(valid_dfs)
    final_df.to_parquet(OUTPUT_PARQUET, compression="brotli")
    print(f"Success! Saved to {OUTPUT_PARQUET}")
else:
    print("No tickers matched the criteria.")
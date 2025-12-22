import pandas as pd
import numpy as np
import os

# -- VARIABLES --
INPUT_FILE = "../Data/nyse_cleaned.parquet"
START_CAPITAL = 100_000
DEMO_TICKERS = ["BAC", "JPM", "DB"]
CUTOFF_DATE = "2011-11-11" # Is for analysis, not trading
START_DATE = "2017-01-01" # Is updated to the first available date
END_DATE = "2024-12-31" # Is updated to the last available date
SELL_FEE = 0.01

# 1. Pivot and Resample
df = pd.read_parquet(INPUT_FILE)
MS_prices = df.pivot(columns="Ticker", values="Close").resample("MS").first()
MS_prices = MS_prices.loc[START_DATE:END_DATE]
START_DATE = MS_prices.index.min() # Ensure we start at the first available date
ME_prices = df.pivot(columns="Ticker", values="Close").resample("ME").last()
ME_prices = ME_prices.loc[CUTOFF_DATE:END_DATE]
END_DATE = ME_prices.index.max() # Ensure we end at the last available date

# 2. FIX: Add Cash after pivoting
MS_prices["cash"] = 1.0
ME_prices["cash"] = 1.0
MS_prices = MS_prices.ffill()
ME_prices = ME_prices.ffill()

# 3. Calculate Returns and Volatility
monthly_simple_returns = ME_prices.pct_change() # Keep all history for rolling calc

# Rolling 5yr (60 months) and 1yr (12 months)
# min_periods=12 ensures we get data sooner, but 60 is strictly 5 years
monthly_simple_mean_5yr = monthly_simple_returns.rolling(window=60).mean()
monthly_simple_std_5yr = monthly_simple_returns.rolling(window=60).std()
monthly_simple_std_1yr = monthly_simple_returns.rolling(window=12).std()

# 4. Define Categories
category_limits = {
    "A": {"1yr": 0.03/np.sqrt(12), "5yr": 0.025/np.sqrt(12)},
    "B": {"1yr": 0.06/np.sqrt(12), "5yr": 0.05/np.sqrt(12)},
    "C": {"1yr": 0.12/np.sqrt(12), "5yr": 0.1/np.sqrt(12)},
    "D": {"1yr": 0.3/np.sqrt(12),  "5yr": 0.25/np.sqrt(12)},
    "E": {"1yr": 0.6/np.sqrt(12),  "5yr": 0.5/np.sqrt(12)},
    # F is default
}

# Initialize with 'F'
monthly_categories_1yr = pd.DataFrame("F", index=monthly_simple_std_1yr.index, columns=monthly_simple_std_1yr.columns)
monthly_categories_5yr = pd.DataFrame("F", index=monthly_simple_std_5yr.index, columns=monthly_simple_std_5yr.columns)

# Apply limits (Loose -> Strict overwrites)
for category in ["E", "D", "C", "B", "A"]:
    mask_1 = monthly_simple_std_1yr < category_limits[category]["1yr"]
    monthly_categories_1yr[mask_1] = category
    
    mask_5 = monthly_simple_std_5yr < category_limits[category]["5yr"]
    monthly_categories_5yr[mask_5] = category

# 5. Merge Categories (Conservative approach: Take the higher risk rating)
monthly_categories = np.maximum(monthly_categories_1yr, monthly_categories_5yr)
monthly_categories = pd.DataFrame(monthly_categories, index=monthly_simple_std_1yr.index, columns=monthly_simple_std_1yr.columns)

# 6. Selection Function
def best_category(category, date_str):
    try:
        date = pd.Timestamp(date_str)
        
        # Ensure date is in index, or find nearest previous
        if date not in monthly_categories.index:
            # get_loc with method='pad' finds the nearest previous index
            idx_loc = monthly_categories.index.get_indexer([date], method='pad')[0]
            date = monthly_categories.index[idx_loc]

        # Filter by Category
        current_categories = monthly_categories.loc[date]
        eligible_tickers = current_categories[current_categories == category].index

        if eligible_tickers.empty:
            return "cash", 0.0

        # FIX: Select based on 5-year average return (Momentum), not 1-month spot return
        # If you prefer 1-year momentum, change to: monthly_simple_returns.rolling(12).mean()
        momentum_returns = monthly_simple_mean_5yr.loc[date, eligible_tickers]
        
        max_ticker = momentum_returns.idxmax()
        max_return = momentum_returns.max()
        
        return max_ticker, max_return

    except Exception as e:
        return None, f"Error: {e}"

def get_best_portfolios(date):
    best_ticker = "none"
    best_return = -1.0
    best_portfolios = {}
    for category in ["A", "B", "C", "D", "E", "F"]:
        ticker, ticker_return = best_category(category, date)
        if ticker_return > best_return:
            best_return = ticker_return
            best_ticker = ticker
        best_portfolios[category] = {
            "ticker": ticker,
            "return": ticker_return,
        }
    return best_portfolios
    

# 7. Buy/Sell Logic

def buy_ticker(ticker, date, capital):
    if ticker == "cash":
        return {
            "ticker": "cash",
            "date": date,
            "price": 1.0,
            "num_shares": capital,
        }
    else:
        price = MS_prices.loc[date, ticker]
        shares = capital / price
        return {
            "ticker": ticker,
            "date": date,
            "price": price,
            "num_shares": shares,
        }

def sell_ticker(ticker, date, num_shares):
    if ticker == "cash":
        return num_shares
    else:
        price = MS_prices.loc[date, ticker]
        capital = num_shares * price * (1 - SELL_FEE)
        return capital

# 8.Initialize Portfolios

temp_portfolios = {}
cash_portfolio = {
    "ticker": "cash",
    "price": 1.0,
    "num_shares": START_CAPITAL,
    "networth": START_CAPITAL,
    "rebalance": False,
}
for category in ["A", "B", "C", "D", "E", "F"]:
    temp_portfolios[category] = cash_portfolio.copy()

# 9. Trade bot
portfolio_history = []
for date in MS_prices.index:
    # Update keep portfolios (date is a pandas Timestamp from the index)
    keep_portfolios = {}
    # Map month-start (MS) dates to month-end (ME) indices used by monthly_* datasets
    try:
        me_idx = monthly_simple_mean_5yr.index.get_indexer([date], method='pad')[0]
        if me_idx == -1:
            me_date = monthly_simple_mean_5yr.index[0]
        else:
            me_date = monthly_simple_mean_5yr.index[me_idx]
    except Exception:
        me_date = date
    for category in ["A", "B", "C", "D", "E", "F"]:
        ticker = temp_portfolios[category]["ticker"]
        price = MS_prices.loc[date, ticker]
        num_shares = temp_portfolios[category]["num_shares"]
        monthly_simple_return_mean_5yr = monthly_simple_mean_5yr.loc[me_date, ticker]
        monthly_category = monthly_categories.loc[me_date, ticker]
        keep_portfolios[category] = {
            "ticker": ticker,
            "date": date,
            "price": price,
            "num_shares": num_shares,
            "networth": price * num_shares,
            "rebalance": monthly_category > category, # Rebalance if current category is higher risk
        }
    # Update candidate portfolios
    rebalance_portfolios = {}
    best_portfolios = get_best_portfolios(date)
    for category in ["A", "B", "C", "D", "E", "F"]:
        networth_sell = keep_portfolios[category]["networth"]
        if keep_portfolios[category]["ticker"] != "cash":
            networth_sell = networth_sell * (1 - SELL_FEE)
        best_ticker = best_portfolios[category]["ticker"]
        price = MS_prices.loc[date, best_ticker]
        num_shares_buy = buy_ticker(best_ticker, date, networth_sell)
        rebalance_portfolios[category] = {
            "ticker": best_ticker,
            "date": date,
            "price": price,
            "num_shares": num_shares_buy["num_shares"],
            "networth": networth_sell,
            "rebalance": False,
        }
    # Decide to keep or rebalance
    temp_portfolios = {}
    for category in ["A", "B", "C", "D", "E", "F"]:
        if keep_portfolios[category]["rebalance"]:
            temp_portfolios[category] = rebalance_portfolios[category].copy()
        else:
            # Compare which has hiher return
            keep_ticker = keep_portfolios[category]["ticker"]
            rebalance_ticker = rebalance_portfolios[category]["ticker"]
            # Safely fetch returns (default to 0.0 if ticker is 'cash' or missing)
            try:
                keep_return = monthly_simple_mean_5yr.loc[me_date, keep_ticker] if keep_ticker in monthly_simple_mean_5yr.columns else 0.0
            except Exception:
                keep_return = 0.0
            try:
                rebalance_return = monthly_simple_mean_5yr.loc[me_date, rebalance_ticker] if rebalance_ticker in monthly_simple_mean_5yr.columns else 0.0
            except Exception:
                rebalance_return = 0.0
            if rebalance_return > keep_return:
                temp_portfolios[category] = rebalance_portfolios[category].copy()
            else:
                temp_portfolios[category] = keep_portfolios[category].copy()
    # Log portfolio status
    print(f"Date: {date.date()}, A: {temp_portfolios['A']['ticker']} {temp_portfolios['A']['networth']:.2f}, B: {temp_portfolios['B']['ticker']} {temp_portfolios['B']['networth']:.2f}, C: {temp_portfolios['C']['ticker']} {temp_portfolios['C']['networth']:.2f}, D: {temp_portfolios['D']['ticker']} {temp_portfolios['D']['networth']:.2f}, E: {temp_portfolios['E']['ticker']} {temp_portfolios['E']['networth']:.2f}, F: {temp_portfolios['F']['ticker']} {temp_portfolios['F']['networth']:.2f}")

print("\nFinal Networth:")
for category in ["A", "B", "C", "D", "E", "F"]:
    ticker = temp_portfolios[category]["ticker"]
    capital = temp_portfolios[category]["networth"]
    if ticker != "cash":
        capital = capital * (1 - SELL_FEE)
    print(f"{category}: {capital:.2f}")
    
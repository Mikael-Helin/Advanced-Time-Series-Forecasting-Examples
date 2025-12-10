import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../Data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
VALID_TICKERS_FILE = os.path.join(DATA_DIR, 'valid_daily_tickers.csv')
START_DATE = '2017-01-02'
END_DATE = '2024-12-31' # Run until end of 2024
INITIAL_CAPITAL = 100000.0
FEE = 0.01

# Categories
CATEGORIES = {
    'A': {'target_vol': 0.025, 'max_vol': 0.03},
    'B': {'target_vol': 0.05, 'max_vol': 0.06},
    'C': {'target_vol': 0.10, 'max_vol': 0.12},
    'D': {'target_vol': 0.20, 'max_vol': 0.24},
    'E': {'target_vol': float('inf'), 'max_vol': float('inf')}
}

def get_first_mondays(start_year, end_year):
    """Get all first Mondays of each month in the range."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Find first day of month
            d = datetime(year, month, 1)
            # Find first Monday (weekday 0)
            while d.weekday() != 0:
                d += timedelta(days=1)
            dates.append(d)
    return dates

def load_all_data(tickers):
    """Load Close prices for all tickers into a single DataFrame."""
    print("Loading data...")
    price_data = {}
    for ticker in tickers:
        file_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue
        try:
            # Handle multi-header format
            df = pd.read_csv(file_path, header=None, skiprows=3)
            if df.empty:
                continue
            df[0] = pd.to_datetime(df[0])
            df.set_index(0, inplace=True)
            # Close is column 4
            price_data[ticker] = pd.to_numeric(df[4], errors='coerce')
        except Exception:
            print(f"Error loading data for {ticker}")
            continue
            
    prices_df = pd.DataFrame(price_data)
    prices_df.sort_index(inplace=True)
    return prices_df

def calculate_metrics(prices_df, current_date):
    """
    Calculate Volatility (1yr) and Expected Return (5yr) for all tickers at current_date.
    Returns a DataFrame with columns ['vol', 'exp_ret'].
    """
    # Lookback windows
    one_year_ago = current_date - timedelta(days=365)
    five_years_ago = current_date - timedelta(days=365*5)
    
    # Slice data
    # We need data UP TO current_date (exclusive of current_date for prediction? 
    # Usually we use data available at close of previous day to trade on current_date open/close)
    # Let's assume we use data up to current_date - 1 day.
    
    history = prices_df.loc[:current_date] # Includes current_date if present, but we should probably shift?
    # If we trade on Monday, we know Friday's close.
    # So we use data up to current_date.
    
    if history.empty:
        return pd.DataFrame()
        
    # Calculate daily returns
    returns = history.pct_change(fill_method=None)
    
    # 1 Year Volatility
    # Last 252 trading days
    last_year_returns = returns.iloc[-252:]
    if len(last_year_returns) < 200: # Minimum requirement
        vol = pd.Series(index=returns.columns, dtype=float) # NaN
    else:
        vol = last_year_returns.std() * np.sqrt(252)
        
    # 5 Year Expected Return
    # Last 5 years (approx 1260 days)
    last_5yr_returns = returns.iloc[-1260:]
    if len(last_5yr_returns) < 1000: # Minimum requirement
        exp_ret = pd.Series(index=returns.columns, dtype=float)
    else:
        exp_ret = last_5yr_returns.mean() * 252
        
    metrics = pd.DataFrame({'vol': vol, 'exp_ret': exp_ret})
    return metrics

def run_solver():
    # 1. Load valid tickers
    if not os.path.exists(VALID_TICKERS_FILE):
        print("Valid tickers file not found.")
        return
    valid_tickers = pd.read_csv(VALID_TICKERS_FILE)['Ticker'].tolist()
    
    # 2. Load Data
    prices_df = load_all_data(valid_tickers)
    print(f"Loaded data for {len(prices_df.columns)} tickers.")
    
    # 3. Simulation Dates
    # Start: Jan 2017. End: Dec 2024.
    # "Trading starts from Monday 2017 January 2nd"
    # "First time one can rebalance is on Monday 2017 February 1st" (which is Feb 6th)
    # We will make an INITIAL selection on Jan 2nd 2017.
    
    trading_dates = get_first_mondays(2017, 2024)
    trading_dates = [d for d in trading_dates if d >= datetime(2017, 1, 2) and d <= datetime(2024, 12, 31)]
    
    # State for each category
    # { 'A': {'cash': 100k, 'ticker': None, 'shares': 0, 'value': 100k}, ... }
    portfolios = {cat: {'cash': INITIAL_CAPITAL, 'ticker': None, 'shares': 0, 'value': INITIAL_CAPITAL} for cat in CATEGORIES}
    
    print(f"Starting simulation on {len(trading_dates)} rebalancing dates...")
    
    for i, date in enumerate(trading_dates):
        date_str = date.strftime('%Y-%m-%d')
        # print(f"Processing {date_str}...")
        
        # Get metrics for this date
        metrics = calculate_metrics(prices_df, date)
        if metrics.empty:
            continue
            
        # Current prices for execution
        # We assume we trade at the Close of this day (or Open? README doesn't specify, usually Close for simplicity)
        try:
            current_prices = prices_df.loc[date]
        except KeyError:
            # If date not in index (e.g. holiday), use next available
            # print(f"Date {date_str} not in data, looking forward...")
            future = prices_df.loc[date:]
            if future.empty:
                break
            current_prices = future.iloc[0]
            # Update date for logging?
            
        for cat, params in CATEGORIES.items():
            state = portfolios[cat]
            current_ticker = state['ticker']
            
            # Update Portfolio Value
            if current_ticker:
                if current_ticker in current_prices and not pd.isna(current_prices[current_ticker]):
                    price = current_prices[current_ticker]
                    state['value'] = state['cash'] + state['shares'] * price
                else:
                    # Price missing, keep old value? Or assume 0?
                    # Keep old value roughly
                    pass
            else:
                state['value'] = state['cash']
                
            # --- Rebalancing Logic ---
            
            # 1. Check if we MUST rebalance (Volatility constraint)
            must_rebalance = False
            current_ticker_metrics = None
            
            if current_ticker:
                if current_ticker in metrics.index:
                    current_vol = metrics.loc[current_ticker, 'vol']
                    current_exp_ret = metrics.loc[current_ticker, 'exp_ret']
                    current_ticker_metrics = {'vol': current_vol, 'exp_ret': current_exp_ret}
                    
                    if current_vol > params['max_vol']:
                        must_rebalance = True
                        # print(f"[{cat}] {date_str}: Forced rebalance. {current_ticker} Vol {current_vol:.4f} > {params['max_vol']}")
                else:
                    # Data missing for current ticker, must sell?
                    must_rebalance = True
            else:
                # Cash -> Must buy if possible
                must_rebalance = True
                
            # 2. Find Best Candidate
            # Filter candidates by target_vol
            candidates = metrics[metrics['vol'] <= params['target_vol']]
            
            if candidates.empty:
                # No valid candidates. Go to Cash.
                if current_ticker:
                    # Sell everything
                    price = current_prices.get(current_ticker, 0)
                    proceeds = state['shares'] * price
                    state['cash'] += proceeds * (1 - FEE) # Fee on sell? "To rebalance... costs 1% of portfolio value"
                    # Actually "To rebalance a portfolio, costs 1% of the portfolio value."
                    # This usually means 1% of the *transacted* amount or the whole portfolio?
                    # "1% fee as penalty of the whole portfolio value."
                    # So if we change ANYTHING, we pay 1% of TOTAL portfolio value.
                    
                    fee = state['value'] * FEE
                    state['cash'] -= fee
                    
                    state['shares'] = 0
                    state['ticker'] = None
                continue
                
            # Pick best expected return
            best_ticker = candidates['exp_ret'].idxmax()
            best_metric = candidates.loc[best_ticker]
            
            # 3. Decide to Switch
            do_switch = False
            
            if must_rebalance:
                do_switch = True
            elif current_ticker:
                # Check optimization condition
                # "New > Current / 0.99"
                if best_metric['exp_ret'] > (current_ticker_metrics['exp_ret'] / 0.99):
                    do_switch = True
                    # print(f"[{cat}] {date_str}: Opportunistic switch. {best_ticker} ({best_metric['exp_ret']:.4f}) > {current_ticker} ({current_ticker_metrics['exp_ret']:.4f})")
            
            if do_switch and best_ticker != current_ticker:
                # Execute Switch
                # 1. Sell current (if any)
                portfolio_val_before_trade = state['value']
                
                # Fee is 1% of portfolio value
                fee = portfolio_val_before_trade * FEE
                
                # Net equity available for new purchase
                equity = portfolio_val_before_trade - fee
                
                if equity <= 0:
                    state['shares'] = 0
                    state['cash'] = 0
                    state['ticker'] = None
                    continue
                
                # 2. Buy new
                new_price = current_prices.get(best_ticker)
                if new_price and not pd.isna(new_price):
                    state['shares'] = equity / new_price
                    state['ticker'] = best_ticker
                    state['cash'] = 0 # All in
                    state['value'] = equity # Updated value
                else:
                    # Cannot buy, stay in cash (minus fee)
                    state['shares'] = 0
                    state['ticker'] = None
                    state['cash'] = equity
                    
    # Final Results
    print("\nResults at 2024-12-31:")
    for cat in CATEGORIES:
        val = portfolios[cat]['value']
        print(f"    {cat}: {val:,.2f}")

if __name__ == "__main__":
    run_solver()

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../Data/raw')

def get_data(ticker):
    """
    Reads data for a ticker.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        return None
    
    # Handle the multi-header format
    try:
        df = pd.read_csv(file_path, header=None, skiprows=3)
        if df.empty:
            return None
            
        # Columns: Date, Open, High, Low, Close, Volume
        # We need Date (0) and Close (4)
        df = df[[0, 4]]
        df.columns = ['Date', 'Close']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error reading {ticker}: {e}")
        return None

def compute_log_returns(ticker, start_date, end_date, returns_type='daily'):
    """
    Computes log returns for a ticker.
    returns_type: 'daily', 'weekly', 'monthly', 'yearly'
    """
    df = get_data(ticker)
    if df is None:
        return None
        
    # Filter by date
    df = df.loc[start_date:end_date]
    if df.empty:
        return None
        
    close_prices = df['Close']
    
    if returns_type == 'daily':
        resampled = close_prices
    elif returns_type == 'weekly':
        # Weekly (Friday)
        resampled = close_prices.resample('W-FRI').last()
    elif returns_type == 'monthly':
        # Monthly (Last day of month)
        resampled = close_prices.resample('ME').last()
    elif returns_type == 'yearly':
        # Yearly (Last day of year)
        resampled = close_prices.resample('YE').last()
    else:
        raise ValueError("Invalid returns_type")
        
    resampled = resampled.dropna()
    
    # Log Return = ln(P_t / P_{t-1})
    log_returns = np.log(resampled / resampled.shift(1))
    
    # First value is NaN
    log_returns = log_returns.dropna()
    
    return log_returns

def compute_correlation_matrix(ticker_1, ticker_2, start_date, end_date, correlation_type='daily'):
    """
    Computes correlation between two tickers.
    If ticker_1 == ticker_2, returns variance.
    """
    r1 = compute_log_returns(ticker_1, start_date, end_date, correlation_type)
    r2 = compute_log_returns(ticker_2, start_date, end_date, correlation_type)
    
    if r1 is None or r2 is None:
        return np.nan
        
    # Align dates
    combined = pd.concat([r1, r2], axis=1, join='inner')
    if combined.empty:
        return np.nan
        
    r1_aligned = combined.iloc[:, 0]
    r2_aligned = combined.iloc[:, 1]
    
    if ticker_1 == ticker_2:
        return r1_aligned.var()
    else:
        return r1_aligned.corr(r2_aligned)

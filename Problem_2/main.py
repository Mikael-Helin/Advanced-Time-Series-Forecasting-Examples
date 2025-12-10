import pandas as pd
import numpy as np
from metrics import compute_log_returns, compute_correlation_matrix, get_data
from portfolio import portfolio_value

# Constants
START_DATE = '2017-01-02'
END_DATE = '2024-12-09'
TICKER = 'AAPL' 

def test_additivity():
    print(f"--- Testing Additivity for {TICKER} ---")
    
    daily_returns = compute_log_returns(TICKER, START_DATE, END_DATE, 'daily')
    weekly_returns = compute_log_returns(TICKER, START_DATE, END_DATE, 'weekly')
    monthly_returns = compute_log_returns(TICKER, START_DATE, END_DATE, 'monthly')
    yearly_returns = compute_log_returns(TICKER, START_DATE, END_DATE, 'yearly')
    
    if daily_returns is None:
        print("Error fetching returns.")
        return

    # Daily vs Weekly
    daily_sum_weekly = daily_returns.resample('W-FRI').sum()
    common = daily_sum_weekly.index.intersection(weekly_returns.index)
    diff = np.abs(daily_sum_weekly[common] - weekly_returns[common]).max()
    print(f"Daily vs Weekly Max Diff: {diff:.6f}")
    
    # Daily vs Monthly
    daily_sum_monthly = daily_returns.resample('ME').sum()
    common_m = daily_sum_monthly.index.intersection(monthly_returns.index)
    diff_m = np.abs(daily_sum_monthly[common_m] - monthly_returns[common_m]).max()
    print(f"Daily vs Monthly Max Diff: {diff_m:.6f}")

    # Daily vs Yearly
    daily_sum_yearly = daily_returns.resample('YE').sum()
    common_y = daily_sum_yearly.index.intersection(yearly_returns.index)
    diff_y = np.abs(daily_sum_yearly[common_y] - yearly_returns[common_y]).max()
    print(f"Daily vs Yearly Max Diff: {diff_y:.6f}")

def test_correlations():
    print("\n--- Testing Correlations ---")
    tickers = ['AAPL', 'DIS', 'GE', 'XOM']
    frequencies = ['daily', 'weekly', 'monthly', 'yearly']
    
    for freq in frequencies:
        print(f"\nCorrelation Matrix ({freq}):")
        matrix = pd.DataFrame(index=tickers, columns=tickers)
        for t1 in tickers:
            for t2 in tickers:
                val = compute_correlation_matrix(t1, t2, START_DATE, END_DATE, freq)
                matrix.loc[t1, t2] = val
        print(matrix)

def test_portfolio():
    print("\n--- Testing Portfolio Value ---")
    initial_val = 100000
    start = '2020-01-02'
    end = '2020-12-31'
    
    portfolio = {'AAPL': 1.0}
    val = portfolio_value(initial_val, start, end, portfolio)
    print(f"AAPL 2020 Value: {val:.2f}")

if __name__ == "__main__":
    test_additivity()
    test_correlations()
    test_portfolio()

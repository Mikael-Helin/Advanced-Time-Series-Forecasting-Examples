import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Constants
INPUT_FILE = '../Data/combined_data.parquet'            # The raw data from Step 1
OUTPUT_FILE = '../Data/closing_prices.parquet'          # The result we want to save
START_CAPITAL = 100_000
REBALANCE_COST = 0.01
END_DATE = pd.Timestamp('2024-12-31')
DEMO_TICKERS = ['BAC', 'JPM', 'DB']

# Categories
CATEGORIES = {
    'A': {'target_vol': 0.025, 'max_vol': 0.03},
    'B': {'target_vol': 0.05, 'max_vol': 0.06},
    'C': {'target_vol': 0.10, 'max_vol': 0.12},
    'D': {'target_vol': 0.20, 'max_vol': 0.24},
    'E': {'target_vol': float('inf'), 'max_vol': float('inf')}
}

# Portfolio Strategy
# Each list item is a tuple of (ticker, value)
portfolio_history = {
    'A': [],
    'B': [],
    'C': [],
    'D': [],
    'E': []
}

def test_log_returns(df_test):
    """
    Compute and validate log returns across daily, weekly, monthly, yearly horizons.
    """
    # Clean invalid values
    df_test = df_test.where(df_test > 0)

    # Log returns
    daily_log_returns = np.log(df_test / df_test.shift(1))
    weekly_log_returns = np.log(df_test.resample('W-FRI').last() / df_test.resample('W-FRI').last().shift(1))
    monthly_log_returns = np.log(df_test.resample('ME').last() / df_test.resample('ME').last().shift(1))
    yearly_log_returns = np.log(df_test.resample('YE').last() / df_test.resample('YE').last().shift(1))

    # --- Additivity checks for each ticker ---
    for ticker in df_test.columns:
        print(f"\nRelative additivity checks for {ticker}:")

        # Reference scale: average absolute yearly log return
        weekly_scale = weekly_log_returns[ticker].abs().mean()
        monthly_scale = monthly_log_returns[ticker].abs().mean()
        yearly_scale = yearly_log_returns[ticker].abs().mean()

        # 5 daily ≈ 1 weekly
        daily_sum = daily_log_returns[ticker].rolling(5).sum()
        diff_weekly = (daily_sum - weekly_log_returns[ticker]).dropna()
        rel_weekly = diff_weekly.abs().mean() / weekly_scale
        print(f"Relative error 5d vs 1w (per week): {rel_weekly*100:.2f}%")

        # 21 daily ≈ 1 monthly
        daily_sum = daily_log_returns[ticker].rolling(21).sum()
        diff_monthly = (daily_sum - monthly_log_returns[ticker]).dropna()
        rel_monthly = diff_monthly.abs().mean() / monthly_scale
        print(f"Relative error 21d vs 1m (per month): {rel_monthly*100:.2f}%")

        # 252 daily ≈ 1 yearly
        daily_sum = daily_log_returns[ticker].rolling(252).sum()
        diff_yearly = (daily_sum - yearly_log_returns[ticker]).dropna()
        rel_yearly = diff_yearly.abs().mean() / yearly_scale
        print(  f"Relative error 252d vs 1y (per year): {rel_yearly*100:.2f}%")

    # --- Correlation matrices ---
    print("\nCorrelation matrix (daily):")
    print(daily_log_returns.corr())

    print("\nCorrelation matrix (weekly):")
    print(weekly_log_returns.corr())

    print("\nCorrelation matrix (monthly):")
    print(monthly_log_returns.corr())

    print("\nCorrelation matrix (yearly):")
    print(yearly_log_returns.corr())

def log_to_simple_monthly_mean(log_mean, log_std):
    return np.exp(log_mean+log_std**2/2) - 1

def log_to_simple_yr_std(log_mean, log_std):
    return np.sqrt(12*(np.exp(log_std**2)-1)*np.exp(2*log_mean+log_std**2))

def get_category(std_1y_monthly, std_5y_monthly):
    """Based on simple returns"""
    for category in ['A', 'B', 'C', 'D']:
        if std_1y_monthly <= CATEGORIES[category]['target_vol'] and std_5y_monthly <= CATEGORIES[category]['max_vol']:
            return category
    return 'E'

def optimize_portfolio(df_closing):

    def buy_ticker(ticker, trading_day, value):
        """Buy a ticker and return the number of shares"""
        ticker_price = df_closing.loc[trading_day, ticker]
        num_shares = value / ticker_price
        return num_shares

    def sell_ticker(ticker, trading_day, num_shares):
        """Sell a ticker and return the value"""
        ticker_price = df_closing.loc[trading_day, ticker]
        value = num_shares * ticker_price
        return value

    def find_best_in_group(rebalance_date):
        one_year_ago = rebalance_date - pd.Timedelta(days=365)
        five_years_ago = rebalance_date - pd.Timedelta(days=365*5)

        # Do not dropna() here, as it drops the entire row if ANY ticker is missing data
        monthly_prices_1y = df_closing.loc[one_year_ago:rebalance_date].resample('ME').last()
        monthly_prices_5y = df_closing.loc[five_years_ago:rebalance_date].resample('ME').last()
        
        monthly_log_returns_1y = np.log(monthly_prices_1y / monthly_prices_1y.shift(1))
        monthly_log_returns_5y = np.log(monthly_prices_5y / monthly_prices_5y.shift(1))

        # Calculate mean/std ignoring NaNs (pandas default)
        monthly_log_returns_1y_mean = monthly_log_returns_1y.mean()
        monthly_log_returns_5y_mean = monthly_log_returns_5y.mean()
        monthly_log_returns_1y_std = monthly_log_returns_1y.std()
        monthly_log_returns_5y_std = monthly_log_returns_5y.std()

        yearly_simple_returns_1y_std = log_to_simple_yr_std(monthly_log_returns_1y_mean, monthly_log_returns_1y_std)
        monthly_simple_returns_5y_mean = log_to_simple_monthly_mean(monthly_log_returns_5y_mean, monthly_log_returns_5y_std)
        yearly_simple_returns_5y_std = log_to_simple_yr_std(monthly_log_returns_5y_mean, monthly_log_returns_5y_std)

        best_in_group = {
            'A': {"ticker": None, "monthly_simple_return": -1000},
            'B': {"ticker": None, "monthly_simple_return": -1000},
            'C': {"ticker": None, "monthly_simple_return": -1000},
            'D': {"ticker": None, "monthly_simple_return": -1000},
            'E': {"ticker": None, "monthly_simple_return": -1000},
        }
        
        # Iterate over tickers that have at least some data
        for ticker in monthly_log_returns_5y.columns:
            # Check if we have enough data points for this ticker
            # For 5 years, we expect ~60 months. Let's require at least 48 (4 years).
            if monthly_log_returns_5y[ticker].count() < 48:
                print(f"UNEXPECTED: Skipping {ticker}: Not enough data points for 5 years !!!!!!!!!!!")
                continue
                
            # m_1y = monthly_simple_returns_1y_mean[ticker]
            m_5y = monthly_simple_returns_5y_mean[ticker]
            s_1y = yearly_simple_returns_1y_std[ticker]
            s_5y = yearly_simple_returns_5y_std[ticker]
            
            if pd.isna(m_5y) or pd.isna(s_1y) or pd.isna(s_5y):
                continue

            category = get_category(s_1y, s_5y)
            if m_5y > best_in_group[category]['monthly_simple_return']:
                best_in_group[category]['ticker'] = ticker
                best_in_group[category]['monthly_simple_return'] = m_5y
                best_in_group[category]['yearly_simple_std'] = s_5y

        return best_in_group

    def get_expected_simple_return(ticker, trade_day):
        five_years_ago = trade_day - pd.Timedelta(days=365*5)
        # Next line has an error
        monthly_prices_5y = df_closing.loc[five_years_ago:trade_day, ticker].resample('ME').last().dropna()
        monthly_log_returns_5y = np.log(monthly_prices_5y / monthly_prices_5y.shift(1)).dropna()
        monthly_log_returns_5y_mean = monthly_log_returns_5y.mean()
        monthly_log_returns_5y_std = monthly_log_returns_5y.std()
        monthly_simple_return_5y = log_to_simple_monthly_mean(monthly_log_returns_5y_mean, monthly_log_returns_5y_std)
        return monthly_simple_return_5y

    # Initialize the portfolio
    portfolio = {
        'A': [],
        'B': [],
        'C': [],
        'D': [],
        'E': []
    }
    
    def get_trading_days(df, start_date, end_date):
        """Get the first trading day of each month."""
        trading_days = []
        # Generate month start dates
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        for d in dates:
            # Find first available date in df on or after d
            mask = (df.index >= d)
            available = df.index[mask]
            if not available.empty:
                # Check if it's still in the same month (just to be safe, though likely yes)
                first_day = available[0]
                if first_day.month == d.month and first_day.year == d.year:
                    trading_days.append(first_day)
        return trading_days

    # Generate monthly trading days
    trading_days = get_trading_days(df_closing, '2017-01-01', END_DATE)

    if not trading_days:
        print("No trading days found!")
        return

    today = trading_days[0]
    best_in_group = find_best_in_group(today)

    for category in best_in_group:
        ticker = best_in_group[category]['ticker']
        
        # Handle case where no ticker was found
        if ticker is None:
            print(f"No suitable ticker found for category {category} on {today.date()}")
            d = {
                "ticker": None,
                "amount": 0,
                "value_in": START_CAPITAL, # Assuming cash held
                "value_out": START_CAPITAL,
                "monthly_simple_return": 0,
                "yearly_simple_std": 0,
                "category": category
            }
            portfolio[category].append(d)
            continue

        amount_buy = buy_ticker(ticker, today, START_CAPITAL)
        monthly_simple_return = best_in_group[category]['monthly_simple_return']
        yearly_simple_std = best_in_group[category]['yearly_simple_std']
        d = {
            "ticker": ticker,
            "amount": amount_buy,
            "value_in": START_CAPITAL,
            "value_out": START_CAPITAL,
            "monthly_simple_return": monthly_simple_return,
            "yearly_simple_std": yearly_simple_std,
            "category": category
        }
        portfolio[category].append(d)
    
    # Trading bot stepping through time
    for today in trading_days[1:]:
        last_best_in_group = best_in_group.copy()
        best_in_group = find_best_in_group(today)

        for category in best_in_group:
            old_ticker = last_best_in_group[category]['ticker']
            new_ticker = best_in_group[category]['ticker']
            
            # Case 0: No ticker found previously or currently
            if old_ticker is None and new_ticker is None:
                 d = {
                    "ticker": None,
                    "amount": 0,
                    "value_in": portfolio[category][-1]['value_out'],
                    "value_out": portfolio[category][-1]['value_out'],
                    "monthly_simple_return": 0,
                    "yearly_simple_std": 0,
                    "category": category
                }
                 portfolio[category].append(d)
                 continue
            
            # Case 0.5: No ticker found currently, but we have one from before. 
            # Strategy: Sell everything and hold cash? Or keep holding? 
            # Let's assume we sell and hold cash if no suitable ticker is found.
            if new_ticker is None:
                 # If we had a ticker, sell it.
                 amount_old = portfolio[category][-1]['amount']
                 price_old = df_closing.loc[today, old_ticker]
                 value_out = amount_old * price_old
                 
                 d = {
                    "ticker": None,
                    "amount": 0,
                    "value_in": value_out,
                    "value_out": value_out,
                    "monthly_simple_return": 0,
                    "yearly_simple_std": 0,
                    "category": category
                }
                 print(f"Selling {old_ticker} and holding cash in category {category} on {today.date()}")
                 portfolio[category].append(d)
                 continue

            # Case 0.75: We had no ticker, but now we found one.
            if old_ticker is None:
                 # Buy new ticker with available cash
                 cash_available = portfolio[category][-1]['value_out']
                 amount_new = buy_ticker(new_ticker, today, cash_available)
                 monthly_simple_return = best_in_group[category]['monthly_simple_return']
                 yearly_simple_std = best_in_group[category]['yearly_simple_std']
                 
                 d = {
                    "ticker": new_ticker,
                    "amount": amount_new,
                    "value_in": cash_available,
                    "value_out": cash_available,
                    "monthly_simple_return": monthly_simple_return,
                    "yearly_simple_std": yearly_simple_std,
                    "category": category
                 }
                 print(f"Buying {new_ticker} with cash in category {category} on {today.date()}")
                 portfolio[category].append(d)
                 continue


            # Same ticker as last time, update values
            if old_ticker == new_ticker:
                amount = portfolio[category][-1]['amount']
                price = df_closing.loc[today, old_ticker]
                value_in = amount * price
                value_out = value_in
                monthly_simple_return = best_in_group[category]['monthly_simple_return']
                yearly_simple_std = best_in_group[category]['yearly_simple_std']
                d = {
                    "ticker": old_ticker,
                    "amount": amount,
                    "value_in": value_in,
                    "value_out": value_out,
                    "monthly_simple_return": monthly_simple_return,
                    "yearly_simple_std": yearly_simple_std,
                    "category": category
                }
                print(f"Keeping {old_ticker} in category {category} on {today.date()}")
            else:
                # Expected return if we keep the old ticker
                amount_old = portfolio[category][-1]['amount']
                price_old = df_closing.loc[today, old_ticker]
                value_keep_old = amount_old * price_old
                expect_keep_return = get_expected_simple_return(old_ticker, today)
                expected_value_keep = value_keep_old * (1 + expect_keep_return)
                # Expected return if we buy the new ticker
                money_after_sell = value_keep_old * (1 - REBALANCE_COST)
                expected_return_new = best_in_group[category]['monthly_simple_return']
                expected_value_rebalance = money_after_sell * (1 + expected_return_new)
                # Decide whether to rebalance or not
                if expected_value_rebalance > expected_value_keep:
                    # Rebalance
                    amount_new = buy_ticker(new_ticker, today, money_after_sell)
                    value_in = value_keep_old
                    value_out = money_after_sell
                    yearly_simple_std = best_in_group[category]['yearly_simple_std']
                    d = {
                        "ticker": new_ticker,
                        "amount": amount_new,
                        "value_in": value_in,
                        "value_out": value_out,
                        "monthly_simple_return": expected_return_new,
                        "yearly_simple_std": yearly_simple_std,
                        "category": category
                    }
                    print(f"Rebalancing from {old_ticker} to {new_ticker} in category {category} on {today.date()}")
                else:
                    # Keep old ticker
                    price = df_closing.loc[today, old_ticker]
                    yearly_simple_std = best_in_group[category]['yearly_simple_std']
                    d = {
                        "ticker": old_ticker,
                        "amount": amount_old,
                        "value_in": value_keep_old,
                        "value_out": value_keep_old,
                        "monthly_simple_return": expect_keep_return,
                        "yearly_simple_std": yearly_simple_std,
                        "category": category
                    }
                    print(f"Keeping {old_ticker} in category {category} on {today.date()}")

            portfolio[category].append(d)

    # Print final values
    print("\nFinal portfolio values on", END_DATE.date())
    for category in portfolio:
        final_entry = portfolio[category][-1]
        ticker = final_entry['ticker']
        amount = final_entry['amount']
        
        if ticker is None:
             final_value = final_entry['value_out']
             print(f"Category {category}: No Ticker, Final Value: ${final_value:,.2f}")
        else:
            price = df_closing.loc[END_DATE, ticker]
            final_value = amount * price
            print(f"Category {category}: Ticker {ticker}, Final Value: ${final_value:,.2f}")


def main():
    # Check input parquet exists
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} does not exist. Please run download  or filter tickers first.")
        return

    # Build or load the wide closing‑price DataFrame
    if not os.path.exists(OUTPUT_FILE):
        df = pd.read_parquet(INPUT_FILE)
        print("Pivoting data...")

        # Case 1: MultiIndex from yfinance (Date, Ticker)
        if isinstance(df.index, pd.MultiIndex):
            # Extract Close and unstack tickers
            df_closing = df['Close'].unstack('Ticker')

        else:
            # Case 2: flat table with Date column
            if 'Date' not in df.columns:
                df = df.reset_index()
            df_closing = df.pivot(index='Date', columns='Ticker', values='Close')


        # Restrict to start date
        # df_closing = df_closing.loc['2017-01-02':]

        # Clean data: Replace 0 or negative prices with NaN to avoid log errors
        df_closing = df_closing.where(df_closing > 0)

        # Save to parquet
        df_closing.to_parquet(OUTPUT_FILE)
        print(f"Saved pivoted data to {OUTPUT_FILE}")

    else:
        df_closing = pd.read_parquet(OUTPUT_FILE)
        # Clean data: Replace 0 or negative prices with NaN
        df_closing = df_closing.where(df_closing > 0)
        print(f"Loaded pivoted data from {OUTPUT_FILE}")

    # Run validation tests
    print("\nRunning validation tests...")
    df_demo = df_closing[DEMO_TICKERS].dropna(how="all", axis=0)
    test_log_returns(df_demo)

    # Run portfolio optimization
    optimize_portfolio(df_closing)

if __name__ == "__main__":
    main()
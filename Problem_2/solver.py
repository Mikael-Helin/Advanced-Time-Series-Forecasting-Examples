import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../Data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
VALID_TICKERS_FILE = os.path.join(DATA_DIR, 'valid_daily_tickers.csv')
START_DATE = '2017-01-02'
END_DATE = '2024-12-31'
INITIAL_CAPITAL = 100000.0
FEE = 0.01

CATEGORIES = {
    'A': {'target_vol': 0.025, 'max_vol': 0.03},
    'B': {'target_vol': 0.05, 'max_vol': 0.06},
    'C': {'target_vol': 0.10, 'max_vol': 0.12},
    'D': {'target_vol': 0.20, 'max_vol': 0.24},
    'E': {'target_vol': float('inf'), 'max_vol': float('inf')}
}

def get_first_mondays(start_year, end_year):
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = datetime(year, month, 1)
            while d.weekday() != 0:
                d += timedelta(days=1)
            dates.append(d)
    return dates

def load_all_data(tickers):
    print("Loading data...")
    price_data = {}
    for ticker in tickers:
        file_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(file_path, header=None, skiprows=3)
            if df.empty:
                continue
            df[0] = pd.to_datetime(df[0])
            df.set_index(0, inplace=True)
            price_data[ticker] = pd.to_numeric(df[4], errors='coerce')
        except Exception:
            continue
    prices_df = pd.DataFrame(price_data)
    prices_df.sort_index(inplace=True)
    return prices_df

def calculate_metrics(prices_df, current_date):
    """
    Calculate Volatility (1yr), Expected Return (5yr), and Monthly Correlations (5yr).
    """
    history = prices_df.loc[:current_date]
    if history.empty:
        return None, None, None
        
    # Daily Returns for Vol and Exp Ret
    returns = history.pct_change(fill_method=None)
    
    # 1 Year Volatility (252 days)
    last_year = returns.iloc[-252:]
    if len(last_year) < 200:
        vol = pd.Series(index=returns.columns, dtype=float)
    else:
        vol = last_year.std() * np.sqrt(252)
        
    # 5 Year Expected Return (1260 days)
    last_5yr = returns.iloc[-1260:]
    if len(last_5yr) < 1000:
        exp_ret = pd.Series(index=returns.columns, dtype=float)
    else:
        exp_ret = last_5yr.mean() * 252
        
    # Monthly Correlations (5 years)
    # Resample history to monthly
    # Use 'ME' for Month End
    monthly_prices = history.resample('ME').last()
    
    # Drop NaNs before log to avoid warnings
    monthly_prices = monthly_prices.dropna(how='all') 
    # Actually we need to handle per-column?
    # np.log handles dataframe elementwise.
    # If price <= 0 or NaN, log warns.
    
    # We can just ignore warnings or be safer.
    # Better: dropna.
    
    # Use np.errstate to suppress warnings for log(NaN) or log(<=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        monthly_returns = np.log(monthly_prices / monthly_prices.shift(1))
    
    # Last 60 months
    last_60_months = monthly_returns.iloc[-60:]
    
    return vol, exp_ret, last_60_months

def optimize_pair(t1, t2, vol1, vol2, ret1, ret2, corr, limit):
    """
    Find best weights for pair (t1, t2) to maximize return s.t. vol <= limit.
    """
    # Grid search for weights
    weights = np.linspace(0, 1, 101) # 0 to 1 step 0.01
    
    best_ret = -np.inf
    best_w = None
    
    # Vectorized calculation
    w1 = weights
    w2 = 1 - weights
    
    p_ret = w1 * ret1 + w2 * ret2
    
    # Variance = w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 rho s1 s2
    p_var = (w1**2 * vol1**2) + (w2**2 * vol2**2) + (2 * w1 * w2 * corr * vol1 * vol2)
    p_vol = np.sqrt(p_var)
    
    # Filter valid
    valid = p_vol <= limit
    
    if np.any(valid):
        valid_rets = p_ret[valid]
        valid_weights = w1[valid]
        
        idx = np.argmax(valid_rets)
        best_ret = valid_rets[idx]
        best_w = valid_weights[idx]
        
        return best_ret, best_w
    else:
        return -np.inf, None

def run_solver():
    if not os.path.exists(VALID_TICKERS_FILE):
        print("Valid tickers file not found.")
        return
    valid_tickers = pd.read_csv(VALID_TICKERS_FILE)['Ticker'].tolist()
    
    prices_df = load_all_data(valid_tickers)
    print(f"Loaded data for {len(prices_df.columns)} tickers.")
    
    trading_dates = get_first_mondays(2017, 2024)
    trading_dates = [d for d in trading_dates if d >= datetime(2017, 1, 2) and d <= datetime(2024, 12, 31)]
    
    # State: { 'A': {'cash': ..., 'portfolio': {'T1': w1, 'T2': w2}, 'value': ...} }
    portfolios = {cat: {'cash': INITIAL_CAPITAL, 'portfolio': {}, 'value': INITIAL_CAPITAL} for cat in CATEGORIES}
    
    print(f"Starting simulation on {len(trading_dates)} dates...")
    
    for date in trading_dates:
        date_str = date.strftime('%Y-%m-%d')
        # print(f"Processing {date_str}...")
        
        vol, exp_ret, monthly_returns = calculate_metrics(prices_df, date)
        if vol is None or monthly_returns.empty:
            continue
            
        try:
            current_prices = prices_df.loc[date]
        except KeyError:
            future = prices_df.loc[date:]
            if future.empty: break
            current_prices = future.iloc[0]
            
        # Filter candidates: Top 50 by expected return
        # Must have valid vol and ret
        valid_metrics = pd.DataFrame({'vol': vol, 'ret': exp_ret}).dropna()
        if valid_metrics.empty: continue
        
        top_candidates = valid_metrics.sort_values('ret', ascending=False).head(50).index.tolist()
        
        # Compute Correlation Matrix for Candidates ONLY
        # Ensure candidates are in monthly_returns
        top_candidates = [t for t in top_candidates if t in monthly_returns.columns]
        
        if len(top_candidates) < 2:
            # Need at least 2 for pairs, or 1 for single
            pass
            
        # Subset returns
        subset_returns = monthly_returns[top_candidates]
        if len(subset_returns) < 24:
            # Not enough history for correlation
            corr_matrix = pd.DataFrame(index=top_candidates, columns=top_candidates).fillna(0)
            # Or identity?
            corr_matrix = pd.DataFrame(np.eye(len(top_candidates)), index=top_candidates, columns=top_candidates)
        else:
            corr_matrix = subset_returns.corr()
            
        for cat, params in CATEGORIES.items():
            state = portfolios[cat]
            current_port = state['portfolio']
            
            # Update Value
            port_val = state['cash']
            for t, w in current_port.items():
                if t in current_prices:
                    # Value = shares * price. 
                    # But we stored weights? No, we need to store shares to track value correctly.
                    # Let's change state to store shares.
                    pass
            
            # Correction: Store shares, not weights.
            # 'portfolio': {'T1': shares, 'T2': shares}
            current_val = state['cash']
            for t, shares in current_port.items():
                if t in current_prices:
                    current_val += shares * current_prices[t]
            state['value'] = current_val
            
            # Check Rebalance
            must_rebalance = False
            
            # Calculate current portfolio metrics
            cur_ret = 0
            cur_vol = 0
            
            if current_port:
                # Re-calculate weights based on current value
                temp_weights = {}
                total_pos_val = 0
                for t, s in current_port.items():
                    val = s * current_prices.get(t, 0)
                    total_pos_val += val
                    
                if total_pos_val > 0:
                    # Calculate portfolio vol
                    # Need weights
                    p_tickers = list(current_port.keys())
                    if all(t in valid_metrics.index for t in p_tickers):
                        ws = np.array([current_port[t] * current_prices[t] / total_pos_val for t in p_tickers])
                        vs = np.array([valid_metrics.loc[t, 'vol'] for t in p_tickers])
                        rs = np.array([valid_metrics.loc[t, 'ret'] for t in p_tickers])
                        
                        cur_ret = np.sum(ws * rs)
                        
                        # Vol
                        if len(p_tickers) == 1:
                            cur_vol = vs[0]
                        elif len(p_tickers) == 2:
                            t1, t2 = p_tickers
                            if t1 in corr_matrix.index and t2 in corr_matrix.index:
                                rho = corr_matrix.loc[t1, t2]
                                var = (ws[0]*vs[0])**2 + (ws[1]*vs[1])**2 + 2*ws[0]*ws[1]*rho*vs[0]*vs[1]
                                cur_vol = np.sqrt(var)
                            else:
                                cur_vol = 1.0 # Penalty
                        else:
                            # Should not happen
                            cur_vol = 1.0
                            
                        if cur_vol > params['max_vol']:
                            must_rebalance = True
                    else:
                        must_rebalance = True
                else:
                    must_rebalance = True # All cash or invalid
            else:
                must_rebalance = True # All cash
                
            # Find Best Portfolio
            best_p_ret = -np.inf
            best_p_weights = {} # {Ticker: weight}
            
            # Iterate pairs
            # Include single stocks (pair with itself or weight=1)
            # We iterate i from 0 to N, j from i to N
            
            n_cands = len(top_candidates)
            for i in range(n_cands):
                t1 = top_candidates[i]
                v1 = valid_metrics.loc[t1, 'vol']
                r1 = valid_metrics.loc[t1, 'ret']
                
                # Single stock check
                if v1 <= params['target_vol']:
                    if r1 > best_p_ret:
                        best_p_ret = r1
                        best_p_weights = {t1: 1.0}
                
                for j in range(i + 1, n_cands):
                    t2 = top_candidates[j]
                    v2 = valid_metrics.loc[t2, 'vol']
                    r2 = valid_metrics.loc[t2, 'ret']
                    
                    rho = corr_matrix.loc[t1, t2]
                    
                    # Optimize
                    pret, w1 = optimize_pair(t1, t2, v1, v2, r1, r2, rho, params['target_vol'])
                    
                    if pret > best_p_ret:
                        best_p_ret = pret
                        best_p_weights = {t1: w1, t2: 1.0 - w1}
            
            # Decide Switch
            do_switch = False
            if must_rebalance:
                do_switch = True
            elif best_p_ret > (cur_ret / 0.99) and best_p_ret > 0:
                do_switch = True
                
            if do_switch and best_p_weights:
                # Execute
                # Sell all
                equity = state['value']
                fee = equity * FEE
                equity -= fee
                
                if equity <= 0:
                    state['portfolio'] = {}
                    state['cash'] = 0
                    continue
                    
                # Buy new
                new_port = {}
                for t, w in best_p_weights.items():
                    if w > 0.001: # Ignore tiny weights
                        price = current_prices.get(t)
                        if price and not pd.isna(price):
                            shares = (equity * w) / price
                            new_port[t] = shares
                            
                state['portfolio'] = new_port
                state['cash'] = 0 # Fully invested (approx)
                
                # Recalculate value to be precise?
                # Value is equity.
                state['value'] = equity
                
    print("\nResults at 2024-12-31:")
    for cat in CATEGORIES:
        val = portfolios[cat]['value']
        print(f"    {cat}: {val:,.2f}")

if __name__ == "__main__":
    run_solver()

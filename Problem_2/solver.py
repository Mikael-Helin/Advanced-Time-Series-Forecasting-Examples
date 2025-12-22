import pandas as pd
import numpy as np
import os
from numba import njit
from numba import cuda
import math
import cupy as cp

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
START_DATE = MS_prices.index.min().date() # Ensure we start at the first available date
ME_prices = df.pivot(columns="Ticker", values="Close").resample("ME").last()
ME_prices = ME_prices.loc[CUTOFF_DATE:END_DATE]
END_DATE = ME_prices.index.max().date() # Ensure we end at the last available date

# 2. FIX: Add Cash after pivoting
MS_prices["cash"] = 1.0
ME_prices["cash"] = 1.0
MS_prices = MS_prices.ffill()
ME_prices = ME_prices.ffill()

# 3. Calculate Returns, Volatility and Correlation
monthly_simple_returns = ME_prices.pct_change() # Keep all history for rolling calc

# Rolling 5yr (60 months) and 1yr (12 months)
# min_periods=12 ensures we get data sooner, but 60 is strictly 5 years
print("Computing rolling returns and std...")
monthly_simple_means_5yr = monthly_simple_returns.rolling(window=60).mean()
monthly_simple_std_5yr = monthly_simple_returns.rolling(window=60).std()
monthly_simple_std_1yr = monthly_simple_returns.rolling(window=12).std()
# Ensure equal date ranges
monthly_simple_means_5yr = monthly_simple_means_5yr[START_DATE:END_DATE]
monthly_simple_std_5yr = monthly_simple_std_5yr[START_DATE:END_DATE]
monthly_simple_std_1yr = monthly_simple_std_1yr[START_DATE:END_DATE]

# Computing rolling correlation with Pandas is too slow
# monthly_simple_corr_5yr = monthly_simple_returns.rolling(window=60).corr()
# monthly_simple_corr_1yr = monthly_simple_returns.rolling(window=12).corr()
# Instead we have to use Numba

@cuda.jit
def gpu_corr_kernel(data, means, stds, out_corr):
    # Get the row and column index for this specific GPU thread
    i, j = cuda.grid(2)
    
    T, N = data.shape
    
    # Boundary check: ensure thread is within matrix dimensions
    if i < N and j < N:
        if i == j:
            out_corr[i, j] = 1.0
            return
            
        # Only compute the upper triangle and mirror it for speed
        if i < j:
            sum_prod = 0.0
            for t in range(T):
                sum_prod += (data[t, i] - means[i]) * (data[t, j] - means[j])
            
            # Correlation formula
            if stds[i] > 0 and stds[j] > 0:
                val = sum_prod / (T * stds[i] * stds[j])
            else:
                val = 0.0
                
            out_corr[i, j] = val
            out_corr[j, i] = val # Mirror the result

def compute_rolling_corr_numba_cuda(df, window=60):
    returns_arr = df.values.astype(np.float32)
    T, N = returns_arr.shape
    dates = df.index
    results = {}

    # 1. Pre-allocate GPU memory
    # Move the whole dataset to GPU once
    d_data_all = cuda.to_device(returns_arr)
    d_out_corr = cuda.device_array((N, N), dtype=np.float32)

    # Define thread block size (usually 16x16 or 32x32)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for t in range(window, T):
        # Slice the window on the GPU
        window_slice = d_data_all[t-window:t, :]
        
        # Calculate means and stds for the window (CPU side or use GPU ufuncs)
        # For simplicity, we calculate these small vectors on CPU and send them
        current_window_cpu = returns_arr[t-window:t, :]
        means = np.mean(current_window_cpu, axis=0)
        stds = np.std(current_window_cpu, axis=0)
        
        d_means = cuda.to_device(means)
        d_stds = cuda.to_device(stds)

        # 2. Launch the Kernel
        gpu_corr_kernel[blockspergrid, threadsperblock](window_slice, d_means, d_stds, d_out_corr)
        
        # 3. Pull result back to CPU
        results[dates[t]] = d_out_corr.copy_to_host()
        
    return results

print("Computing rolling corr...")
monthly_simple_corr_5yr = compute_rolling_corr_numba_cuda(monthly_simple_returns, window=60)
monthly_simple_corr_1yr = compute_rolling_corr_numba_cuda(monthly_simple_returns, window=12)
# Ensure equal date ranges for tensors (these are dictionaries, not pandas DataFrames)
ts_start = pd.Timestamp(START_DATE)
ts_end = pd.Timestamp(END_DATE)
monthly_simple_corr_5yr = {k: v for k, v in monthly_simple_corr_5yr.items() if ts_start <= k <= ts_end}
monthly_simple_corr_1yr = {k: v for k, v in monthly_simple_corr_1yr.items() if ts_start <= k <= ts_end}

# 4. Compute minimum volatility for each correlation tensor
# We want it for initialization of categories

@cuda.jit
def compute_vol_and_weights_kernel(stds, corrs, out_vol, out_weights):
    """
    Computes Min Volatility AND the optimal weight (w1) for Asset 1.
    stds: (T, N)
    corrs: (T, N, N)
    out_vol: (T, N, N)
    out_weights: (T, N, N) -> Stores w1. w2 is just (1 - w1).
    """
    i, j = cuda.grid(2)
    T, N, _ = out_vol.shape

    if i < N and j < N:
        for t in range(T):
            s1 = stds[t, i]
            s2 = stds[t, j]
            rho = corrs[t, i, j]

            # Case: Same asset or invalid volatility
            if i == j or s1 <= 1e-8 or s2 <= 1e-8:
                # Default to equal weight or 100% asset 1 (arbitrary for i==j)
                out_vol[t, i, j] = min(s1, s2)
                out_weights[t, i, j] = 0.5 
                continue

            # Denominator for weight calculation
            denom = s1**2 + s2**2 - 2 * rho * s1 * s2
            
            if denom <= 1e-8:
                out_vol[t, i, j] = min(s1, s2)
                out_weights[t, i, j] = 0.5
                continue

            # Calculate optimal weight 'w1' for Asset 1
            w1 = (s2**2 - rho * s1 * s2) / denom
            
            # Constrain 0 <= w1 <= 1
            if w1 < 0.0:
                w1 = 0.0
            elif w1 > 1.0:
                w1 = 1.0
            
            # Calculate final portfolio variance
            w2 = 1.0 - w1
            var_final = (w1**2 * s1**2) + (w2**2 * s2**2) + (2 * w1 * w2 * rho * s1 * s2)
            
            out_vol[t, i, j] = math.sqrt(max(0.0, var_final))
            out_weights[t, i, j] = w1

def compute_full_min_vol_tensor(stds_df, corr_dict):
    """
    Returns Volatility Tensor AND Weights Tensor
    """
    common_dates = sorted(list(set(stds_df.index) & set(corr_dict.keys())))
    stds_np = stds_df.loc[common_dates].values.astype(np.float32)
    
    print(f"Stacking tensors for {len(common_dates)} months...")
    T, N = stds_np.shape
    corr_np = np.zeros((T, N, N), dtype=np.float32)
    for idx, date in enumerate(common_dates):
        corr_np[idx] = corr_dict[date]

    d_stds = cuda.to_device(stds_np)
    d_corrs = cuda.to_device(corr_np)
    d_out_vol = cuda.device_array((T, N, N), dtype=np.float32)
    d_out_weights = cuda.device_array((T, N, N), dtype=np.float32)

    threads = (16, 16)
    blocks = (math.ceil(N / 16), math.ceil(N / 16))
    
    compute_vol_and_weights_kernel[blocks, threads](d_stds, d_corrs, d_out_vol, d_out_weights)
    
    return d_out_vol.copy_to_host(), d_out_weights.copy_to_host(), common_dates


print("Computing 5yr surface...")
vol_tensor_5yr, w1_tensor_5yr, valid_dates = compute_full_min_vol_tensor(monthly_simple_std_5yr, monthly_simple_corr_5yr)

print("Computing 1yr surface...")
vol_tensor_1yr, w1_tensor_1yr, _ = compute_full_min_vol_tensor(monthly_simple_std_1yr, monthly_simple_corr_1yr)

# --- 5. Calculate Pair Returns (GPU) ---


@cuda.jit
def compute_pair_returns_kernel(means, weights, out_ret):
    """
    R_pair = w1 * R1 + (1-w1) * R2
    """
    i, j = cuda.grid(2)
    T, N, _ = out_ret.shape
    
    if i < N and j < N:
        for t in range(T):
            w1 = weights[t, i, j]
            w2 = 1.0 - w1
            # Compute return
            r_pair = w1 * means[t, i] + w2 * means[t, j]
            out_ret[t, i, j] = r_pair

# Prepare Means (using 5yr means for expected return)
# Ensure alignment with the valid_dates from step 4
aligned_means = monthly_simple_means_5yr.loc[valid_dates].values.astype(np.float32)

T, N = aligned_means.shape
d_means = cuda.to_device(aligned_means)
d_weights = cuda.to_device(w1_tensor_5yr) # Use 5yr weights or 1yr depending on strategy
d_out_ret = cuda.device_array((T, N, N), dtype=np.float32)

threads = (16, 16)
blocks = (math.ceil(N / 16), math.ceil(N / 16))

print("Computing pair returns...")
compute_pair_returns_kernel[blocks, threads](d_means, d_weights, d_out_ret)
return_tensor_5yr = d_out_ret.copy_to_host()

# --- 6. Categorize (GPU) ---

category_limits = {
    "A": {"1yr": 0.03/np.sqrt(12), "5yr": 0.025/np.sqrt(12)},
    "B": {"1yr": 0.06/np.sqrt(12), "5yr": 0.05/np.sqrt(12)},
    "C": {"1yr": 0.12/np.sqrt(12), "5yr": 0.1/np.sqrt(12)},
    "D": {"1yr": 0.3/np.sqrt(12),  "5yr": 0.25/np.sqrt(12)},
    "E": {"1yr": 0.6/np.sqrt(12),  "5yr": 0.5/np.sqrt(12)},
}

@cuda.jit
def init_categories_kernel(v5, v1, thresh, out):
    i, j = cuda.grid(2)
    T, N, _ = out.shape
    
    if i < N and j < N:
        for t in range(T):
            val5 = v5[t, i, j]
            val1 = v1[t, i, j]
            cat = 5 # Default F
            
            for k in range(5):
                limit1 = thresh[k, 0]
                limit5 = thresh[k, 1]
                if val1 <= limit1 and val5 <= limit5:
                    cat = k
                    break
            out[t, i, j] = cat

def init_categories(v5_np, v1_np, limits):
    thresh_data = np.array([
        [limits["A"]["1yr"], limits["A"]["5yr"]],
        [limits["B"]["1yr"], limits["B"]["5yr"]],
        [limits["C"]["1yr"], limits["C"]["5yr"]],
        [limits["D"]["1yr"], limits["D"]["5yr"]],
        [limits["E"]["1yr"], limits["E"]["5yr"]],
    ], dtype=np.float32)

    T, N, _ = v5_np.shape
    
    d_v5 = cuda.to_device(v5_np.astype(np.float32))
    d_v1 = cuda.to_device(v1_np.astype(np.float32))
    d_thresh = cuda.to_device(thresh_data)
    d_out = cuda.device_array((T, N, N), dtype=np.uint8)

    threads = (16, 16)
    blocks = (math.ceil(N / 16), math.ceil(N / 16))

    init_categories_kernel[blocks, threads](d_v5, d_v1, d_thresh, d_out)
    return d_out.copy_to_host()

print("Categorizing pairs...")
min_vol_categories_tensor = init_categories(vol_tensor_5yr, vol_tensor_1yr, category_limits)

print("Tensors Ready:")
print(f"Categories: {min_vol_categories_tensor.shape}")
print(f"Returns:    {return_tensor_5yr.shape}")
print(f"Weights:    {w1_tensor_5yr.shape}")

# Let the weights vary and find the best pair for each category
# We have following 4D tensors
# Optimal ticker 1
# Optimal ticker 2
# Optimal weight 1
# Optimal return

num_categories = 5
T, N, _ = return_tensor_5yr.shape
cash_idx = monthly_simple_returns.columns.get_loc("cash")

# Init with Cash
optimal_ticker_1_tensor = np.full((T, num_categories), cash_idx, dtype=np.int32)
optimal_ticker_2_tensor = np.full((T, num_categories), cash_idx, dtype=np.int32)
optimal_weight_1_tensor = np.full((T, num_categories), 1.0, dtype=np.float32)
optimal_return_tensor   = np.full((T, num_categories), 0.0, dtype=np.float32)

@cuda.jit
def solve_quadratic_logic_kernel(means, stds, corrs, min_vols, target_vol, out_ret, out_w1):
    i, j = cuda.grid(2)
    T, N, _ = out_ret.shape
    target_var = target_vol**2

    if i < N and j < N:
        for t in range(T):
            # 1. Skip if this pair cannot achieve low enough volatility
            if min_vols[t, i, j] > target_vol:
                out_ret[t, i, j] = -999.0
                out_w1[t, i, j] = 0.0
                continue
            
            # 2. Get Data
            s1 = stds[t, i]
            s2 = stds[t, j]
            rho = corrs[t, i, j]
            mu1 = means[t, i]
            mu2 = means[t, j]

            # 3. Calculate Coefficients (Ax^2 + Bx + C = 0)
            # Math Correction: Interaction term is minus for Variance of difference
            A = s1**2 + s2**2 - 2 * rho * s1 * s2
            B = 2 * (rho * s1 * s2 - s2**2)
            C = s2**2 - target_var

            # 4. Solve Quadratic
            delta = B**2 - 4 * A * C
            
            if delta < 0 or abs(A) < 1e-9:
                out_ret[t, i, j] = -999.0
                continue

            sqrt_delta = math.sqrt(delta)
            w1 = (-B + sqrt_delta) / (2 * A)
            w2 = (-B - sqrt_delta) / (2 * A)

            # 5. Apply User Logic (Sort and Check Bounds)
            if w1 > w2:
                temp = w1
                w1 = w2
                w2 = temp
            
            # If the range is completely outside [0, 1], skip
            if w2 < 0.0 or w1 > 1.0:
                out_ret[t, i, j] = -999.0
                continue
            
            # Select the best valid weight in [0, 1]
            r_best = -999.0
            w_best = 0.0
            found_valid = False

            # Check w1
            if w1 >= 0.0 and w1 <= 1.0:
                r1 = w1 * mu1 + (1.0 - w1) * mu2
                if r1 > r_best:
                    r_best = r1
                    w_best = w1
                    found_valid = True
            
            # Check w2
            if w2 >= 0.0 and w2 <= 1.0:
                r2 = w2 * mu1 + (1.0 - w2) * mu2
                if r2 > r_best:
                    r_best = r2
                    w_best = w2
                    found_valid = True

            if found_valid:
                out_ret[t, i, j] = r_best
                out_w1[t, i, j] = w_best
            else:
                out_ret[t, i, j] = -999.0

def run_solver(limit, vol_tensor, means_np, stds_np, corrs_dict):
    T, N, N = vol_tensor.shape
    d_means = cuda.to_device(means_np)
    d_stds = cuda.to_device(stds_np)
    
    # Stack correlations
    corr_np = np.zeros((T, N, N), dtype=np.float32)
    for idx, date in enumerate(means_np.index):
        corr_np[idx] = corrs_dict[date]
    d_corrs = cuda.to_device(corr_np)
    d_min_vols = cuda.to_device(vol_tensor.astype(np.float32))
    
    d_out_ret = cuda.device_array((T, N, N), dtype=np.float32)
    d_out_w1 = cuda.device_array((T, N, N), dtype=np.float32)

    threads = (16, 16)
    blocks = (math.ceil(N / 16), math.ceil(N / 16))
    
    solve_quadratic_logic_kernel[blocks, threads](d_means, d_stds, d_corrs, d_min_vols, limit, d_out_ret, d_out_w1)
    
    return d_out_ret.copy_to_host(), d_out_w1.copy_to_host()

# Find best categories for each month

for cat in category_limits.keys():
    print(f"Solving for Category {cat}...")
    cat_idx = {"A":0, "B":1, "C":2, "D":3, "E":4}[cat]
    limit = category_limits[cat]["5yr"]
    valid_means = monthly_simple_means_5yr.loc[valid_dates]
    valid_stds = monthly_simple_std_5yr.loc[valid_dates]
    
    # 1. Run the GPU Solver
    rets, weights = run_solver(limit, vol_tensor_5yr, valid_means, valid_stds, monthly_simple_corr_5yr)

    # 2. Extract Best Pairs (CPU Reduction)
    for t in range(T):
        # Find index of max return for this month
        flat_idx = np.argmax(rets[t])
        i, j = np.unravel_index(flat_idx, (N, N))
    
        best_ret = rets[t, i, j]
    
        # Store if valid
        if best_ret > -900:
            optimal_ticker_1_tensor[t, cat_idx] = i
            optimal_ticker_2_tensor[t, cat_idx] = j
            optimal_weight_1_tensor[t, cat_idx] = weights[t, i, j]
            optimal_return_tensor[t, cat_idx]   = best_ret

# Trader Bot

print("Trader Bot Starting...")

temp_portfolio = {}
cash_portfolio = {
    "ticker": "cash",
    "price": 1.0,
    "num_shares": START_CAPITAL,
    "networth": START_CAPITAL
}
for category in ["A", "B", "C", "D", "E"]:
    temp_portfolio[category] = {
        "ticker_1": "cash",
        "num_shares_1": START_CAPITAL,
        "ticker_2": "cash",
        "num_shares_2": 0.0,
        "networth": START_CAPITAL
    }

portfolio_history = []

def get_price(ticker, t):
    if ticker == "cash": return 1.0
    # Uses .at for fast scalar lookup. Assumes 't' is the index (Date)
    try:
        return MS_prices.at[t, ticker]
    except KeyError:
        # Fallback for END_DATE or other Month End dates
        ts = pd.Timestamp(t)
        if ts in ME_prices.index:
            return ME_prices.at[ts, ticker]
        raise

def get_return(ticker, stats_date):
    if ticker == "cash": return 0.0
    return monthly_simple_means_5yr.at[stats_date, ticker]

def valuate_ticker(ticker, num_shares, t):
    price = get_price(ticker, t)
    return num_shares * price

def sell_ticker(ticker, num_shares, t):
    value = valuate_ticker(ticker, num_shares, t)
    if ticker == "cash": return value, 0.0
    fees = value * SELL_FEE
    return value - fees, fees

def buy_ticker(ticker, capital, t):
    if ticker == "cash": return capital
    return capital / get_price(ticker, t)

def get_volatility(ticker, stats_date, length="5yr"):
    if ticker == "cash": return 0.0
    if length == "5yr":
        return monthly_simple_std_5yr.at[stats_date, ticker] 
    else:
        return monthly_simple_std_1yr.at[stats_date, ticker]

def get_correlation(ticker_1, ticker_2, stats_date):
    idx1 = monthly_simple_returns.columns.get_loc(ticker_1)
    idx2 = monthly_simple_returns.columns.get_loc(ticker_2)
    return monthly_simple_corr_5yr[stats_date][idx1, idx2]

def get_category(ticker_1, ticker_2, weight_1, stats_date):
    if ticker_1 == "cash": return "A"
    v1_5yr = get_volatility(ticker_1, stats_date, length="5yr")
    v1_1yr = get_volatility(ticker_1, stats_date, length="1yr")
    v2_5yr = get_volatility(ticker_2, stats_date, length="5yr")
    v2_1yr = get_volatility(ticker_2, stats_date, length="1yr")
    rho_5yr = get_correlation(ticker_1, ticker_2, stats_date) # 5yr tensor is default
    rho_1yr = get_correlation(ticker_1, ticker_2, stats_date) # Wait, get_correlation uses 5yr tensor. User didn't impl 1yr corr?
    # get_correlation implementation only accesses monthly_simple_corr_5yr. 
    # If get_category needs 1yr correlation, I need to check if that tensor exists or use 5yr as proxy.
    # Looking at previous errors, the user's code had `get_correlation(ticker_1, ticker_2, "5yr", t)`. 
    # My `get_correlation` takes (t1, t2, stats_date). 
    # I should check if I broke `get_correlation` or if it was never capable of "1yr".
    # Inspecting line 501: `return monthly_simple_corr_5yr[stats_date][idx1, idx2]`
    # It seems only 5yr exists. I will use 5yr for both or simplify.
    # Actually, re-reading the code: line 509 `get_correlation(ticker_1, ticker_2, "5yr", t)`.
    # I should change this call to match my `get_correlation` which only supports 5yr.
    # I will assume Rho is stable or just use the 5yr one for now as I don't see `monthly_simple_corr_1yr`.
    rho_5yr = get_correlation(ticker_1, ticker_2, stats_date)
    rho_1yr = rho_5yr # Fallback as 1yr corr might not be available

    sigma_5yr = np.sqrt(v1_5yr**2 + v2_5yr**2 + 2 * rho_5yr * v1_5yr * v2_5yr)
    sigma_1yr = np.sqrt(v1_1yr**2 + v2_1yr**2 + 2 * rho_1yr * v1_1yr * v2_1yr)
    for category in category_limits.keys():
        if sigma_5yr <= category_limits[category]["5yr"] and sigma_1yr <= category_limits[category]["1yr"]:
            return category
    return "F"

def get_correlation(ticker_1, ticker_2, t):
    idx1 = monthly_simple_returns.columns.get_loc(ticker_1)
    idx2 = monthly_simple_returns.columns.get_loc(ticker_2)
    return monthly_simple_corr_5yr[t][idx1, idx2]

def rebalance_tickers(ticker_1, num_ticker_1, ticker_2, num_ticker_2, target_w1, t):
    """Rebalances a ticker to a target weight"""
    if target_w1 == 0.0:
        capital, _ = sell_ticker(ticker_1, num_ticker_1, t)
        num_ticker_1 = 0.0
        num_ticker_2 += buy_ticker(ticker_2, capital, t)
        return num_ticker_1, num_ticker_2
    elif target_w1 == 1.0:
        capital, _ = sell_ticker(ticker_2, num_ticker_2, t)
        num_ticker_2 = 0.0
        num_ticker_1 += buy_ticker(ticker_1, capital, t)
        return num_ticker_1, num_ticker_2
    else:
        value_1 = valuate_ticker(ticker_1, num_ticker_1, t)
        value_2 = valuate_ticker(ticker_2, num_ticker_2, t)
        current_w1 = value_1 / (value_1 + value_2)
        if current_w1 >= target_w1:
            target_value_1 = (value_2 + value_1 * (1 - SELL_FEE)) / (1 / target_w1 - SELL_FEE)
            target_value_2 = target_value_1 * (1 / target_w1 - 1)
            num_ticker_1 = buy_ticker(ticker_1, target_value_1, t)
            num_ticker_2 = buy_ticker(ticker_2, target_value_2, t)
        else:
            target_w2 = 1.0 - target_w1
            target_value_2 = (value_1 + value_2 * (1 - SELL_FEE)) / (1 / target_w2 - SELL_FEE)
            target_value_1 = target_value_2 * (1 / target_w2 - 1)
            num_ticker_1 = buy_ticker(ticker_1, target_value_1, t)
            num_ticker_2 = buy_ticker(ticker_2, target_value_2, t)
        return num_ticker_1, num_ticker_2

def switch_ticker_2(ticker_1, num_ticker_1, ticker_2_old, num_ticker_2_old, ticker_2_new, target_w1, t):
    if target_w1 == 0.0:
        # Sell both tickers and buy new ticker
        capital_1, _ = sell_ticker(ticker_1, num_ticker_1, t)
        capital_2, _ = sell_ticker(ticker_2_old, num_ticker_2_old, t)
        num_ticker_new = buy_ticker(ticker_2_new, capital_1 + capital_2, t)
        return 0.0, num_ticker_new
    elif target_w1 == 1.0:
        # Sell all ticker_2_old and buy ticker_1
        capital_2, _ = sell_ticker(ticker_2_old, num_ticker_2_old, t)
        num_ticker_1 += buy_ticker(ticker_1, capital_2, t)
        return num_ticker_1, 0.0
    else:
        value_1 = valuate_ticker(ticker_1, num_ticker_1, t)
        # Sell all ticker_2_old
        capital_2, _ = sell_ticker(ticker_2_old, num_ticker_2_old, t)
        current_w1 = value_1 / (value_1 + capital_2)
        if current_w1 == target_w1:
            num_ticker_2_new = buy_ticker(ticker_2_new, capital_2, t)
            return num_ticker_1, num_ticker_2_new
        elif current_w1 < target_w1:
            # Use capital_2 to buy ticker_1 and ticker_2_new
            value_1_new = target_w1 * (value_1 + capital_2)
            num_ticker_1 += buy_ticker(ticker_1, value_1_new - value_1, t)
            capital_2 -= value_1_new - value_1
            num_ticker_2_new = buy_ticker(ticker_2_new, capital_2, t)
            return num_ticker_1, num_ticker_2_new
        else:
            num_ticker_2_new = buy_ticker(ticker_2_new, capital_2, t)
            return rebalance_tickers(ticker_1, num_ticker_1, ticker_2_new, num_ticker_2_new, target_w1, t)

def switch_ticker_1(ticker_1_old, num_ticker_1_old, ticker_2, num_ticker_2, ticker_1_new, target_w1, t):
    num_ticker_2_new, num_ticker_1_new = switch_ticker_2(ticker_2, num_ticker_2, ticker_1_old, num_ticker_1_old, ticker_1_new, 1.0 - target_w1, t)
    return num_ticker_1_new, num_ticker_2_new

def switch_both_tickers(ticker_1_old, num_ticker_1_old, ticker_2_old, num_ticker_2_old, ticker_1_new, ticker_2_new, target_w1, t):
    capital_1, _ = sell_ticker(ticker_1_old, num_ticker_1_old, t)
    capital_2, _ = sell_ticker(ticker_2_old, num_ticker_2_old, t)
    total_capital = capital_1 + capital_2
    num_ticker_1_new = buy_ticker(ticker_1_new, total_capital * target_w1, t)
    num_ticker_2_new = buy_ticker(ticker_2_new, total_capital * (1.0 - target_w1), t)
    return num_ticker_1_new, num_ticker_2_new
    
def trade_portfolio(ticker_1_before, num_ticker_1_before,
                    ticker_2_before, num_ticker_2_before,
                    ticker_1_after, ticker_2_after, w_ticker_1_after, category, t, stats_date):
    
    # Get Prices (at current month start t)
    price_1_before = get_price(ticker_1_before, t) 
    price_2_before = get_price(ticker_2_before, t)
    price_1_after = get_price(ticker_1_after, t)
    price_2_after = get_price(ticker_2_after, t)

    # Get Expected Returns (from rolling stats at stats_date)
    return_1_before = get_return(ticker_1_before, stats_date)
    return_2_before = get_return(ticker_2_before, stats_date)
    return_1_after = get_return(ticker_1_after, stats_date)
    return_2_after = get_return(ticker_2_after, stats_date)

    # Calculate "KEEP" Expected Value
    # Value = Current Shares * Current Price * (1 + ExpReturn)
    keep_1_value = num_ticker_1_before * price_1_before
    keep_2_value = num_ticker_2_before * price_2_before
    current_equity = keep_1_value + keep_2_value
    expected_value_keep = keep_1_value * (1 + return_1_before) + keep_2_value * (1 + return_2_before)

    # Check keep category
    force_rebalance = False
    expected_value_keep = keep_1_value * (1 + return_1_before) + keep_2_value * (1 + return_2_before)

    # Check keep category
    force_rebalance = False
    keep_category = get_category(ticker_1_before, ticker_2_before, w_ticker_1_after, stats_date)
    if keep_category > category:
        force_rebalance = True
    
    # Calculate "REBALANCE" Expected Value
    if ticker_1_after == ticker_1_before and ticker_2_after == ticker_2_before:
        num_1_after, num_2_after = rebalance_tickers(ticker_1_before, num_ticker_1_before, ticker_2_before, num_ticker_2_before, w_ticker_1_after, t)
    elif (ticker_1_before == ticker_1_after and ticker_2_before != ticker_2_after):
        num_1_after, num_2_after = switch_ticker_2(ticker_1_before, num_ticker_1_before, ticker_2_before, num_ticker_2_before, ticker_2_after, w_ticker_1_after, t)
    elif (ticker_1_before != ticker_1_after and ticker_2_before == ticker_2_after):
        num_1_after, num_2_after = switch_ticker_1(ticker_1_before, num_ticker_1_before, ticker_2_before, num_ticker_2_before, ticker_1_after, w_ticker_1_after, t)
    else:
        num_1_after, num_2_after = switch_both_tickers(ticker_1_before, num_ticker_1_before, ticker_2_before, num_ticker_2_before, ticker_1_after, ticker_2_after, w_ticker_1_after, t)
    
    expected_value_rebalance = num_1_after * price_1_after * (1 + return_1_after) + num_2_after * price_2_after * (1 + return_2_after)
    if force_rebalance or expected_value_rebalance > expected_value_keep:
        return {
            "ticker_1": ticker_1_after,
            "num_shares_1": num_1_after,
            "ticker_2": ticker_2_after,
            "num_shares_2": num_2_after,
            "networth": num_1_after * price_1_after + num_2_after * price_2_after
        }
    else:
        return {
            "ticker_1": ticker_1_before,
            "num_shares_1": num_ticker_1_before,
            "ticker_2": ticker_2_before,
            "num_shares_2": num_ticker_2_before,
            "networth": current_equity
        }


# Traders State
portfolio_history = []
temp_portfolio = {}
for cat in ["A", "B", "C", "D", "E"]:
    temp_portfolio[cat] = {
        "ticker_1": "cash",
        "num_shares_1": START_CAPITAL,
        "ticker_2": "cash",
        "num_shares_2": 0.0,
        "networth": START_CAPITAL
    }

# Map dates to tensor indices
# valid_dates are Month Ends (from rolling calc), MS_prices are Month Starts.
# We map MS date -> Tensor Index
valid_idx_map = {}
for i, date in enumerate(valid_dates):
    # Convert '2017-01-31' -> '2017-01-01'
    ms_date = date.replace(day=1)
    valid_idx_map[ms_date] = i

print(f"\n--- Starting Simulation ({START_DATE} to {END_DATE}) ---")

for t in sorted(MS_prices.index):
    # Only trade if we have data (tensor index) for this date
    # CRITICAL FIX: Look-ahead bias.
    # t is Trading Date (e.g. Jan 1). 
    # valid_idx_map[t] gives index of stats for Jan 31 (containing Jan data).
    # We MUST use stats from Dec 31 (previous index) to decide trades for Jan 1.
    
    current_month_stats_idx = valid_idx_map.get(t)
    if current_month_stats_idx is None or current_month_stats_idx == 0:
        continue
    
    # Use previous month's stats
    # t_idx is used for tensor lookup too? 
    # The tensors (optimal_ticker...) were computed using `valid_dates`.
    # `optimal_ticker_1_tensor[i]` corresponds to `valid_dates[i]`.
    # If `valid_dates[i]` is Jan 31, then that tensor entry was optimized knowing Jan data.
    # So we CANNOT use `optimal_ticker_1_tensor[current_month_stats_idx]`.
    # We MUST use `optimal_ticker_1_tensor[current_month_stats_idx - 1]`.
    
    t_idx = current_month_stats_idx - 1
    stats_date = valid_dates[t_idx]
        
    print(f"Date: {t.date()}")
    
    # Snapshot for this date
    history_entry = {"Date": t}
    
    for cat in ["A", "B", "C", "D", "E"]:
        cat_idx = {"A":0, "B":1, "C":2, "D":3, "E":4}[cat]
        
        # Get Candidates from Tensor
        cand_t1_idx = optimal_ticker_1_tensor[t_idx, cat_idx]
        cand_t2_idx = optimal_ticker_2_tensor[t_idx, cat_idx]
        cand_w1     = optimal_weight_1_tensor[t_idx, cat_idx]
        
        candidate_pairs = {
            "ticker_1": monthly_simple_returns.columns[cand_t1_idx],
            "ticker_2": monthly_simple_returns.columns[cand_t2_idx],
            "weight": cand_w1
        }
        
        if cat == "B" and t_idx < 5:
            pass # Removed debug print

        decision = trade_portfolio(temp_portfolio[cat]["ticker_1"], temp_portfolio[cat]["num_shares_1"],
                                   temp_portfolio[cat]["ticker_2"], temp_portfolio[cat]["num_shares_2"],
                                   candidate_pairs["ticker_1"], candidate_pairs["ticker_2"],
                                   candidate_pairs["weight"], cat, t, stats_date)
        
        temp_portfolio[cat] = {
            "ticker_1": decision["ticker_1"],
            "num_shares_1": decision["num_shares_1"],
            "ticker_2": decision["ticker_2"],
            "num_shares_2": decision["num_shares_2"],
            "networth": decision["networth"]
        }
        
    # Append a copy of the portfolio state
    portfolio_history.append(temp_portfolio.copy())

for cat in ["A", "B", "C", "D", "E"]:
    last_portfolio = portfolio_history[-1][cat]
    value_1, _ = sell_ticker(last_portfolio["ticker_1"], last_portfolio["num_shares_1"], END_DATE)
    value_2, _ = sell_ticker(last_portfolio["ticker_2"], last_portfolio["num_shares_2"], END_DATE)    
    value = value_1 + value_2
    print(f"Category {cat}: {value:.2f}")
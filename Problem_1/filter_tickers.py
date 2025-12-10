import pandas as pd
import os
from tqdm import tqdm

# Constants
DATA_DIR = '../Data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_PARQUET = os.path.join(DATA_DIR, 'combined_data.parquet')
REQUIRED_DENSITY = 200 
MIN_START_DATE = pd.Timestamp('2012-01-01')

def process_all_data():
    valid_data_frames = []
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    print(f"Processing {len(files)} files with high-performance filter...")

    # Iterate through all files
    for f in tqdm(files):
        try:
            file_path = os.path.join(RAW_DATA_DIR, f)
            ticker_name = f.replace('.csv', '')
            
            # Read file
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Handle duplicate columns (e.g. Close, Close.1)
            # Some files have empty columns with the original name and data in the .1 column
            for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                # Find all columns that start with the name (e.g. Close, Close.1, Close.2)
                candidates = [c for c in df.columns if c == col or c.startswith(f"{col}.")]
                
                if len(candidates) > 1:
                    # Pick the one with the most non-NaN values
                    best_col = max(candidates, key=lambda c: df[c].count())
                    
                    # If the best column isn't the original name, replace the original with it
                    if best_col != col:
                        df[col] = df[best_col]
                    
                    # Drop the duplicates/suffixes, keep only the canonical name
                    cols_to_drop = [c for c in candidates if c != col]
                    df.drop(columns=cols_to_drop, inplace=True)
            
            if df.empty: continue

            # --- Validation Logic ---
            start = df.index[0]
            end = df.index[-1]
            total_days = (end - start).days
            
            if total_days <= 0: continue

            # Start Date Check
            if start > MIN_START_DATE:
                # print(f"Skipping {ticker_name}: Starts too late ({start.date()})")
                continue

            # Negative Price Check
            if df['Close'].min() <= 0:
                # print(f"Skipping {ticker_name}: Contains negative or zero prices")
                continue

            # Density Check
            expected_count = (total_days / 365.0) * REQUIRED_DENSITY
            # print(f"Density check: {ticker_name} has {len(df):,} rows, expected {expected_count:,}")
            if len(df) <= expected_count:
                continue
            
            # --- Preparation ---
            # Tag the data with the Ticker name so we can distinguish it later
            df['Ticker'] = ticker_name
            df.index.name = 'Date'
            
            # Add to list (RAM is cheap for you, so we keep it all)
            valid_data_frames.append(df)

        except Exception:
            continue

    if valid_data_frames:
        print(f"\nMerging {len(valid_data_frames)} datasets into one Giant DataFrame...")
        
        # This operation is heavy for 16GB RAM, but instant for 256GB
        master_df = pd.concat(valid_data_frames)
        
        print(f"Success! Created DataFrame with {len(master_df):,} rows.")
        print("-" * 30)

        # Save as Parquet (Recommended for speed)
        # Requires: pip install pyarrow
        try:
            print(f"Saving to {OUTPUT_PARQUET} (Fastest)...")
            master_df.to_parquet(OUTPUT_PARQUET)
        except ImportError:
            print("To save as Parquet (much faster), run: pip install pyarrow")
        
        print("Done.")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    process_all_data()

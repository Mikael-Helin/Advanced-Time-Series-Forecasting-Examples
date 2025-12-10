import pandas as pd
import numpy as np
import os
from metrics import get_data

def portfolio_value(initial_value, initial_day, end_day, ticker_dict):
    """
    Calculates the value of a portfolio on end_day.
    """
    initial_day = pd.to_datetime(initial_day)
    end_day = pd.to_datetime(end_day)
    
    total_weight = sum(ticker_dict.values())
    if abs(total_weight - 1.0) > 1e-6:
        # Normalize
        ticker_dict = {t: w / total_weight for t, w in ticker_dict.items()}
        
    current_value = 0.0
    
    for ticker, weight in ticker_dict.items():
        if weight <= 0:
            continue
            
        df = get_data(ticker)
        if df is None:
            current_value += initial_value * weight
            continue
            
        period_data = df.loc[initial_day:]
        if period_data.empty:
            current_value += initial_value * weight
            continue
            
        start_price = period_data.iloc[0]['Close']
        
        allocation = initial_value * weight
        shares = allocation / start_price
        
        period_data_end = df.loc[:end_day]
        if period_data_end.empty:
             current_value += allocation
             continue
             
        end_price = period_data_end.iloc[-1]['Close']
        
        position_value = shares * end_price
        current_value += position_value
        
    return current_value

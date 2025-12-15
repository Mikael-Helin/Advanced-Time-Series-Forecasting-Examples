# Advanced Time Series Forecasting: From Naive Models to Transformers

This project explores time series forecasting for financial data, specifically stock prices. The goal is to compare naive methods, traditional methods and modern deep learning algorithms such as LSTM, GRU and transformers.

The goal in each problem, is to select the best performing portfolio, based on the selected metrics. Data is fetched from Yahoo Finance in the time interval from 1980 to end of 2024.

To sell a stock, it costs 1% of its value. To buy a stock, there are no fees. Trades can be done on the first open trading day of each month. So on each first trading day of each month, the algorithm stops and makes optional trades. Trades start in January 2017 and end in December 2024. In the end of December 2024, the portfolio is sold. The objective is to have maximum amount of cash at the end of December 2024.

Cash is stored in a bank account with 0% interest rate.

For the non-deep learning algorithms, we use 5 years of closing price data to estimate parameters.

We measure use two different data sets for measuring monthly volatility, 1 year of data or 5 years of data. We have following categories of portfolios in which we want to maximize the return:

    A: Max 5 year yearly volatility of 2.5% or max 1 year yearly volatility of 3.0%
    B: Max 5 year yearly volatility of 5% or max 1 year yearly volatility of 6%
    C: Max 5 year yearly volatility of 10% or max 1 year yearly volatility of 12%
    D: Max 5 year yearly volatility of 20% or max 1 year yearly volatility of 24%
    E: Max 5 year yearly volatility of 40% or max 1 year yearly volatility of 50%
    F: Infinite volatility

A portfolio is rebalanced if the expected return of the new portfolio is higher than the expected return of the current portfolio/0.99. A portfolio can have at most 20 stocks. The start capital is 100k USD. The date format is `YYYY-MM-DD` or `YYYYMMDD`.

## Problem 1: Data Acquisition

Fetching data from Yahoo Finance and preparing it for modeling. We take tickers from NYSE that are in the file Data/nyse_listed.csv. A requirement is that the stock has existed since 2011 December 22nd, such accepted tickers are listed in Data/valid_tickers.csv.

The data is fetched from Yahoo Finance and stored in `Data/raw/{ticker}.csv`. The data is stored as a csv file with the following columns:

    Date, Open, High, Low, Close, Volume

The downloaded tickers are then tested for timesteps. If a ticker has more than 200 timesteps for each year, then it is added to `Data/nyse_cleaned.csv`. The algorithm is straightforward, it takes (last day - first day)/365*200 > number of timesteps as condition. For example last day is 2024-12-09 and first day is 2017-01-02, then we need at least 2898/365*200 = 1588 timesteps to pass such ticker.

### Portfolio Bot

This portfolio bot takes only a singel stock or keeps the cash. It does not mix stocks and cash in a portfolio.

## Problem 2: Computing Correlations and Log Returns

The log returns are computed as

    ln(close_price_t / close_price_{t-1}).

Create the functions, that compute the correlation matrix and the log returns of the stocks. To compute the log returns, we use the closing prices. For the log returns, we compute daily, weekly, monthly and yearly log returns. For the correlation matrix, we use the closing prices and we compute the daily, weekly, monthly and yearly correlation matrix.

To compute the log returns, the inparameters are: ticker, start_date, end_date, returns_type (daily, weekly, monthly, yearly).

To compute the correlation matrix, the inparameters are: ticker_1, ticker_2, start_date, end_date, correlation_type (daily, weekly, monthly, yearly). When ticker_1 == ticker_2, then the correlation matrix is the variance of the stock.

Assume we sum N log returns of a stock, it holds that

    ln(close_price_{t+1}/close_price_t) + ... + ln(close_price_{t+N}/close_price_{t+N-1}) = ln(close_price_{t+N}/close_price_t).

The correlation is computed on the log returns. So the correlation is computed as

    corr(r_1, r_2) = cov(r_1, r_2) / (std(r_1) * std(r_2))

where r_1 and r_2 are the log returns of the two stocks.

We assume a week is 5 days and a month is 21 days.

### Log Returns

* We use all daily data from 2017-01-02 to 2024-12-09 to compute daily log returns.
* We use weekly data (friday closings) from 2017-01-02 to 2024-12-09 to compute weekly log returns.
* We use monthly data (last day of month closings) from 2017-01-02 to 2024-12-09 to compute monthly log returns.
* We use yearly data (last day of year closings) from 2017-01-02 to 2024-12-09 to compute yearly log returns.
* Then we will test if 5 days of log returns are equivalent to 1 week of log returns.
* Then we will test if 21 days of log returns are equivalent to 1 month of log returns.
* Then we will test if 252 days of log returns are equivalent to 1 year of log returns.

Added log returns are equivalent to the whole log return of the stock.

### Correlation Matrix of Log Returns

* We use all daily data from 2017-01-02 to 2024-12-09 to compute daily correlation matrix of log returns.
* We use weekly data (friday closings) from 2017-01-02 to 2024-12-09 to compute weekly correlation matrix of log returns.
* We use monthly data (last day of month closings) from 2017-01-02 to 2024-12-09 to compute monthly correlation matrix of log returns.
* We use yearly data (last day of year closings) from 2017-01-02 to 2024-12-09 to compute yearly correlation matrix of log returns.
* Then compare the correlation matrices of log returns.

It shows that the correlation matrix of log returns changes over time, which indicates that an i.i.d assumption is not valid.

### Portfolio Solver

This portfolio solver uses the monthly correlation matrix of log returns to find the best portfolio. In this portfolio, we can pick only 2 stocks.

Results at 2024-12-31:

    A:
    B:
    C:
    D:
    E:

STOP HERE!

We create a function `portfolio_value(initial_value, initial_day, end_day, ticker_dict)` where ticker_dict keys are the tickers and the values are the weights of the portfolio. Ideally every weight is non-negative and the sum of all weights is 1. The function returns the value of the portfolio on end_day. And we cannot have more than 20 stocks in the portfolio.





STOP HERE!


The tickers are found in Data/nyse_listed.csv.





We assume a years is 252 trading days. So the yearly volatility is calculated as the standard deviation of daily returns times sqrt(252). When we pick a portfolio we base on past volatility of last 252 trading days. We do not try to adjust for future volatility.

A portfolio starts with 100k USD and has to end with as much money as possible. A portfolio can have minimum 1 stock and maximum 20 stocks. There is no interest rate on cash.

## Project Structure

The project is structured into a series of progressive tasks ("Problems"):

*   **Problem 1: Data Acquisition & Exploration**: Fetching data from Yahoo Finance and preparing it for modeling.
*   **Problem 2: Naive Models & Baselines**: Establishing baselines using simple methods (Persistence, Moving Average, ARIMA).
*   **Problem 3: Advanced Models - LSTM**: Implementing Long Short-Term Memory (LSTM) networks for forecasting.
*   **Problem 4: Advanced Models - Transformer**: Implementing Transformer architectures with attention mechanisms.
*   **Problem 5: Portfolio Data & Analysis**: Expanding to multivariate time series with multiple stocks.
*   **Problem 6: Portfolio Forecasting**: Applying advanced models to the full portfolio.

## Key Objectives

*   **Model Implementation**: Build and train LSTM and Transformer models.
*   **Performance Comparison**: Evaluate models using RMSE, MAE, and directional accuracy.
*   **Interpretability**: Visualize attention weights to understand model decision-making.

## Tech Stack

*   **Language**: Python
*   **Data Source**: Yahoo Finance (`yfinance`)
*   **Libraries**: Pandas, NumPy, Matplotlib/Seaborn, PyTorch/TensorFlow, Scikit-learn

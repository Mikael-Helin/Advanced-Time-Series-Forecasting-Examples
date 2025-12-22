# Advanced Time Series Forecasting: From Naive Models to Transformers

This project explores time series forecasting for financial data, specifically stock prices. The goal is to compare naive methods, traditional methods and modern deep learning algorithms such as LSTM, GRU and transformers.

The goal in each problem, is to select the best performing portfolio, based on the selected metrics. Data is fetched from Yahoo Finance in the time interval from 1980 to end of 2024.

To sell a stock, it costs 1% of its value. To buy a stock, there are no fees. Trades can be done on the first open trading day of each month. So on each first trading day of each month, the algorithm stops and makes optional trades. Trades start in January 2017 and end in December 2024. In the end of December 2024, the portfolio is sold. The objective is to have maximum amount of cash at the end of December 2024.

Cash is stored in a bank account with 0% interest rate.

For the non-deep learning algorithms, we use 5 years of monthly closing price data to estimate parameters.

We measure two different data sets for measuring monthly volatility, 1 year of data or 5 years of data. We have following categories of portfolios in which we want to maximize the return:

    A: Max 5 year yearly volatility of 2.5% or max 1 year yearly volatility of 3.0%
    B: Max 5 year yearly volatility of 5% or max 1 year yearly volatility of 6%
    C: Max 5 year yearly volatility of 10% or max 1 year yearly volatility of 12%
    D: Max 5 year yearly volatility of 25% or max 1 year yearly volatility of 30%
    E: Max 5 year yearly volatility of 50% or max 1 year yearly volatility of 60%
    F: Infinite volatility

A portfolio is rebalanced if the expected return of the new portfolio is higher than the expected return of the current portfolio/0.99. A portfolio can have at most 20 stocks. The start capital is 100k USD. The date format is `YYYY-MM-DD` or `YYYYMMDD`.

## Problem 1: Data Acquisition

Fetching data from Yahoo Finance and preparing it for modeling. We take tickers from NYSE that are in the file Data/nyse_listed.csv. A requirement is that the stock has existed since 2011 December 22nd, such accepted tickers are listed in Data/valid_tickers.csv.

The data is fetched from Yahoo Finance and stored in `Data/raw/{ticker}.csv`. The data is stored as a csv file with the following columns:

    Date, Open, High, Low, Close, Volume

The downloaded tickers are then tested for timesteps. If a ticker has more than ~200 timesteps for each year, then it is added to `Data/nyse_cleaned.csv`.

### Portfolio Bot

This portfolio bot takes only a singel stock or keeps the cash. It does not mix stocks and cash in a portfolio.

Results:

    Date: 2024-12-01
    A: cash 100000.00
    B: cash 82348.65
    C: TVE 85244.02
    D: ETN 100832.25
    E: FIX 52900.00,
    F: LEU 148451.29


There were about 1300 tickers that were selected for the portfolio bot. It appears no ticker was good enough for low volatility categories. Next step is to allow 2 stocks in the portfolio.

## Problem 2: Computing Correlations and Log Returns

Now we allow 2 stocks in the portfolio, we also consider cash as a riskless stock. We compute the expected return and the correlation matrix on the simple returns of the stocks. The solution uses a 3-D tensor to store the correlation matrix for each date.

    Date: 2024-12-01
    A: 108732.38
    B: 91129.08
    C: 124528.29
    D: 209908.35
    E: 51450.81
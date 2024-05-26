import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.patches as patches
import matplotlib.dates as mdates
import itertools as itt
import numbers
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Iterable, Tuple, List
from scipy.optimize import brute
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from itertools import combinations

# Fetch data
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data['Adj Close']

# Calculate moving averages and signals (MODIFY PER STRAT)
def strategy_signals(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data
    signals['short_mavg'] = data.rolling(window=int(short_window), min_periods=1, center=False).mean()
    signals['long_mavg'] = data.rolling(window=int(long_window), min_periods=1, center=False).mean()
    signals['signal'] = 0.0
    short_window = int(short_window)
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Backtest the strategy
def backtest_strategy(signals):
    initial_capital = float(100000.0)
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['stock'] = 1000 * signals['signal']
    portfolio = positions.multiply(signals['price'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()   
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['drawdown'] = (portfolio['total'].cummax() - portfolio['total']) * 100 / portfolio['total'].cummax()
    portfolio['max_drawdown'] = portfolio['drawdown'].cummax()
    return portfolio


# Parameter optimization
def optimize_parameters(data, parameter_ranges):
    def objective_function(parameters, *data):
        signals = strategy_signals(data[0], parameters[0], parameters[1])
        portfolio = backtest_strategy(signals)
        
        # Calculate returns
        returns = portfolio['returns']
        mean_return = returns.mean()
        std_return = returns.std()
        downside_std = returns[returns < 0].std()
        
        # Avoid division by zero
        std_return = std_return if std_return != 0 else np.nan
        downside_std = downside_std if downside_std != 0 else np.nan

        # Calculate annualized return assuming 252 trading days
        annualized_return = (1 + mean_return) ** 252 - 1

        # Calculate maximum drawdown
        drawdown = (portfolio['total'].cummax() - portfolio['total']).max()
        calmar_ratio = annualized_return / drawdown if drawdown != 0 else np.nan

        # Calculate Sortino Ratio
        sortino_ratio = annualized_return / downside_std if downside_std != 0 else np.nan

        # Calculate the Sharpe Ratio
        sharpe_ratio = mean_return / std_return

        # Combine the ratios for optimization
        # Here, we negatively weight them as we want to minimize the function with brute
        # Adjust the weights as necessary based on your preference for each metric's importance
        combined_metric = -1 * (sortino_ratio + calmar_ratio + sharpe_ratio)

        return combined_metric
    
    result = brute(objective_function, ranges=parameter_ranges, args=(data,), full_output=True, finish=None)
    return result[0]  # Return the best parameters found

# Walk-Forward Analysis
def walk_forward_analysis(data, in_sample_years, out_sample_years, parameter_ranges):
    start_date = data.index.min()
    end_date = data.index.max()
    in_sample_period = pd.DateOffset(years=in_sample_years)
    out_sample_period = pd.DateOffset(years=out_sample_years)
    
    current_date = start_date
    results = []
    all_portfolios = []  # Store each out-of-sample portfolio for plotting
    top_params = []

    while current_date + in_sample_period + out_sample_period < end_date:
        in_sample_data = data[current_date:current_date+in_sample_period]
        out_sample_data = data[current_date+in_sample_period:current_date+in_sample_period+out_sample_period]
        
        # Optimize parameters on in-sample data
        best_params = optimize_parameters(in_sample_data, parameter_ranges)
        best_signals = strategy_signals(out_sample_data, best_params[0], best_params[1])
        best_portfolio = backtest_strategy(best_signals)
        
        results.append(best_portfolio['total'].iloc[-1])
        all_portfolios.append(best_portfolio)  # Collect each portfolio
        top_params.append(best_params)

        current_date += out_sample_period
    
    print(all_portfolios[0])
    print(best_signals)
    return results, all_portfolios, top_params
        
def plot_results(portfolio):
    # Assuming 'portfolios' is a list of DataFrames returned from backtest_strategy
    # Concatenate all portfolio DataFrames along the index
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

    # Plot returns on the first subplot
    axs[0].plot(portfolio.index, portfolio['returns'], label='Returns', color='blue')
    axs[0].set_title('Portfolio Returns')
    axs[0].set_ylabel('Returns (%)')
    axs[0].grid(True)

    # Plot drawdown on the second subplot
    axs[1].plot(portfolio.index, portfolio['drawdown'], label='Drawdown', color='red')
    axs[1].set_title('Portfolio Drawdown')
    axs[1].set_ylabel('Drawdown (%)')
    axs[1].set_xlabel('Date')
    axs[1].grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

# MODIFY PER STRAT
def evaluate_strategy_performance(data, short_window, long_window):
    signals = strategy_signals(data, short_window, long_window)
    portfolio = backtest_strategy(signals)
    if portfolio['returns'].std() != 0:  # Avoid division by zero
        sharpe_ratio = portfolio['returns'].mean() / portfolio['returns'].std() 
    else:
        sharpe_ratio = 0
    return sharpe_ratio

# MODIFY PER STRAT
def CPCV(data, num_splits, purge_length):
    if len(data) < purge_length + num_splits:  # Basic check to ensure there's enough data
        return []

    tscv = TimeSeriesSplit(n_splits=num_splits)
    results = []
    
    for train_index, test_index in tscv.split(data):
        # Applying purge
        purge_start = test_index[0] - purge_length
        train_index = train_index[train_index < purge_start]

        # Split data into in-sample and out-sample
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Here, implement the strategy testing and parameter optimization
        best_params = optimize_parameters(train_data, parameter_ranges)
        best_signals = strategy_signals(test_data, best_params[0], best_params[1])
        portfolio = backtest_strategy(best_signals)

        results.append(portfolio)

        '''new_test_start_index = test_index[0] + purge_length

        if new_test_start_index < len(data):  # Ensure the new index does not exceed the data length
            train_data = data.iloc[train_index]
            test_data = data.iloc[new_test_start_index:]
            if not test_data.empty:
                train_performance = evaluate_strategy_performance(train_data, 30, 200)
                test_performance = evaluate_strategy_performance(test_data, 30, 200)
                results.append((train_performance, test_performance))
            else:
                print(f"No data available for testing starting from index {new_test_start_index}")
        else:
            print(f"Adjusted test start index {new_test_start_index} exceeds data length {len(data)}")'''


    print(results)
    return results

'''def combinatorial_purged_cross_validation(data, n_splits, purge_gap):
    """
    Implement combinatorial purged cross-validation.

    Args:
    - data: The full dataset as a Pandas DataFrame.
    - n_splits: The number of splits for the cross-validation.
    - purge_gap: The gap between the training and testing data to avoid lookahead bias.

    Returns:
    - A list of results for each out-of-sample test.
    """
    # Generate all possible combinations of train-test splits
    kf = KFold(n_splits=n_splits, shuffle=False)
    splitted_data = kf.split(data)
    results = []
    for i, j in splitted_data:
        print(i, j)
    for train_indices, test_indices in splitted_data:
        # Apply a purge gap
        train_indices = np.array([idx for idx in train_indices if idx < test_indices[0] - purge_gap])
        test_indices = np.array([idx for idx in test_indices if idx > train_indices[-1] + purge_gap])

        if len(train_indices) == 0 or len(test_indices) == 0:
            continue

        # Split data
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

        # Evaluate parameters on the train set
        best_params = optimize_parameters(train_data, parameter_ranges)  # Implement this function based on strategy
        best_signals = strategy_signals(test_data, best_params[0], best_params[1])  # Implement this function based on strategy
        portfolio = backtest_strategy(best_signals)  # Implement this function based on strategy
        
        # Evaluate performance on the test set
        test_performance = evaluate_strategy_performance(portfolio)  # Implement this function based on strategy
        
        results.append(test_performance)

    return results
'''

def combinatorial_purged_cross_validation(data, total_splits, out_sample_splits, purge_gap):
    """
    Implement combinatorial purged cross-validation.

    Args:
    - data: The full dataset as a Pandas DataFrame.
    - total_splits: Total number of splits (N).
    - out_sample_splits: Number of out-sample splits in each combination (K).
    - purge_gap: The gap between the training and testing data to avoid lookahead bias.

    Returns:
    - A list of results for each out-of-sample test configuration.
    """
    results = []
    indices = np.array_split(np.arange(len(data)), total_splits)  # Split indices into N parts
    
    # Generate all combinations of indices to be used as out-sample data
    for out_sample_indices in combinations(range(total_splits), out_sample_splits):
        in_sample_indices = [idx for idx in range(total_splits) if idx not in out_sample_indices]
        
        # Combine indices for in-sample and out-sample data
        train_indices = np.concatenate([indices[idx] for idx in in_sample_indices]).astype(int)
        test_indices = np.concatenate([indices[idx] for idx in out_sample_indices]).astype(int)

        # Apply a purge gap
        if len(train_indices) > 0 and len(test_indices) > 0:
            test_indices = test_indices[test_indices > (train_indices[-1] + purge_gap)]

        if len(train_indices) == 0 or len(test_indices) == 0:
            continue  # Skip if there's no valid data to test or train

        # Split data
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

        # Evaluate parameters on the train set
        best_params = optimize_parameters(train_data)  # Define or adjust your function
        best_signals = strategy_signals(test_data, best_params)  # Define or adjust your function
        portfolio = backtest_strategy(best_signals)  # Define or adjust your function
        
        # Evaluate and collect performance
        test_performance = evaluate_strategy_performance(portfolio)  # Define or adjust your function
        results.append(test_performance)

    return results

def analyze_results(results):
    if not results:
        return 0, 0  # Return 0 for PBO and PPSR if no results are available

    pbo_count = sum(1 for train_perf, test_perf in results if test_perf < train_perf) / len(results)
    ppsr_count = sum(1 for _, test_perf in results if test_perf > 1) / len(results)
    return pbo_count, ppsr_count

def robustness_test(data, num_paths, out_sample_splits, purge_length):
    results = combinatorial_purged_cross_validation(data, num_paths, out_sample_splits, purge_length)
    print(results)
    pbo, ppsr = analyze_results(results)
    return pbo, ppsr

# Monte Carlo Simulation
def monte_carlo_simulation(data, n_simulations, forecast_horizon):
    log_returns = np.log(data / data.shift(1))
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (forecast_horizon, n_simulations)))
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1]
    for t in range(1, forecast_horizon):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]
    
    plt.figure(figsize=(10,6))
    plt.plot(price_paths)
    plt.title('Monte Carlo Simulation of Stock Prices')
    plt.show()
    return price_paths

# Backtest Monte Carlo Simulations (MODIFY PER STRAT)
def backtest_on_simulations(price_paths, short_window, long_window):
    results = []
    sharpe_ratio = []
    for i in range(price_paths.shape[1]):
        signals = strategy_signals(pd.Series(price_paths[:, i]), short_window, long_window)  
        portfolio = backtest_strategy(signals)
        if portfolio['returns'].std() != 0:
            sharpe_ratio.append(portfolio['returns'].mean() / portfolio['returns'].std())
        else:
            sharpe_ratio.append(0)
        results.append(portfolio['total'].iloc[-1])
    return results, sharpe_ratio

if __name__ == "__main__":
    symbol = 'AAPL'
    start = '2013-01-01'
    end = '2022-12-31'
    data = fetch_data(symbol, start, end)
    
    # Walk-Forward Analysis
    parameter_ranges = (slice(20, 60, 10), slice(180, 240, 20))  # Define ranges for short_window and long_window (MODIFY PER STRAT)
    wf_results, portfolios, top_params = walk_forward_analysis(data, 3, 1, parameter_ranges)
    print("Walk-Forward Results:", wf_results)
    print("Optimized Parameters:", top_params)
    #signals = strategy_signals(data, top_params[-1][0], top_params[-1][1])
    #portfolio = backtest_strategy(signals)
    #plot_results(portfolio)

    # Robustness Testing
    #pbo, ppsr = robustness_test(data, 10, 2, 5)  # Example: 10 paths, 5 days purge length
    # Constants
    num_paths = 5
    k = 2
    N = num_paths + 1
    t_final = 6
    embargo_td = pd.Timedelta(days=1)* t_final
    cv = CombPurgedKFoldCV(n_splits=N, n_test_splits=k, embargo_td=embargo_td)
    _, paths, _= back_test_paths_generator(X.shape[0], N, k)

    # Plotting
    groups = list(range(X.shape[0]))
    fig, ax = plt.subplots()
    plot_cv_indices(cv, X, y, groups, ax, num_paths, k)
    plt.gca().invert_yaxis()
    print("Probability of Overfitting:", pbo)
    print("Probability of Positive Sharpe Ratio:", ppsr)

    # Monte Carlo Simulation
    mc_paths = monte_carlo_simulation(data, 1000, 252)  # 1000 simulations over 252 trading days (about 1 year)
    final_values, final_sharpe = backtest_on_simulations(mc_paths, 30, 200) # (MODIFY PER STRAT)
    average_gain = sum(final_values) / len(final_values)
    average_sharpe = sum(final_sharpe) / len(final_sharpe)
    print("Average Gain from Monte Carlo Simulations:", average_gain)    
    print("Average Sharpe Ratio from Monte Carlo Simulations:", average_sharpe)    



# FROM WALK FORWARD OPTIMIZATION I WANT RETURNS AND DRAWDOWN OVER THE ENTIRE PERIOD OF STOCK ON A PARTICULAR PARAM, IF NOT
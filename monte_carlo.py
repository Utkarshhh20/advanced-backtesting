import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


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
    portfolio['cumulative_returns'] = (portfolio['total'] / initial_capital - 1) * 100  # Cumulative returns in percentage
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
        calmar_ratio = annualized_return / drawdown if drawdown != 0 else 0

        # Calculate Sortino Ratio
        sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0

        # Calculate the Sharpe Ratio
        sharpe_ratio = mean_return / std_return

        # Combine the ratios for optimization
        # Here, we negatively weight them as we want to minimize the function with brute
        # Adjust the weights as necessary based on your preference for each metric's importance
        combined_metric = -1 * (sortino_ratio + calmar_ratio + sharpe_ratio)

        return combined_metric
    
    result = brute(objective_function, ranges=parameter_ranges, args=(data,), full_output=True, finish=None)
    return result[0]  # Return the best parameters found

# Monte Carlo Simulation
def monte_carlo_simulation(data, n_simulations, forecast_horizon):
    log_returns = np.log(data / data.shift(1))
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (forecast_horizon, n_simulations)))
    price_paths = np.zeros((forecast_horizon, n_simulations))
    price_paths[0] = data.iloc[-1]
    for t in range(1, forecast_horizon):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]
    
    return price_paths

def plot_monte_carlo(price_paths):
    """
    Plot Monte Carlo simulation results.
    
    Parameters:
        price_paths (np.ndarray): Simulated price paths.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths)
    plt.title('Monte Carlo Simulation of Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()

# Backtest Monte Carlo Simulations (MODIFY PER STRAT)
def backtest_on_simulations(price_paths, short_window, long_window):
    returns_df = []
    metrics = []
    cum_returns_df = []

    for i in range(price_paths.shape[1]):
        signals = strategy_signals(pd.Series(price_paths[:, i]), short_window, long_window)  
        portfolio = backtest_strategy(signals)
        returns = portfolio['returns']
        cum_returns = portfolio['cumulative_returns']
        # Sharpe Ratio
        if portfolio['returns'].std() != 0:
            sharpe_ratios = returns.mean() / returns.std()
        else:
            sharpe_ratios = 0

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        if negative_returns.std() != 0:
            sortino_ratios = returns.mean() / negative_returns.std()
        else:
            sortino_ratios = 0

        # Calmar Ratio
        portfolio['drawdown'] = (portfolio['total'].cummax() - portfolio['total']) / portfolio['total'].cummax()
        portfolio['max_drawdown'] = portfolio['drawdown'].cummax()
        max_drawdown = portfolio['max_drawdown'].iloc[-1]
        #if portfolio['total'].iloc[-1] < portfolio['total'].iloc[0]:
        #    annualized_return = -(np.power(abs(portfolio['total'].iloc[-1]), (252.0 / len(portfolio))) - 1)
        #else:
        #    annualized_return = np.power(portfolio['total'].iloc[-1], (365.0 / len(portfolio))) - 1
        annualized_return = ((portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) ** (252 / len(portfolio))) - 1

        if max_drawdown != 0:
            calmar_ratios = annualized_return / max_drawdown
        else:
            calmar_ratios = 0

        # Final Portfolio Value and Increase
        final_value = portfolio['total'].iloc[-1]
        initial_value = portfolio['total'].iloc[0]
        percentage_increase = ((final_value - initial_value) / initial_value) * 100

        returns_df.append(
            portfolio['returns']
        )

        cum_returns_df.append(
            portfolio['cumulative_returns']
        )

        metrics.append({
            'Sharpe Ratio': sharpe_ratios,
            'Sortino Ratio': sortino_ratios,
            'Calmar Ratio': calmar_ratios,
            'Max Drawdown': max_drawdown,
            'Final Portfolio Value': final_value,
            'Percentage Increase': percentage_increase
        })

    metrics_df = pd.DataFrame(metrics).fillna(0)
    returns_df = pd.DataFrame(returns_df).transpose().fillna(0)
    returns_df.columns = [f'Simulation_{i + 1}' for i in range(price_paths.shape[1])]  
    cum_returns_df = pd.DataFrame(cum_returns_df).transpose().fillna(0)
    cum_returns_df.columns = [f'Simulation_{i + 1}' for i in range(price_paths.shape[1])]  
    return metrics_df, returns_df, cum_returns_df

def quantile_graph(results):
    # Calculate required percentiles for each day
    p5 = results.quantile(0.05, axis=1)
    p10 = results.quantile(0.10, axis=1)
    p25 = results.quantile(0.25, axis=1)
    p75 = results.quantile(0.75, axis=1)
    p90 = results.quantile(0.90, axis=1)
    p95 = results.quantile(0.95, axis=1)
    median = results.median(axis=1)

    # Plotting

    plt.figure(figsize=(15, 8))
    #plt.fill_between(results.index, p0, p100, color='green', alpha=0.7, label='0th-100th Percentile')
    plt.fill_between(results.index, p5, p95, color='green', alpha=0.2, label='5th-95th Percentile')
    plt.fill_between(results.index, p10, p90, color='green', alpha=0.3, label='10th-90th Percentile')
    plt.fill_between(results.index, p25, p75, color='green', alpha=0.5, label='25th-75th Percentile')
    plt.plot(median, color='red', label='Median')
    plt.title('Strategy Return Projections')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()

    plt.show()

def monte_carlo_returns(returns_paths):
    """
    Plot Monte Carlo simulation results for strategy returns.
    
    Parameters:
        returns_paths (np.ndarray): Simulated returns paths.
    """
    plt.figure(figsize=(15, 8))
    days = np.arange(returns_paths.shape[0])
    for i in range(returns_paths.shape[1]):
        plt.plot(days, returns_paths[:, i], 'g-', lw=1, alpha=0.3)  # Green lines with low opacity

    plt.title('Monte Carlo Simulation of Strategy Returns')
    plt.xlabel('Days')
    plt.ylabel('Strategy Return %')
    plt.grid(True)  # Optional: Adds grid for better readability
    plt.show()

def analyze_drawdowns_and_risk_of_ruin(metrics_df, drawdown_threshold=20.0):
    """
    Analyze drawdowns and compute the risk of ruin.

    Parameters:
        metrics_df (DataFrame): DataFrame containing drawdown data.
        drawdown_threshold (float): Threshold for considering a ruinous drawdown.
    """
    drawdowns = metrics_df['Max Drawdown'] * 100
    risk_of_ruin = np.sum(drawdowns >= drawdown_threshold) / len(drawdowns)

    # Plotting the drawdown distribution
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(drawdowns, bins=20, alpha=0.75, color='red')
    
    plt.axvline(x=drawdown_threshold, color='red', linestyle='dashed', linewidth=1)
    plt.text(drawdown_threshold + 0.5, plt.ylim()[1] * 0.9, f'RISK OF RUIN = {risk_of_ruin:.3%}', color='black')
    plt.title('Maximum Drawdown distribution over hundreds of simulations')
    plt.xlabel('Drawdown %')
    plt.ylabel('Frequency')
    plt.show()

    return risk_of_ruin

if __name__ == "__main__":
    symbol = 'AAPL'
    start = '2013-01-01'
    end = '2022-12-31'
    data = fetch_data(symbol, start, end)

    n_simulations = 1000
    forecast_horizon = 252
    # Monte Carlo Simulation
    mc_paths = monte_carlo_simulation(data, n_simulations, forecast_horizon)  # x simulations over y trading days
    plot_monte_carlo(mc_paths)
    final_results, returns, cum_returns = backtest_on_simulations(mc_paths, 30, 200) # (MODIFY PER STRAT)
    print(final_results)
    mc_paths_df = pd.DataFrame(mc_paths, columns=[f'Simulation_{i+1}' for i in range(n_simulations)])
    print(mc_paths_df)
    #print(returns)
    quantile_graph(cum_returns)
    print(returns)
    cum_returns_array = np.array([cum_returns[f'Simulation_{i+1}'] for i in range(n_simulations)]).T
    monte_carlo_returns(cum_returns_array)
    analyze_drawdowns_and_risk_of_ruin(final_results, drawdown_threshold=20.0)
# FIX ALL RATIO FORMULAS ESP CALMAR, CHECK PERCENTAGE CALC, CHECK DRAWDOWN CALC AND QUANTILE GRAPH CALC, CHECK MONTE CARLO RETURNS GRAPH
# STRATEGY IS BUGGED, GOES TO 1000 UP AND DOWN, NO STOP LOSS ETC, STOCKS BOUGHT BASED ON NUMBER, ALGO APPLIED BUT COMPLETE TRADING LOGIC ISNT
# CORRECT RETURNS ARRAY BUGGED GRAPH
# RATIO VALUES ARE WRONG CAUSE NEGATIVE VALUES ETC, EXAMPLE MAX DRAWDOWN MUST BE < 1 AND > 0, CHECK
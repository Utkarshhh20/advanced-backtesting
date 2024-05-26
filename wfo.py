import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.patches as patches

from scipy.optimize import brute


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
        #annualized_return = (1 + mean_return) ** 252 - 1
        annualized_return = ((portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) ** (252 / len(portfolio))) - 1

        # Calculate maximum drawdown
        portfolio['drawdown'] = (portfolio['total'].cummax() - portfolio['total']) / portfolio['total'].cummax()
        portfolio['max_drawdown'] = portfolio['drawdown'].cummax()
        max_drawdown = portfolio['max_drawdown'].iloc[-1]
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan

        # Calculate Sortino Ratio
        sortino_ratio = annualized_return / downside_std if downside_std != 0 else np.nan

        # Calculate the Sharpe Ratio
        sharpe_ratio = mean_return / std_return

        # Combine the ratios for optimization
        # Here, we negatively weight them as we want to minimize the function with brute
        # Adjust the weights as necessary based on your preference for each metric's importance
        #combined_metric = -1 * (sortino_ratio + calmar_ratio + sharpe_ratio)
        combined_metric = -1 * sharpe_ratio
        return combined_metric
    
    is_result = []
    result = brute(objective_function, ranges=parameter_ranges, args=(data,), full_output=True, finish=None)
    optimal_params = result[0]
    combined_value = result[1]
    param_grid = result[2]
    performance_values = result[3].flatten()
    for i in range(param_grid.shape[1]):
        for j in range(param_grid.shape[2]):
            short_window = param_grid[0, i, j]
            long_window = param_grid[1, i, j]
            performance = -1 * performance_values[i*param_grid.shape[2] + j]
            is_result.append([short_window, long_window, performance])

    return result[0], is_result  # Return the best parameters found

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
    is_results = []
    columns = ['Short Window', 'Long Window', 'Performance Value'] # 'Optimal Short Window', 'Optimal Long Window', 'Combined Objective Value'

    while current_date + in_sample_period + out_sample_period <= end_date:
        in_sample_data = data[current_date:current_date+in_sample_period]
        out_sample_data = data[current_date+in_sample_period:current_date+in_sample_period+out_sample_period]
        
        # Optimize parameters on in-sample data
        best_params,is_result = optimize_parameters(in_sample_data, parameter_ranges)
        best_signals = strategy_signals(out_sample_data, best_params[0], best_params[1])
        best_portfolio = backtest_strategy(best_signals)
        is_result = pd.DataFrame(is_result, columns = columns)

        results.append(best_portfolio['total'].iloc[-1])
        all_portfolios.append(best_portfolio)  # Collect each portfolio
        top_params.append(best_params)
        is_results.append(is_result)

        current_date += out_sample_period
    
    if current_date + in_sample_period <= end_date:
        in_sample_data = data[current_date:current_date+in_sample_period]

        best_params, is_result = optimize_parameters(in_sample_data, parameter_ranges)
        best_signals = strategy_signals(in_sample_data, best_params[0], best_params[1])
        best_portfolio = backtest_strategy(best_signals)
        is_result = pd.DataFrame(is_result, columns = columns)

        results.append(best_portfolio['total'].iloc[-1])
        all_portfolios.append(best_portfolio)  # Collect each portfolio
        top_params.append(best_params)
        is_results.append(is_result)
        
    is_results_df = pd.concat(is_results)
    # Group by Short Window and Long Window and calculate the mean Performance Value
    is_results_df = is_results_df.groupby(['Short Window', 'Long Window'], as_index=False)['Performance Value'].mean()

    print(is_results_df)

    return results, all_portfolios, top_params, is_results_df, is_results

if __name__ == "__main__":
    symbol = 'AAPL'
    start = '2012-12-27' # CHECK TRADING DAYS GAP FROM START TO END TO MAXIMIZE IN SAMPLE OUT SAMPLE COMBOS
    end = '2022-12-31'
    data = fetch_data(symbol, start, end)
    
    # Walk-Forward Analysis
    parameter_ranges = (slice(20, 60, 10), slice(180, 240, 20))  # Define ranges for short_window and long_window (MODIFY PER STRAT) 
    
    # FOR PARAMETER RANGES ITS START, END AND INCREMENT POINT. NOTE END POINT NOT INCLUDED
    wf_results, portfolios, top_params, is_results_df, is_results = walk_forward_analysis(data, 3, 1, parameter_ranges)

    # Find the row with the highest Performance Value
    max_performance_row = is_results_df.loc[is_results_df['Performance Value'].idxmax()]
    best_param = [max_performance_row[0], max_performance_row[1]]

    print("Walk-Forward Results:", wf_results)
    print("Optimized Parameters:", best_param)

    #signals = strategy_signals(data, top_params[-1][0], top_params[-1][1])
    #portfolio = backtest_strategy(signals)
    #plot_results(portfolio)

# FIX THE WALK FORWARD PLOT, SELECT BEST PARAM CONDITION TO BE ADDED (MAYBE MAKE DF OF ALL RESULTS AND ADD TO GET FINAL RANK)


# Sort the DataFrame in descending order by Performance Value
sorted_df = df.sort_values(by='Performance Value', ascending=False).reset_index(drop=True)
# Selected short and long window values
selected_short_window = 30
selected_long_window = 180

# Find the index of the selected combination
selected_index = sorted_df[(sorted_df['Short Window'] == selected_short_window) & (sorted_df['Long Window'] == selected_long_window)].index[0]
rank = selected_index  + 1
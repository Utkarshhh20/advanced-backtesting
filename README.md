# Scientific Backtester

## Overview

The Advanced Backtesting tool is a comprehensive framework designed to evaluate and optimize financial trading strategies. It leverages several advanced techniques to ensure robust performance assessment and parameter optimization, enabling traders and analysts to make informed decisions based on historical data and simulated projections.

## Features

### Walk-Forward Optimization
- **Walk-Forward Optimization (WFO)**: This technique involves repeatedly training and testing a strategy on different segments of data to simulate a realistic trading environment. It helps to avoid overfitting by ensuring the strategy is validated on unseen data.

### Combinatorial Purged Cross-Validation
- **Combinatorial Purged Cross-Validation (CPCV)**: This method splits the data into multiple combinations of training and testing sets, applying a purge to avoid lookahead bias. It provides a comprehensive evaluation by testing the strategy on various data segments.

### Monte Carlo Simulation
- **Monte Carlo Simulation**: Generates multiple price paths based on historical data to assess the strategy's performance under different market scenarios. It helps to understand the potential risks and returns by simulating numerous future price movements.

### No-Code Strategy Builder
- **No-Code Strategy Builder**: An intuitive interface for building trading strategies without writing code. Users can select from predefined indicators and set parameters to create and backtest their strategies.

### Detailed Performance Metrics
- **Performance Metrics**: The tool calculates various performance metrics, including Sharpe Ratio, Sortino Ratio, Calmar Ratio, maximum drawdown, and cumulative returns. These metrics provide a comprehensive assessment of the strategy's risk and return profile.

## Technologies Used

- **Python**: Core programming language for the entire framework.
- **pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **matplotlib**: Visualization of results and metrics.
- **yfinance**: Fetching historical market data.
- **scipy**: Optimization algorithms.
- **sklearn**: Cross-validation techniques.
- **Streamlit**: Interactive web application for user interaction and visualization.

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/AdvancedBacktesting.git
cd AdvancedBacktesting
```
Install the required dependencies:
```bash
git clone https://github.com/your-username/AdvancedBacktesting.git
cd AdvancedBacktesting
```

Set up API keys in your environment (if required):
```bash
export NEWSAPI_API_KEY='your-newsapi-key'
export OPENAI_API_KEY='your-openai-key'
export COINMARKETCAP_API_KEY='your-coinmarketcap-key'
```

## Usage
### Run the main script to start the analysis:

```bash
python main.py
```

## File Structure

- **main.py**: Main script for launching the Streamlit app.
- **wfo.py**: Contains functions for walk-forward optimization.
- **cpcv.py**: Contains functions for combinatorial purged cross-validation.
- **monte_carlo.py**: Contains functions for Monte Carlo simulations.
- **user_defined_strategy.py**: Handles user-defined strategies.
- **no_code_strategy.py**: Provides a no-code interface for building strategies.
- **requirements.txt**: List of dependencies required for the project.
- **README.md**: Project documentation.

## Example Output

The final output includes detailed metrics and visualizations such as:

- **Optimized Parameters**: Best parameter sets identified through WFO and CPCV.
- **Performance Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, and more.
- **Cumulative Returns**: Percentage increase in portfolio value over time.
- **Drawdown Analysis**: Maximum drawdown and drawdown distribution.
- **Monte Carlo Simulation Results**: Simulated price paths and return distributions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

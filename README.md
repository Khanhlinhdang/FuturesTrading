# FuturesTrading

Several functions that are commonly used in futures trading

AutoTradingKit project: https://github.com/Khanhlinhdang/AutoTradingKit


This `PlotTrading` class provides  comprehensive capabilities for analyzing run-up and drawdown metrics for  both open and closed positions, with specific features for generating  plot data. The implementation includes:

1. **Position Management**:
   * Adding and tracking positions
   * Closing positions with PnL calculations
   * Storing price data for historical analysis
2. **Run-up and Drawdown Analysis**:
   * `calculate_position_run_up()`: Calculates the maximum unrealized profit a position has seen
   * `calculate_position_drawdown()`: Calculates the maximum unrealized loss a position has seen
   * Handling of both open and closed positions
3. **Performance Analysis**:
   * `analyze_position_performance()`: Comprehensive analysis including run-up, drawdown, and risk metrics
   * `analyze_all_positions()`: Portfolio-level analysis across all positions
   * Generation of equity curves and drawdown analysis
4. **Plotting Support**:
   * `generate_plot_data()`: Creates data formatted specifically for plotting charts
   * Time-series data for price, equity, PnL, and drawdown curves
   * Support for both individual positions and portfolio-level visualization
5. **Reporting**:
   * `generate_trade_report()`: Creates comprehensive trade reports
   * Configurable to include or exclude plot data for efficiency

The implementation handles different scenarios for data availability  (complete price history, entry/exit prices only, etc.) and provides  meaningful metrics even with limited data. All functions include  detailed English documentation explaining their purpose, parameters, and  return values

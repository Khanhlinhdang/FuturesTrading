import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.ticker import FuncFormatter

class TradingPlotter:
    """
    A class for visualizing trading performance metrics including run-up, drawdown,
    equity curves, and various performance analytics through plots and charts.
    
    This class provides tools for traders to:
    - Visualize position performance over time
    - Create equity curves with drawdown overlays
    - Generate heatmaps for trading performance by time/day
    - Analyze position entry/exit timing relative to run-up and drawdown
    - Compare actual trading results against optimal scenarios
    """
    
    def __init__(self, figsize=(12, 8), style='darkgrid', palette='viridis'):
        """
        Initialize the TradingPlotter with customizable visual parameters.
        
        Parameters:
        -----------
        figsize: tuple
            Default figure size for plots (width, height in inches)
        style: str
            Seaborn style theme ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        palette: str
            Color palette to use for plots
        """
        self.figsize = figsize
        sns.set_style(style)
        sns.set_palette(palette)
        self.colors = {
            'profit': '#4CAF50',         # Green
            'loss': '#F44336',           # Red
            'run_up': '#2196F3',         # Blue
            'drawdown': '#FF9800',       # Orange
            'entry': '#9C27B0',          # Purple
            'exit': '#607D8B',           # Blue Grey
            'equity': '#3F51B5',         # Indigo
            'optimal': '#00BCD4',        # Cyan
            'actual': '#FFC107'          # Amber
        }
        
        # Set default formatting for currency and percentage
        self.currency_formatter = FuncFormatter(lambda x, _: f'${x:,.2f}')
        self.percentage_formatter = FuncFormatter(lambda x, _: f'{x:.2f}%')
    
    def plot_position_performance(self, position_data, price_history, save_path=None):
        """
        Plot detailed performance metrics for a single trading position.
        
        Parameters:
        -----------
        position_data: dict
            A dictionary containing position performance metrics as returned by
            FuturesTrading.track_position_performance()
        price_history: list of dict
            List of price points with timestamps for the position's duration
        save_path: str or None
            If provided, saves the plot to this file path
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object containing the plot
        """
        # Extract key position data
        symbol = position_data.get('symbol', 'Unknown')
        side = position_data.get('side', 'Unknown')
        entry_price = position_data.get('entry_price', 0)
        exit_price = position_data.get('exit_price', None)
        entry_timestamp = position_data.get('entry_timestamp', None)
        is_closed = position_data.get('is_closed', False)
        
        # Convert price history to DataFrame for easier plotting
        if not price_history:
            return plt.figure(figsize=self.figsize)
            
        df = pd.DataFrame(price_history)
        df = df.sort_values('timestamp')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # Price chart with run-up and drawdown markers
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['timestamp'], df['price'], label='Price', color='black', linewidth=1.5)
        
        # Mark entry and exit points
        if entry_timestamp:
            ax1.axvline(x=entry_timestamp, color=self.colors['entry'], linestyle='--', 
                       linewidth=1, label='Entry')
            ax1.plot(entry_timestamp, entry_price, 'o', color=self.colors['entry'], markersize=8)
        
        if is_closed and exit_price:
            exit_timestamp = position_data.get('exit_timestamp', df['timestamp'].iloc[-1])
            ax1.axvline(x=exit_timestamp, color=self.colors['exit'], linestyle='--',
                       linewidth=1, label='Exit')
            ax1.plot(exit_timestamp, exit_price, 'o', color=self.colors['exit'], markersize=8)
        
        # Mark maximum run-up and drawdown points
        run_up_time = position_data.get('max_run_up_timestamp', None)
        drawdown_time = position_data.get('max_drawdown_timestamp', None)
        max_favorable_price = position_data.get('max_favorable_price', None)
        max_unfavorable_price = position_data.get('max_unfavorable_price', None)
        
        if run_up_time and max_favorable_price:
            ax1.plot(run_up_time, max_favorable_price, 'o', color=self.colors['run_up'], 
                    markersize=8, label='Max Run-up')
        
        if drawdown_time and max_unfavorable_price:
            ax1.plot(drawdown_time, max_unfavorable_price, 'o', color=self.colors['drawdown'], 
                    markersize=8, label='Max Drawdown')
        
        # Add horizontal lines for entry, exit, and optimal prices
        ax1.axhline(y=entry_price, color=self.colors['entry'], linestyle='-', alpha=0.3)
        if exit_price:
            ax1.axhline(y=exit_price, color=self.colors['exit'], linestyle='-', alpha=0.3)
        
        # Format the price chart
        ax1.set_title(f"{symbol} {side} Position Performance", fontsize=16)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.grid(True, alpha=0.3)
        
        # PnL/equity curve chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Create PnL data
        if 'pnl_path' in position_data and position_data['pnl_path']:
            pnl_df = pd.DataFrame(position_data['pnl_path'])
            ax2.plot(pnl_df['timestamp'], pnl_df['pnl'], label='PnL', 
                    color=self.colors['equity'], linewidth=1.5)
            
            # Color areas based on profit/loss
            ax2.fill_between(pnl_df['timestamp'], 0, pnl_df['pnl'], 
                            where=(pnl_df['pnl'] >= 0), color=self.colors['profit'], alpha=0.3)
            ax2.fill_between(pnl_df['timestamp'], 0, pnl_df['pnl'], 
                            where=(pnl_df['pnl'] < 0), color=self.colors['loss'], alpha=0.3)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('PnL', fontsize=12)
        
        # Format the PnL chart
        ax2.yaxis.set_major_formatter(self.currency_formatter)
        ax2.grid(True, alpha=0.3)
        
        # Run-up and Drawdown chart
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Create run-up and drawdown data if available
        if 'pnl_path' in position_data and position_data['pnl_path']:
            pnl_df['run_up'] = pnl_df['pnl'].cummax()
            pnl_df['drawdown'] = pnl_df['pnl'] - pnl_df['run_up']
            
            ax3.plot(pnl_df['timestamp'], pnl_df['run_up'], label='Run-up', 
                    color=self.colors['run_up'], linewidth=1.5)
            ax3.fill_between(pnl_df['timestamp'], 0, pnl_df['drawdown'], 
                            color=self.colors['drawdown'], alpha=0.5, label='Drawdown')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Run-up & Drawdown', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        
        # Format the Run-up/Drawdown chart
        ax3.yaxis.set_major_formatter(self.currency_formatter)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add performance summary text box
        max_run_up = position_data.get('max_run_up', 0)
        max_run_up_pct = position_data.get('max_run_up_percentage', 0)
        max_drawdown = position_data.get('max_drawdown', 0)
        max_drawdown_pct = position_data.get('max_drawdown_percentage', 0)
        final_pnl = position_data.get('final_pnl', 0)
        final_pnl_pct = position_data.get('final_pnl_percentage', 0)
        
        textbox_text = (
            f"Entry: ${entry_price:,.2f}\n"
            f"{'Exit' if is_closed else 'Current'}: ${exit_price:,.2f}\n"
            f"Max Run-up: ${max_run_up:,.2f} ({max_run_up_pct:.2f}%)\n"
            f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)\n"
            f"{'Final' if is_closed else 'Current'} PnL: ${final_pnl:,.2f} ({final_pnl_pct:.2f}%)"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax1.text(0.02, 0.98, textbox_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_equity_curve(self, equity_data, drawdown_data=None, save_path=None):
        """
        Plot an equity curve with optional drawdown overlay.
        
        Parameters:
        -----------
        equity_data: dict or pandas.DataFrame
            Either a DataFrame with 'timestamp' and 'equity' columns,
            or a dictionary with these keys
        drawdown_data: dict or pandas.DataFrame or None
            Either a DataFrame with 'timestamp' and 'drawdown' columns,
            or a dictionary with these keys, or None to skip drawdown plot
        save_path: str or None
            If provided, saves the plot to this file path
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object containing the plot
        """
        # Convert input data to DataFrames if necessary
        if isinstance(equity_data, dict):
            if 'timestamp' in equity_data and 'equity' in equity_data:
                equity_df = pd.DataFrame({
                    'timestamp': equity_data['timestamp'],
                    'equity': equity_data['equity']
                })
            else:
                # Assume it's a series of {timestamp: equity_value} pairs
                equity_df = pd.DataFrame({
                    'timestamp': list(equity_data.keys()),
                    'equity': list(equity_data.values())
                })
        else:
            equity_df = equity_data
            
        if drawdown_data is not None:
            if isinstance(drawdown_data, dict):
                if 'timestamp' in drawdown_data and 'drawdown' in drawdown_data:
                    drawdown_df = pd.DataFrame({
                        'timestamp': drawdown_data['timestamp'],
                        'drawdown': drawdown_data['drawdown']
                    })
                else:
                    # Assume it's a series of {timestamp: drawdown_value} pairs
                    drawdown_df = pd.DataFrame({
                        'timestamp': list(drawdown_data.keys()),
                        'drawdown': list(drawdown_data.values())
                    })
            else:
                drawdown_df = drawdown_data
        else:
            # Calculate drawdown from equity curve if not provided
            drawdown_df = pd.DataFrame({
                'timestamp': equity_df['timestamp'],
                'equity': equity_df['equity']
            })
            drawdown_df['peak'] = drawdown_df['equity'].cummax()
            drawdown_df['drawdown'] = -((drawdown_df['equity'] / drawdown_df['peak']) - 1) * 100
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], color=self.colors['equity'], linewidth=2)
        ax1.set_title('Account Equity Curve', fontsize=16)
        ax1.set_ylabel('Equity', fontsize=12)
        ax1.yaxis.set_major_formatter(self.currency_formatter)
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(drawdown_df['timestamp'], 0, drawdown_df['drawdown'], 
                        color=self.colors['drawdown'], alpha=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(bottom=0, top=max(drawdown_df['drawdown']) * 1.1 if len(drawdown_df) > 0 else 10)
        ax2.grid(True, alpha=0.3)
        
        # Add max drawdown line and annotation
        if len(drawdown_df) > 0:
            max_dd = drawdown_df['drawdown'].max()
            max_dd_date = drawdown_df.loc[drawdown_df['drawdown'].idxmax(), 'timestamp']
            
            ax2.axhline(y=max_dd, color='red', linestyle='--', alpha=0.7)
            ax2.text(equity_df['timestamp'].iloc[-1], max_dd, f'Max DD: {max_dd:.2f}%', 
                    verticalalignment='bottom', horizontalalignment='right', color='red')
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Add summary statistics
        if len(equity_df) > 0:
            initial_equity = equity_df['equity'].iloc[0]
            final_equity = equity_df['equity'].iloc[-1]
            total_return = ((final_equity / initial_equity) - 1) * 100
            
            annualized_return = 0
            if len(equity_df) > 1:
                days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
                if days > 0:
                    annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
            
            textbox_text = (
                f"Initial Equity: ${initial_equity:,.2f}\n"
                f"Final Equity: ${final_equity:,.2f}\n"
                f"Total Return: {total_return:.2f}%\n"
                f"Annualized Return: {annualized_return:.2f}%\n"
                f"Max Drawdown: {max_dd:.2f}%" if len(drawdown_df) > 0 else "Max Drawdown: N/A"
            )
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax1.text(0.02, 0.98, textbox_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_runup_drawdown_scatter(self, trades_data, save_path=None):
        """
        Create a scatter plot comparing maximum run-up vs drawdown for each trade.
        
        Parameters:
        -----------
        trades_data: list of dict
            List of trade performance dictionaries as returned by
            FuturesTrading.track_position_performance()
        save_path: str or None
            If provided, saves the plot to this file path
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object containing the plot
        """
        if not trades_data:
            return plt.figure(figsize=self.figsize)
        
        # Extract data for scatter plot
        run_ups = [trade.get('max_run_up_percentage', 0) for trade in trades_data]
        drawdowns = [trade.get('max_drawdown_percentage', 0) for trade in trades_data]
        final_pnls = [trade.get('final_pnl_percentage', 0) for trade in trades_data]
        symbols = [trade.get('symbol', 'Unknown') for trade in trades_data]
        sides = [trade.get('side', 'Unknown') for trade in trades_data]
        
        # Create colors based on final PnL
        colors = ['green' if pnl > 0 else 'red' for pnl in final_pnls]
        
        # Create size based on absolute PnL value (scaled)
        sizes = [abs(pnl) * 5 + 50 for pnl in final_pnls]  # Min size 50, scales with PnL
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create scatter plot
        scatter = ax.scatter(run_ups, drawdowns, c=colors, s=sizes, alpha=0.6, edgecolors='black')
        
        # Add diagonal line (equal run-up and drawdown)
        max_val = max(max(run_ups), max(drawdowns)) if run_ups and drawdowns else 10
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        # Add annotations for notable trades
        for i, (ru, dd, pnl, symbol, side) in enumerate(zip(run_ups, drawdowns, final_pnls, symbols, sides)):
            # Annotate trades with large run-ups, drawdowns, or PnLs
            if ru > np.percentile(run_ups, 75) or dd > np.percentile(drawdowns, 75) or abs(pnl) > np.percentile(np.abs(final_pnls), 75):
                ax.annotate(f"{symbol} {side}", 
                           xy=(ru, dd), 
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('Maximum Run-up (%)', fontsize=12)
        ax.set_ylabel('Maximum Drawdown (%)', fontsize=12)
        ax.set_title('Trade Performance: Maximum Run-up vs Maximum Drawdown', fontsize=16)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add colorbar legend
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn), ax=ax)
        cbar.set_label('PnL Direction (Green = Profit, Red = Loss)', fontsize=10)
        
        # Add explanatory text
        ax.text(0.02, 0.98, 
               "Bubble size represents PnL magnitude\n"
               "Trades above the diagonal line had higher drawdown than run-up\n"
               "Trades below the diagonal line had higher run-up than drawdown",
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_runup_drawdown_efficiency(self, trades_data, save_path=None):
        """
        Plot the efficiency of exits relative to maximum run-up and drawdown.
        
        Parameters:
        -----------
        trades_data: list of dict
            List of trade performance dictionaries as returned by
            FuturesTrading.track_position_performance()
        save_path: str or None
            If provided, saves the plot to this file path
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object containing the plot
        """
        if not trades_data:
            return plt.figure(figsize=self.figsize)
        
        # Filter for closed trades only
        closed_trades = [t for t in trades_data if t.get('is_closed', False)]
        
        if not closed_trades:
            return plt.figure(figsize=self.figsize)
        
        # Calculate efficiency metrics
        run_up_efficiency = []  # What percentage of maximum run-up was captured
        drawdown_efficiency = []  # What percentage of maximum drawdown was avoided
        symbols = []
        sides = []
        pnls = []
        
        for trade in closed_trades:
            max_run_up = trade.get('max_run_up', 0)
            max_drawdown = trade.get('max_drawdown', 0)
            final_pnl = trade.get('final_pnl', 0)
            
            # Calculate run-up efficiency (captured profit / maximum possible profit)
            if max_run_up > 0:
                ru_eff = min(final_pnl / max_run_up, 1) * 100  # Cap at 100%
            else:
                ru_eff = 0
            
            # Calculate drawdown efficiency (avoided loss / maximum possible loss)
            # Negative efficiency means final PnL is worse than max drawdown
            if max_drawdown > 0:
                dd_eff = (1 - max(0, -final_pnl) / max_drawdown) * 100
            else:
                dd_eff = 100  # Perfect efficiency if no drawdown
            
            run_up_efficiency.append(ru_eff)
            drawdown_efficiency.append(dd_eff)
            symbols.append(trade.get('symbol', 'Unknown'))
            sides.append(trade.get('side', 'Unknown'))
            pnls.append(final_pnl)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot run-up efficiency
        bars1 = ax1.bar(range(len(run_up_efficiency)), run_up_efficiency, 
                      color=[self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls])
        ax1.set_xlabel('Trade Number', fontsize=10)
        ax1.set_ylabel('Captured Run-up (%)', fontsize=10)
        ax1.set_title('Profit Capture Efficiency', fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add trade labels
        ax1.set_xticks(range(len(symbols)))
        ax1.set_xticklabels([f"{s} {d}" for s, d in zip(symbols, sides)], rotation=90, fontsize=8)
        
        # Plot drawdown efficiency
        bars2 = ax2.bar(range(len(drawdown_efficiency)), drawdown_efficiency,
                      color=[self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls])
        ax2.set_xlabel('Trade Number', fontsize=10)
        ax2.set_ylabel('Avoided Drawdown (%)', fontsize=10)
        ax2.set_title('Loss Avoidance Efficiency', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add trade labels
        ax2.set_xticks(range(len(symbols)))
        ax2.set_xticklabels([f"{s} {d}" for s, d in zip(symbols, sides)], rotation=90, fontsize=8)
        
        # Calculate average efficiencies
        avg_ru_eff = np.mean(run_up_efficiency)
        avg_dd_eff = np.mean(drawdown_efficiency)
        
        # Add average lines
        ax1.axhline(y=avg_ru_eff, color='black', linestyle='--', linewidth=1)
        ax1.text(len(run_up_efficiency)-1, avg_ru_eff, f' Avg: {avg_ru_eff:.1f}%', 
                fontsize=8, verticalalignment='bottom')
        
        ax2.axhline(y=avg_dd_eff, color='black', linestyle='--', linewidth=1)
        ax2.text(len(drawdown_efficiency)-1, avg_dd_eff, f' Avg: {avg_dd_eff:.1f}%', 
                fontsize=8, verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Add explanation text
        fig.text(0.5, 0.01, 
                "Profit Capture: Higher values mean exits were closer to optimal profit points.\n"
                "Loss Avoidance: Higher values mean exits successfully avoided potential losses.",
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_timing_heatmap(self, trades_data, save_path=None):
        """
        Create a heatmap showing trade performance by day of week and hour of day.
        
        Parameters:
        -----------
        trades_data: list of dict
            List of trade performance dictionaries as returned by
            FuturesTrading.track_position_performance()
        save_path: str or None
            If provided, saves the plot to this file path
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object containing the plot
        """
        if not trades_data:
            return plt.figure(figsize=self.figsize)
        
        # Filter for closed trades only
        closed_trades = [t for t in trades_data if t.get('is_closed', False)]
        
        if not closed_trades:
            return plt.figure(figsize=self.figsize)
        
        # Extract entry and exit timestamps and PnL
        entries = []
        exits = []
        pnls = []
        pnl_pcts = []
        
        for trade in closed_trades:
            if 'entry_timestamp' in trade and 'exit_timestamp' in trade:
                entries.append(trade['entry_timestamp'])
                exits.append(trade['exit_timestamp'])
                pnls.append(trade.get('final_pnl', 0))
                pnl_pcts.append(trade.get('final_pnl_percentage', 0))
        
        if not entries or not exits:
            return plt.figure(figsize=self.figsize)
        
        # Create DataFrames
        entry_df = pd.DataFrame({
            'timestamp': entries,
            'pnl': pnls,
            'pnl_pct': pnl_pcts,
            'type': 'entry'
        })
        
        exit_df = pd.DataFrame({
            'timestamp': exits,
            'pnl': pnls,
            'pnl_pct': pnl_pcts,
            'type': 'exit'
        })
        
        # Extract day of week and hour
        for df in [entry_df, exit_df]:
            df['day'] = df['timestamp'].dt.day_name()
            df['hour'] = df['timestamp'].dt.hour
            df['weekday'] = df['timestamp'].dt.weekday
        
        # Set specific order for days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create figure with two heatmaps side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Function to create heatmap data
        def create_heatmap_data(df, metric='count'):
            # Create pivot table
            if metric == 'count':
                pivot = df.pivot_table(index='day', columns='hour', aggfunc='size', fill_value=0)
            elif metric == 'pnl':
                pivot = df.pivot_table(index='day', columns='hour', values='pnl', aggfunc='mean', fill_value=0)
            elif metric == 'pnl_pct':
                pivot = df.pivot_table(index='day', columns='hour', values='pnl_pct', aggfunc='mean', fill_value=0)
            
            # Reindex to ensure all days are in correct order
            pivot = pivot.reindex(day_order)
            
            # Fill missing hours
            all_hours = list(range(24))
            pivot = pivot.reindex(columns=all_hours, fill_value=0)
            
            return pivot
        
        # Entry heatmap (count)
        entry_counts = create_heatmap_data(entry_df, 'count')
        sns.heatmap(entry_counts, ax=ax1, cmap='Blues', annot=True, fmt='g', cbar=True)
        ax1.set_title('Trade Entry Times (Count)', fontsize=14)
        ax1.set_xlabel('Hour of Day', fontsize=10)
        ax1.set_ylabel('Day of Week', fontsize=10)
        
        
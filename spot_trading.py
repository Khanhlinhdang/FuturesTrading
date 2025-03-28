import pandas as pd
import numpy as np
from datetime import datetime


class PlotTrading:
    """
    A comprehensive class for plot trading calculations including run-up, drawdown,
    profit/loss metrics, position management, and visualization data preparation.
    
    This class provides tools for traders to:
    - Calculate entry/exit metrics
    - Track position performance over time
    - Analyze maximum run-up and drawdown statistics
    - Generate data for plotting performance charts
    - Evaluate risk-adjusted returns
    """
    
    def __init__(self, maker_fee=0.0002, taker_fee=0.0004):
        """
        Initialize the PlotTrading class with exchange-specific parameters.
        
        Parameters:
        -----------
        maker_fee: float
            Fee rate applied when providing liquidity to the order book (limit orders)
        taker_fee: float
            Fee rate applied when taking liquidity from the order book (market orders)
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.positions = []
        self.trades_history = []
        self.price_data = {}  # Store price data by symbol
    
    def add_position(self, position):
        """
        Add a position to the tracking system.
        
        Parameters:
        -----------
        position: dict
            A dictionary containing position details including:
            - 'symbol': str - trading pair symbol
            - 'entry_price': float - entry price
            - 'position_size': float - size of the position
            - 'margin': float - margin allocated
            - 'leverage': float - leverage used
            - 'is_long': bool - long or short
            - 'entry_timestamp': datetime - when position was opened
            - 'status': str - 'open' or 'closed'
            
        Returns:
        --------
        dict: The position with added tracking ID
        """
        # Add a unique ID to the position if not present
        if 'id' not in position:
            position['id'] = len(self.positions) + 1
        
        self.positions.append(position)
        return position
    
    def update_position(self, position_id, updates):
        """
        Update an existing position with new data.
        
        Parameters:
        -----------
        position_id: int
            The ID of the position to update
        updates: dict
            Dictionary containing fields to update
            
        Returns:
        --------
        dict or None: The updated position or None if not found
        """
        for i, position in enumerate(self.positions):
            if position.get('id') == position_id:
                self.positions[i] = {**position, **updates}
                return self.positions[i]
        return None
    
    def close_position(self, position_id, exit_price, exit_timestamp=None, fee_type='taker'):
        """
        Close an open position and calculate realized PnL.
        
        Parameters:
        -----------
        position_id: int
            The ID of the position to close
        exit_price: float
            The price at which the position is closed
        exit_timestamp: datetime
            The time when the position was closed (defaults to current time if None)
        fee_type: str
            Type of fee applied ('taker' for market orders, 'maker' for limit orders)
            
        Returns:
        --------
        dict: The trade result details including PnL and percentage return
        """
        if exit_timestamp is None:
            exit_timestamp = datetime.now()
        
        # Find the position
        position = None
        position_index = -1
        for i, p in enumerate(self.positions):
            if p.get('id') == position_id and p['status'] == 'open':
                position = p
                position_index = i
                break
        
        if position is None:
            return {'error': 'Position not found or already closed'}
        
        # Calculate position value and fees
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        
        entry_value = position_size * entry_price
        exit_value = position_size * exit_price
        
        exit_fee_rate = self.taker_fee if fee_type == 'taker' else self.maker_fee
        exit_fee = exit_value * exit_fee_rate
        
        # Calculate PnL
        if is_long:
            pnl = exit_value - entry_value - position.get('entry_fee', 0) - exit_fee
        else:
            pnl = entry_value - exit_value - position.get('entry_fee', 0) - exit_fee
        
        pnl_percentage = (pnl / margin) * 100
        
        # Update the position
        updates = {
            'exit_price': exit_price,
            'exit_timestamp': exit_timestamp,
            'exit_fee': exit_fee,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'status': 'closed'
        }
        
        updated_position = self.update_position(position_id, updates)
        
        # Add to trade history
        trade_result = {**updated_position}
        self.trades_history.append(trade_result)
        
        return trade_result
    
    def add_price_data(self, symbol, timestamp, price, volume=None):
        """
        Add price data point for a symbol.
        
        Parameters:
        -----------
        symbol: str
            The trading pair symbol (e.g., 'BTC-USDT')
        timestamp: datetime
            The timestamp of the price data
        price: float
            The price at this timestamp
        volume: float or None
            The trading volume at this timestamp (if available)
            
        Returns:
        --------
        bool: True if data was added successfully
        """
        if symbol not in self.price_data:
            self.price_data[symbol] = []
        
        data_point = {
            'timestamp': timestamp,
            'price': price
        }
        
        if volume is not None:
            data_point['volume'] = volume
            
        self.price_data[symbol].append(data_point)
        
        # Sort by timestamp to maintain chronological order
        self.price_data[symbol].sort(key=lambda x: x['timestamp'])
        
        return True
    
    def add_price_data_batch(self, symbol, data_points):
        """
        Add multiple price data points for a symbol.
        
        Parameters:
        -----------
        symbol: str
            The trading pair symbol (e.g., 'BTC-USDT')
        data_points: list of dict
            List of price data points, each containing:
            - 'timestamp': datetime - the time of the price point
            - 'price': float - the price at that time
            - 'volume': float (optional) - the volume at that time
            
        Returns:
        --------
        int: Number of data points added
        """
        if symbol not in self.price_data:
            self.price_data[symbol] = []
            
        self.price_data[symbol].extend(data_points)
        
        # Sort by timestamp to maintain chronological order
        self.price_data[symbol].sort(key=lambda x: x['timestamp'])
        
        return len(data_points)
    
    def get_price_history(self, symbol, start_time=None, end_time=None):
        """
        Get price history for a symbol within a specified time range.
        
        Parameters:
        -----------
        symbol: str
            The trading pair symbol (e.g., 'BTC-USDT')
        start_time: datetime or None
            The start time of the range (inclusive)
        end_time: datetime or None
            The end time of the range (inclusive)
            
        Returns:
        --------
        list: The filtered price history data
        """
        if symbol not in self.price_data:
            return []
            
        price_history = self.price_data[symbol]
        
        if start_time is not None:
            price_history = [p for p in price_history if p['timestamp'] >= start_time]
            
        if end_time is not None:
            price_history = [p for p in price_history if p['timestamp'] <= end_time]
            
        return price_history
    
    def get_latest_price(self, symbol):
        """
        Get the latest available price for a symbol.
        
        Parameters:
        -----------
        symbol: str
            The trading pair symbol (e.g., 'BTC-USDT')
            
        Returns:
        --------
        float or None: The latest price or None if no data
        """
        if symbol not in self.price_data or not self.price_data[symbol]:
            return None
            
        # Assuming the data is sorted by timestamp
        latest_data_point = self.price_data[symbol][-1]
        return latest_data_point['price']
    
    def calculate_position_run_up(self, position_id, history=None):
        """
        Calculate the maximum run-up (unrealized profit) for a position.
        
        Parameters:
        -----------
        position_id: int
            The ID of the position to analyze
        history: list or None
            Optional custom price history to use
            If None, will use stored price data for the position's symbol
            
        Returns:
        --------
        dict: Run-up metrics including absolute and percentage values
        """
        # Find the position
        position = None
        for p in self.positions:
            if p.get('id') == position_id:
                position = p
                break
                
        if position is None:
            return {'error': 'Position not found'}
        
        symbol = position['symbol']
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        
        # Determine the appropriate price history
        if history is None:
            if position['status'] == 'closed' and 'exit_timestamp' in position:
                # For closed positions, get price history between entry and exit
                history = self.get_price_history(
                    symbol, 
                    position['entry_timestamp'], 
                    position['exit_timestamp']
                )
            else:
                # For open positions, get all price history since entry
                history = self.get_price_history(
                    symbol, 
                    position['entry_timestamp']
                )
        
        if not history:
            # If we still don't have price history, check if we have exit price for closed positions
            if position['status'] == 'closed' and 'exit_price' in position:
                if is_long:
                    highest_price = max(position['entry_price'], position['exit_price'])
                else:
                    highest_price = min(position['entry_price'], position['exit_price'])
                    
                # Calculate max run-up based on entry and exit prices only
                if is_long:
                    if highest_price <= entry_price:
                        max_run_up = 0
                    else:
                        max_run_up = position_size * (highest_price - entry_price)
                else:
                    if highest_price >= entry_price:
                        max_run_up = 0
                    else:
                        max_run_up = position_size * (entry_price - highest_price)
                
                max_run_up_percentage = (max_run_up / margin) * 100 if margin > 0 else 0
                
                return {
                    'max_run_up': max_run_up,
                    'max_run_up_percentage': max_run_up_percentage,
                    'max_favorable_price': highest_price,
                    'data_source': 'entry_exit_only'
                }
            else:
                # No price history and not a closed position
                return {'error': 'No price history available for this position'}
        
        # Initialize with entry price
        if is_long:
            highest_price = entry_price
        else:
            highest_price = entry_price
            
        max_run_up_timestamp = position['entry_timestamp']
        
        # Find the maximum favorable price movement
        for point in history:
            price = point['price']
            timestamp = point['timestamp']
            
            if is_long:
                # For long positions, we want the highest price
                if price > highest_price:
                    highest_price = price
                    max_run_up_timestamp = timestamp
            else:
                # For short positions, we want the lowest price
                if price < highest_price:
                    highest_price = price
                    max_run_up_timestamp = timestamp
        
        # Calculate maximum run-up
        if is_long:
            # For long positions, run-up is when price goes higher than entry
            if highest_price <= entry_price:
                max_run_up = 0
            else:
                max_run_up = position_size * (highest_price - entry_price)
        else:
            # For short positions, run-up is when price goes lower than entry
            if highest_price >= entry_price:
                max_run_up = 0
            else:
                max_run_up = position_size * (entry_price - highest_price)
        
        # Calculate run-up as a percentage of margin
        max_run_up_percentage = (max_run_up / margin) * 100 if margin > 0 else 0
        
        return {
            'max_run_up': max_run_up,
            'max_run_up_percentage': max_run_up_percentage,
            'max_favorable_price': highest_price,
            'max_run_up_timestamp': max_run_up_timestamp,
            'data_source': 'price_history'
        }
    
    def calculate_position_drawdown(self, position_id, history=None):
        """
        Calculate the maximum drawdown (unrealized loss) for a position.
        
        Parameters:
        -----------
        position_id: int
            The ID of the position to analyze
        history: list or None
            Optional custom price history to use
            If None, will use stored price data for the position's symbol
            
        Returns:
        --------
        dict: Drawdown metrics including absolute and percentage values
        """
        # Find the position
        position = None
        for p in self.positions:
            if p.get('id') == position_id:
                position = p
                break
                
        if position is None:
            return {'error': 'Position not found'}
        
        symbol = position['symbol']
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        
        # Determine the appropriate price history
        if history is None:
            if position['status'] == 'closed' and 'exit_timestamp' in position:
                # For closed positions, get price history between entry and exit
                history = self.get_price_history(
                    symbol, 
                    position['entry_timestamp'], 
                    position['exit_timestamp']
                )
            else:
                # For open positions, get all price history since entry
                history = self.get_price_history(
                    symbol, 
                    position['entry_timestamp']
                )
        
        if not history:
            # If we still don't have price history, check if we have exit price for closed positions
            if position['status'] == 'closed' and 'exit_price' in position:
                if is_long:
                    lowest_price = min(position['entry_price'], position['exit_price'])
                else:
                    lowest_price = max(position['entry_price'], position['exit_price'])
                    
                # Calculate max drawdown based on entry and exit prices only
                if is_long:
                    if lowest_price >= entry_price:
                        max_drawdown = 0
                    else:
                        max_drawdown = position_size * (entry_price - lowest_price)
                else:
                    if lowest_price <= entry_price:
                        max_drawdown = 0
                    else:
                        max_drawdown = position_size * (lowest_price - entry_price)
                
                max_drawdown_percentage = (max_drawdown / margin) * 100 if margin > 0 else 0
                
                return {
                    'max_drawdown': max_drawdown,
                    'max_drawdown_percentage': max_drawdown_percentage,
                    'max_unfavorable_price': lowest_price,
                    'data_source': 'entry_exit_only'
                }
            else:
                # No price history and not a closed position
                return {'error': 'No price history available for this position'}
        
        # Initialize with entry price
        if is_long:
            lowest_price = entry_price
        else:
            lowest_price = entry_price
            
        max_drawdown_timestamp = position['entry_timestamp']
        
        # Find the maximum unfavorable price movement
        for point in history:
            price = point['price']
            timestamp = point['timestamp']
            
            if is_long:
                # For long positions, we want the lowest price
                if price < lowest_price:
                    lowest_price = price
                    max_drawdown_timestamp = timestamp
            else:
                # For short positions, we want the highest price
                if price > lowest_price:
                    lowest_price = price
                    max_drawdown_timestamp = timestamp
        
        # Calculate maximum drawdown
        if is_long:
            # For long positions, drawdown is when price goes lower than entry
            if lowest_price >= entry_price:
                max_drawdown = 0
            else:
                max_drawdown = position_size * (entry_price - lowest_price)
        else:
            # For short positions, drawdown is when price goes higher than entry
            if lowest_price <= entry_price:
                max_drawdown = 0
            else:
                max_drawdown = position_size * (lowest_price - entry_price)
        
        # Calculate drawdown as a percentage of margin
        max_drawdown_percentage = (max_drawdown / margin) * 100 if margin > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown_percentage,
            'max_unfavorable_price': lowest_price,
            'max_drawdown_timestamp': max_drawdown_timestamp,
            'data_source': 'price_history'
        }
    
    def analyze_position_performance(self, position_id):
        """
        Perform comprehensive performance analysis for a position.
        
        Parameters:
        -----------
        position_id: int
            The ID of the position to analyze
            
        Returns:
        --------
        dict: Comprehensive performance metrics including run-up, drawdown, and plot data
        """
        # Find the position
        position = None
        for p in self.positions:
            if p.get('id') == position_id:
                position = p
                break
                
        if position is None:
            return {'error': 'Position not found'}
        
        symbol = position['symbol']
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        leverage = position.get('leverage', 1)
        
        # Get price history
        if position['status'] == 'closed' and 'exit_timestamp' in position:
            # For closed positions, get price history between entry and exit
            history = self.get_price_history(
                symbol, 
                position['entry_timestamp'], 
                position['exit_timestamp']
            )
            exit_price = position['exit_price']
            duration = (position['exit_timestamp'] - position['entry_timestamp']).total_seconds() / 3600  # Hours
        else:
            # For open positions, get all price history since entry
            history = self.get_price_history(
                symbol, 
                position['entry_timestamp']
            )
            exit_price = self.get_latest_price(symbol) if history else None
            if exit_price is None and history:
                exit_price = history[-1]['price']
            duration = (datetime.now() - position['entry_timestamp']).total_seconds() / 3600 if history else 0  # Hours
        
        # Calculate run-up and drawdown
        run_up_data = self.calculate_position_run_up(position_id, history)
        drawdown_data = self.calculate_position_drawdown(position_id, history)
        
        # Calculate current/final PnL
        if exit_price is not None:
            if is_long:
                pnl = position_size * (exit_price - entry_price)
            else:
                pnl = position_size * (entry_price - exit_price)
            
            pnl_percentage = (pnl / margin) * 100
        else:
            pnl = None
            pnl_percentage = None
        
        # Generate PnL curve for plotting
        pnl_curve = []
        equity_curve = []
        if history:
            for point in history:
                price = point['price']
                timestamp = point['timestamp']
                
                if is_long:
                    point_pnl = position_size * (price - entry_price)
                else:
                    point_pnl = position_size * (entry_price - price)
                
                point_pnl_percentage = (point_pnl / margin) * 100
                point_equity = margin + point_pnl
                
                pnl_curve.append({
                    'timestamp': timestamp,
                    'price': price,
                    'pnl': point_pnl,
                    'pnl_percentage': point_pnl_percentage
                })
                
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': point_equity
                })
        
        # Calculate maximum adverse excursion (MAE) and maximum favorable excursion (MFE)
        mae = drawdown_data.get('max_drawdown', 0)
        mfe = run_up_data.get('max_run_up', 0)
        mae_pct = drawdown_data.get('max_drawdown_percentage', 0)
        mfe_pct = run_up_data.get('max_run_up_percentage', 0)
        
        # Calculate risk-reward metrics
        if mae > 0:
            risk_reward_ratio = mfe / mae
        else:
            risk_reward_ratio = float('inf') if mfe > 0 else 0
            
        # Calculate Sharpe-like ratio (assuming zero risk-free rate)
        if pnl_curve:
            returns = [p['pnl_percentage'] for p in pnl_curve]
            avg_return = np.mean(returns) if returns else 0
            std_return = np.std(returns) if len(returns) > 1 else 1
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Value-at-Risk calculation (95% confidence)
        if pnl_curve and len(pnl_curve) > 5:
            returns = [p['pnl_percentage'] for p in pnl_curve]
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% confidence
        else:
            var_95 = -mae_pct  # Simplification using max drawdown as VaR
            
        # Data for plotting drawdown curve
        drawdown_curve = []
        if equity_curve:
            peak = equity_curve[0]['equity']
            for point in equity_curve:
                equity = point['equity']
                peak = max(peak, equity)
                dd = (equity - peak) / peak * 100
                
                drawdown_curve.append({
                    'timestamp': point['timestamp'],
                    'drawdown_percentage': dd
                })
        
        return {
            'position_id': position_id,
            'symbol': symbol,
            'side': 'Long' if is_long else 'Short',
            'status': position['status'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'margin': margin,
            'leverage': leverage,
            'duration_hours': duration,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'max_run_up': mfe,
            'max_run_up_percentage': mfe_pct,
            'max_run_up_price': run_up_data.get('max_favorable_price'),
            'max_run_up_timestamp': run_up_data.get('max_run_up_timestamp'),
            'max_drawdown': mae,
            'max_drawdown_percentage': mae_pct,
            'max_drawdown_price': drawdown_data.get('max_unfavorable_price'),
            'max_drawdown_timestamp': drawdown_data.get('max_drawdown_timestamp'),
            'risk_reward_ratio': risk_reward_ratio,
            'sharpe_ratio': sharpe_ratio,
            'value_at_risk_95': var_95,
            'plot_data': {
                'price_history': history,
                'pnl_curve': pnl_curve,
                'equity_curve': equity_curve,
                'drawdown_curve': drawdown_curve
            }
        }
    
    def analyze_all_positions(self):
        """
        Analyze all positions and generate comprehensive performance statistics.
        
        Returns:
        --------
        dict: Aggregated performance metrics and individual position analyses
        """
        all_analyses = []
        open_positions = []
        closed_positions = []
        
        for position in self.positions:
            position_id = position.get('id')
            if position_id is None:
                continue
                
            analysis = self.analyze_position_performance(position_id)
            if 'error' in analysis:
                continue
                
            all_analyses.append(analysis)
            
            if position['status'] == 'open':
                open_positions.append(analysis)
            else:
                closed_positions.append(analysis)
        
        # Calculate aggregate metrics
        total_positions = len(all_analyses)
        total_open = len(open_positions)
        total_closed = len(closed_positions)
        
        # Metrics for closed positions
        if closed_positions:
            winning_trades = sum(1 for p in closed_positions if p.get('pnl', 0) > 0)
            losing_trades = sum(1 for p in closed_positions if p.get('pnl', 0) <= 0)
            win_rate = winning_trades / total_closed if total_closed > 0 else 0
            
            total_profit = sum(p.get('pnl', 0) for p in closed_positions if p.get('pnl', 0) > 0)
            total_loss = sum(p.get('pnl', 0) for p in closed_positions if p.get('pnl', 0) <= 0)
            net_pnl = total_profit + total_loss
            
            avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            avg_run_up = np.mean([p.get('max_run_up_percentage', 0) for p in closed_positions])
            avg_drawdown = np.mean([p.get('max_drawdown_percentage', 0) for p in closed_positions])
            
            avg_duration = np.mean([p.get('duration_hours', 0) for p in closed_positions])
        else:
            winning_trades = losing_trades = win_rate = 0
            total_profit = total_loss = net_pnl = 0
            avg_profit = avg_loss = profit_factor = 0
            avg_run_up = avg_drawdown = avg_duration = 0
        
        # Current equity for open positions
        open_margin = sum(p.get('margin', 0) for p in open_positions)
        open_pnl = sum(p.get('pnl', 0) for p in open_positions)
        
        # Build equity curve from all closed positions
        equity_history = []
        if closed_positions:
            # Sort by exit timestamp
            sorted_positions = sorted(closed_positions, key=lambda p: p.get('exit_timestamp', datetime.now()))
            
            cumulative_pnl = 0
            for position in sorted_positions:
                cumulative_pnl += position.get('pnl', 0)
                equity_history.append({
                    'timestamp': position.get('exit_timestamp', datetime.now()),
                    'equity': cumulative_pnl
                })
        
        # Calculate drawdown on equity curve
        max_drawdown = 0
        max_drawdown_pct = 0
        if equity_history:
            peak = equity_history[0]['equity']
            for point in equity_history:
                equity = point['equity']
                peak = max(peak, equity)
                drawdown = peak - equity
                drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
        return {
            'summary': {
                'total_positions': total_positions,
                'open_positions': total_open,
                'closed_positions': total_closed,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_pnl': net_pnl,
                'profit_factor': profit_factor,
                'average_profit': avg_profit,
                'average_loss': avg_loss,
                'average_run_up_percentage': avg_run_up,
                'average_drawdown_percentage': avg_drawdown,
                'average_duration_hours': avg_duration,
                'open_positions_margin': open_margin,
                'open_positions_pnl': open_pnl,
                'max_system_drawdown': max_drawdown,
                'max_system_drawdown_percentage': max_drawdown_pct
            },
            'open_positions': open_positions,
            'closed_positions': closed_positions,
            'equity_history': equity_history
        }
    
    def generate_plot_data(self, position_id=None, include_indicators=False):
        """
        Generate data for plotting charts related to a position or all positions.
        
        Parameters:
        -----------
        position_id: int or None
            The ID of the position to generate plot data for, or None for all positions
        include_indicators: bool
            Whether to include technical indicators in the plot data
            
        Returns:
        --------
        dict: Data formatted for plotting charts
        """
        if position_id is not None:
            # Generate plot data for a specific position
            analysis = self.analyze_position_performance(position_id)
            if 'error' in analysis:
                return {'error': analysis['error']}
                
            return {
                'position': analysis,
                'plot_data': analysis['plot_data']
            }
        else:
            # Generate plot data for all positions
            aggregate_analysis = self.analyze_all_positions()
            
            # Merge price histories from all symbols
            all_price_data = {}
            for symbol in self.price_data:
                all_price_data[symbol] = self.price_data[symbol]
            
            return {
                'aggregate_analysis': aggregate_analysis,
                'price_data': all_price_data
            }
    
    def generate_trade_report(self, position_id=None, include_plot_data=True):
        """
        Generate a comprehensive trade report for a position or all positions.
        
        Parameters:
        -----------
        position_id: int or None
            The ID of the position to report on, or None for all positions
        include_plot_data: bool
            Whether to include plot data in the report
            
        Returns:
        --------
        dict: Comprehensive trade report data
        """
        if position_id is not None:
            # Generate report for a specific position
            analysis = self.analyze_position_performance(position_id)
            if 'error' in analysis:
                return {'error': analysis['error']}
            
            # Remove plot data if not requested
            if not include_plot_data and 'plot_data' in analysis:
                del analysis['plot_data']
                
            return {
                'type': 'single_position',
                'position': analysis
            }
        else:
            # Generate report for all positions
            aggregate_analysis = self.analyze_all_positions()
            
            # Remove plot data if not requested
            if not include_plot_data:
                for position in aggregate_analysis.get('open_positions', []):
                    if 'plot_data' in position:
                        del position['plot_data']
                for position in aggregate_analysis.get('closed_positions', []):
                    if 'plot_data' in position:
                        del position['plot_data']
            
            return {
                'type': 'aggregate',
                'analysis': aggregate_analysis
            }


# Example usage
if __name__ == "__main__":
    # Initialize the PlotTrading calculator
    plot_trading = PlotTrading(maker_fee=0.0002, taker_fee=0.0004)
    
    # Add price data for a symbol
    from datetime import datetime, timedelta
    
    # Generate some sample price data
    start_time = datetime.now() - timedelta(days=10)
    sample_prices = []
    
    base_price = 50000
    for i in range(240):  # 10 days of hourly data
        timestamp = start_time + timedelta(hours=i)
        
        # Create some price movement
        price_change = np.random.normal(0, 200)  # Random price change with normal distribution
        price = base_price + price_change
        base_price = price  # Update base price for next iteration
        
        sample_prices.append({
            'timestamp': timestamp,
            'price': price,
            'volume': np.random.uniform(10, 100)  # Random volume
        })
    
    # Add the price data to the system
    plot_trading.add_price_data_batch('BTC-USDT', sample_prices)
    
    # Open a position
    entry_time = start_time + timedelta(days=2)
    entry_price = next((p['price'] for p in sample_prices if p['timestamp'] >= entry_time), None)
    
    if entry_price:
        position = plot_trading.add_position({
            'symbol': 'BTC-USDT',
            'entry_price': entry_price,
            'position_size': 0.1,  # 0.1 BTC
            'margin': 1000,        # $1,000 margin
            'leverage': 5,         # 5x leverage
            'is_long': True,       # Long position
            'entry_timestamp': entry_time,
            'status': 'open'
        })
        
        # Close the position after 5 days
        exit_time = entry_time + timedelta(days=5)
        exit_price = next((p['price'] for p in sample_prices if p['timestamp'] >= exit_time), None)
        
        if exit_price:
            plot_trading.close_position(position['id'], exit_price, exit_time)
            
            # Analyze the position
            analysis = plot_trading.analyze_position_performance(position['id'])
            
            print(f"Position Analysis:")
            print(f"Entry Price: ${analysis['entry_price']:.2f}")
            print(f"Exit Price: ${analysis['exit_price']:.2f}")
            print(f"PnL: ${analysis['pnl']:.2f} ({analysis['pnl_percentage']:.2f}%)")
            print(f"Max Run-Up: ${analysis['max_run_up']:.2f} ({analysis['max_run_up_percentage']:.2f}%)")
            print(f"Max Drawdown: ${analysis['max_drawdown']:.2f} ({analysis['max_drawdown_percentage']:.2f}%)")
            print(f"Risk-Reward Ratio: {analysis['risk_reward_ratio']:.2f}")
            
            # Generate trade report
            report = plot_trading.generate_trade_report(position['id'], include_plot_data=False)
            
            print("\nTrade Report:")
            print(f"Duration: {report['position']['duration_hours']:.2f} hours")
            print(f"Sharpe Ratio: {report['position']['sharpe_ratio']:.2f}")
            print(f"Value at Risk (95%): {report['position']['value_at_risk_95']:.2f}%")
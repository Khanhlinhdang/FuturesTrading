import pandas as pd
import numpy as np
from datetime import datetime

class FuturesTrading:
    """
    A comprehensive class for futures trading calculations including liquidation prices,
    PnL calculations, position management, and portfolio analysis.
    
    This class provides tools for traders to:
    - Calculate entry/exit points
    - Determine liquidation prices
    - Track open positions and their performance
    - Analyze trading history and performance metrics
    - Calculate various risk metrics for futures trading
    """
    
    def __init__(self, maintenance_margin_rate=0.005, taker_fee=0.0004, maker_fee=0.0002):
        """
        Initialize the FuturesTrading class with exchange-specific parameters.
        
        Parameters:
        -----------
        maintenance_margin_rate: float
            The minimum margin ratio required to maintain positions (typically 0.5% or 0.005)
        taker_fee: float
            Fee rate applied when taking liquidity from the order book (market orders)
        maker_fee: float
            Fee rate applied when providing liquidity to the order book (limit orders)
        """
        self.maintenance_margin_rate = maintenance_margin_rate
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.positions = []
        self.trades_history = []
    
    def calculate_liquidation_price(self, entry_price, position_size, margin, leverage, is_long=True):
        """
        Calculate the exact liquidation price for a futures position.
        
        Parameters:
        -----------
        entry_price: float
            The price at which the position was entered
        position_size: float
            The size of the position in quantity of the base asset
        margin: float
            The amount of collateral allocated to this position
        leverage: float
            The leverage multiplier applied to the position
        is_long: bool
            True if this is a long position, False if short
            
        Returns:
        --------
        float: The price at which the position will be liquidated
        """
        position_value = position_size * entry_price
        initial_margin = position_value / leverage
        fee = position_value * self.taker_fee
        
        if is_long:
            bankruptcy_price = entry_price - initial_margin / position_size
            maintenance_margin = position_value * self.maintenance_margin_rate
            liquidation_price = (bankruptcy_price * position_size + maintenance_margin + fee) / position_size
        else:
            bankruptcy_price = entry_price + initial_margin / position_size
            maintenance_margin = position_value * self.maintenance_margin_rate
            liquidation_price = (bankruptcy_price * position_size - maintenance_margin - fee) / position_size
        
        return liquidation_price
    
    def calculate_pnl(self, entry_price, current_price, position_size, is_long=True):
        """
        Calculate basic PnL (Profit and Loss) without fees.
        
        Parameters:
        -----------
        entry_price: float
            The price at which the position was entered
        current_price: float
            The current market price or exit price
        position_size: float
            The size of the position in quantity of the base asset
        is_long: bool
            True if this is a long position, False if short
            
        Returns:
        --------
        float: The PnL value (positive for profit, negative for loss)
        """
        if is_long:
            pnl = position_size * (current_price - entry_price)
        else:
            pnl = position_size * (entry_price - current_price)
        
        return pnl
    
    def calculate_pnl_with_fees(self, entry_price, exit_price, position_size, is_long=True, 
                               entry_fee_type='taker', exit_fee_type='taker'):
        """
        Calculate PnL including transaction fees.
        
        Parameters:
        -----------
        entry_price: float
            The price at which the position was entered
        exit_price: float
            The price at which the position was or will be exited
        position_size: float
            The size of the position in quantity of the base asset
        is_long: bool
            True if this is a long position, False if short
        entry_fee_type: str
            Type of fee applied at entry ('taker' for market orders, 'maker' for limit orders)
        exit_fee_type: str
            Type of fee applied at exit ('taker' for market orders, 'maker' for limit orders)
            
        Returns:
        --------
        float: The PnL value after deducting transaction fees
        """
        entry_value = position_size * entry_price
        exit_value = position_size * exit_price
        
        entry_fee_rate = self.taker_fee if entry_fee_type == 'taker' else self.maker_fee
        exit_fee_rate = self.taker_fee if exit_fee_type == 'taker' else self.maker_fee
        
        entry_fee = entry_value * entry_fee_rate
        exit_fee = exit_value * exit_fee_rate
        
        if is_long:
            pnl = exit_value - entry_value - entry_fee - exit_fee
        else:
            pnl = entry_value - exit_value - entry_fee - exit_fee
        
        return pnl
    
    def calculate_pnl_percentage(self, pnl, initial_margin, leverage=1):
        """
        Calculate the percentage return on investment (ROI).
        
        Parameters:
        -----------
        pnl: float
            The profit or loss amount
        initial_margin: float
            The initial margin/collateral used for the position
        leverage: float
            The leverage multiplier applied to the position (for informational purposes)
            
        Returns:
        --------
        float: The percentage return on the initial margin
        """
        # ROI based on the actual capital deployed (initial margin)
        pnl_percentage = (pnl / initial_margin) * 100
        
        return pnl_percentage
    
    def calculate_unrealized_pnl(self, position, current_price):
        """
        Calculate the unrealized PnL for an open position.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
        current_price: float
            The current market price
            
        Returns:
        --------
        float: The unrealized PnL for the position
        """
        return self.calculate_pnl(
            position['entry_price'],
            current_price,
            position['position_size'],
            position['is_long']
        )
    
    def calculate_unrealized_pnl_percentage(self, position, current_price):
        """
        Calculate the unrealized PnL percentage for an open position.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
            - 'margin': float - the margin allocated to this position
        current_price: float
            The current market price
            
        Returns:
        --------
        float: The unrealized PnL percentage for the position
        """
        unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)
        return self.calculate_pnl_percentage(unrealized_pnl, position['margin'])
    
    def open_position(self, symbol, entry_price, position_size, margin, leverage, is_long=True, 
                      timestamp=None, fee_type='taker'):
        """
        Open a new futures position.
        
        Parameters:
        -----------
        symbol: str
            The trading pair symbol (e.g., 'BTC-USDT')
        entry_price: float
            The price at which the position is opened
        position_size: float
            The size of the position in quantity of the base asset
        margin: float
            The amount of collateral allocated to this position
        leverage: float
            The leverage multiplier applied to the position
        is_long: bool
            True if this is a long position, False if short
        timestamp: datetime
            The time when the position was opened (defaults to current time if None)
        fee_type: str
            Type of fee applied ('taker' for market orders, 'maker' for limit orders)
            
        Returns:
        --------
        dict: The created position details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        position_value = position_size * entry_price
        fee = position_value * (self.taker_fee if fee_type == 'taker' else self.maker_fee)
        
        liquidation_price = self.calculate_liquidation_price(
            entry_price, position_size, margin, leverage, is_long
        )
        
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'position_size': position_size,
            'margin': margin,
            'leverage': leverage,
            'is_long': is_long,
            'liquidation_price': liquidation_price,
            'entry_timestamp': timestamp,
            'entry_fee': fee,
            'status': 'open'
        }
        
        self.positions.append(position)
        return position
    
    def close_position(self, position_index, exit_price, timestamp=None, fee_type='taker'):
        """
        Close an open position and calculate realized PnL.
        
        Parameters:
        -----------
        position_index: int
            The index of the position in the positions list
        exit_price: float
            The price at which the position is closed
        timestamp: datetime
            The time when the position was closed (defaults to current time if None)
        fee_type: str
            Type of fee applied ('taker' for market orders, 'maker' for limit orders)
            
        Returns:
        --------
        dict: The trade result details including PnL and percentage return
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if position_index >= len(self.positions) or self.positions[position_index]['status'] != 'open':
            return None
        
        position = self.positions[position_index]
        position_value = position['position_size'] * exit_price
        exit_fee = position_value * (self.taker_fee if fee_type == 'taker' else self.maker_fee)
        
        pnl = self.calculate_pnl_with_fees(
            position['entry_price'], 
            exit_price, 
            position['position_size'], 
            position['is_long'],
            'taker' if position['entry_fee'] == position['position_size'] * position['entry_price'] * self.taker_fee else 'maker',
            fee_type
        )
        
        pnl_percentage = self.calculate_pnl_percentage(pnl, position['margin'], position['leverage'])
        
        trade_result = {
            'symbol': position['symbol'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'margin': position['margin'],
            'leverage': position['leverage'],
            'is_long': position['is_long'],
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': timestamp,
            'entry_fee': position['entry_fee'],
            'exit_fee': exit_fee,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage
        }
        
        self.trades_history.append(trade_result)
        self.positions[position_index]['status'] = 'closed'
        
        return trade_result
    
    def get_portfolio_summary(self, current_prices):
        """
        Get a summary of the current portfolio including all open positions.
        
        Parameters:
        -----------
        current_prices: dict
            A dictionary mapping symbols to their current market prices
            
        Returns:
        --------
        dict: Portfolio summary including total margin, total unrealized PnL, and all positions
        """
        active_positions = [p for p in self.positions if p['status'] == 'open']
        
        total_margin = sum(p['margin'] for p in active_positions)
        total_unrealized_pnl = 0
        
        positions_summary = []
        
        for position in active_positions:
            if position['symbol'] in current_prices:
                current_price = current_prices[position['symbol']]
                unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)
                
                unrealized_pnl_pct = self.calculate_pnl_percentage(
                    unrealized_pnl, position['margin'], position['leverage']
                )
                
                total_unrealized_pnl += unrealized_pnl
                
                positions_summary.append({
                    'symbol': position['symbol'],
                    'side': 'Long' if position['is_long'] else 'Short',
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'position_size': position['position_size'],
                    'liquidation_price': position['liquidation_price'],
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'roe': unrealized_pnl_pct  # Return on Equity, same as unrealized_pnl_pct
                })
        
        return {
            'total_margin': total_margin,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_pct': (total_unrealized_pnl / total_margin * 100) if total_margin > 0 else 0,
            'positions': positions_summary
        }
    
    def get_trading_statistics(self):
        """
        Analyze completed trades and generate trading performance statistics.
        
        Returns:
        --------
        dict: Comprehensive trading statistics including win rate, profit factor, etc.
        """
        if not self.trades_history:
            return "No trades in history"
        
        df = pd.DataFrame(self.trades_history)
        
        win_trades = df[df['pnl'] > 0]
        loss_trades = df[df['pnl'] <= 0]
        
        stats = {
            'total_trades': len(df),
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'win_rate': len(win_trades) / len(df) * 100 if len(df) > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'max_profit': df['pnl'].max(),
            'max_loss': df['pnl'].min(),
            'avg_profit': win_trades['pnl'].mean() if len(win_trades) > 0 else 0,
            'avg_loss': loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0,
            'profit_factor': abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0 else float('inf'),
            'total_fees': df['entry_fee'].sum() + df['exit_fee'].sum()
        }
        
        return stats
    
    def calculate_position_size(self, account_balance, risk_percentage, entry_price, 
                              stop_loss_price, leverage=1, fee_rate=None):
        """
        Calculate the optimal position size based on risk management parameters.
        
        Parameters:
        -----------
        account_balance: float
            The total account balance available for trading
        risk_percentage: float
            The percentage of account balance willing to risk on this trade (e.g., 1 for 1%)
        entry_price: float
            The planned entry price for the position
        stop_loss_price: float
            The planned stop loss price for the position
        leverage: float
            The leverage multiplier to be used
        fee_rate: float or None
            The transaction fee rate, if None will use taker_fee
            
        Returns:
        --------
        dict: Position size details including size, margin required, and risk metrics
        """
        if fee_rate is None:
            fee_rate = self.taker_fee
        
        # Convert risk percentage to decimal
        risk_decimal = risk_percentage / 100
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * risk_decimal
        
        # Determine if it's a long or short position
        is_long = entry_price < stop_loss_price
        
        # Calculate the price difference between entry and stop loss
        price_diff = abs(entry_price - stop_loss_price)
        
        # Calculate position size excluding fees
        raw_position_size = risk_amount / price_diff
        
        # Adjust for leverage
        leveraged_position_size = raw_position_size * leverage
        
        # Calculate position value and fees
        position_value = leveraged_position_size * entry_price
        entry_fee = position_value * fee_rate
        exit_fee = (leveraged_position_size * stop_loss_price) * fee_rate
        
        # Calculate required margin
        required_margin = position_value / leverage
        
        # Check if margin exceeds account balance
        if required_margin > account_balance:
            max_position_size = (account_balance * leverage) / entry_price
            return {
                'status': 'error',
                'message': 'Required margin exceeds account balance',
                'max_position_size': max_position_size,
                'required_margin': required_margin,
                'account_balance': account_balance
            }
        
        return {
            'status': 'success',
            'position_size': leveraged_position_size,
            'required_margin': required_margin,
            'risk_amount': risk_amount,
            'is_long': is_long,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'total_fees': entry_fee + exit_fee,
            'effective_risk_percentage': ((risk_amount + entry_fee + exit_fee) / account_balance) * 100
        }
    
    def calculate_effective_leverage(self, position, current_price):
        """
        Calculate the effective leverage of a position after price movements.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
            - 'margin': float - the margin allocated to this position
        current_price: float
            The current market price
            
        Returns:
        --------
        float: The current effective leverage of the position
        """
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        initial_margin = position['margin']
        
        # Calculate initial position value
        initial_position_value = position_size * entry_price
        
        # Calculate current position value
        current_position_value = position_size * current_price
        
        # Calculate unrealized PnL
        unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)
        
        # Current equity = initial margin + unrealized PnL
        current_equity = initial_margin + unrealized_pnl
        
        # Effective leverage = current position value / current equity
        if current_equity <= 0:
            return float('inf')  # Position would be liquidated
        
        effective_leverage = current_position_value / current_equity
        
        return effective_leverage
    
    def calculate_funding_payment(self, position, mark_price, funding_rate):
        """
        Calculate the funding payment for a perpetual futures contract.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
        mark_price: float
            The mark price used for funding calculations
        funding_rate: float
            The funding rate (positive or negative) expressed as a decimal
            
        Returns:
        --------
        float: The funding payment amount (positive means receiving, negative means paying)
        """
        position_size = position['position_size']
        is_long = position['is_long']
        
        position_value = position_size * mark_price
        
        # Long positions pay funding when rate is positive
        # Short positions pay funding when rate is negative
        if is_long:
            payment = -position_value * funding_rate
        else:
            payment = position_value * funding_rate
        
        return payment
    
    def calculate_adl_risk(self, position, current_price):
        """
        Calculate the Auto-Deleveraging (ADL) risk level for a position.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
            - 'margin': float - the margin allocated to this position
            - 'leverage': float - the leverage of the position
        current_price: float
            The current market price
            
        Returns:
        --------
        dict: ADL risk assessment including risk level (0-5) and factors
        """
        unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)
        current_margin = position['margin'] + unrealized_pnl
        
        # Calculate position value and maintenance margin
        position_value = position['position_size'] * current_price
        maintenance_margin = position_value * self.maintenance_margin_rate
        
        # Calculate margin ratio
        margin_ratio = current_margin / maintenance_margin if maintenance_margin > 0 else float('inf')
        
        # Calculate profit ratio (ROE)
        roe = (unrealized_pnl / position['margin']) * 100 if position['margin'] > 0 else 0
        
        # Determine ADL risk level (0-5, where 5 is highest risk)
        if margin_ratio <= 1.1:  # Very close to liquidation
            risk_level = 5
        elif margin_ratio <= 1.3:
            risk_level = 4
        elif margin_ratio <= 1.5:
            risk_level = 3
        elif margin_ratio <= 2:
            risk_level = 2
        elif roe > 10:  # Profitable positions with adequate margin
            risk_level = 1
        else:
            risk_level = 0
        
        return {
            'risk_level': risk_level,
            'margin_ratio': margin_ratio,
            'roe': roe,
            'current_margin': current_margin,
            'maintenance_margin': maintenance_margin,
            'unrealized_pnl': unrealized_pnl
        }
    
    def calculate_vwap_entry(self, prices, volumes, target_volume):
        """
        Calculate Volume-Weighted Average Price (VWAP) for a potential entry strategy.
        
        Parameters:
        -----------
        prices: list of float
            List of price points
        volumes: list of float
            List of trading volumes corresponding to each price point
        target_volume: float
            The target volume to accumulate
            
        Returns:
        --------
        dict: VWAP details including the weighted average price and total volume
        """
        if len(prices) != len(volumes):
            return {'error': 'Prices and volumes lists must be the same length'}
        
        total_volume = 0
        total_value = 0
        vwap_points = []
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            total_volume += volume
            total_value += price * volume
            
            current_vwap = total_value / total_volume if total_volume > 0 else 0
            vwap_points.append({'price': price, 'volume': volume, 'vwap': current_vwap})
            
            if total_volume >= target_volume:
                break
        
        final_vwap = total_value / total_volume if total_volume > 0 else 0
        
        return {
            'vwap': final_vwap,
            'accumulated_volume': total_volume,
            'target_achieved': total_volume >= target_volume,
            'vwap_points': vwap_points
        }
    
    def calculate_slippage(self, order_price, execution_price, is_long=True):
        """
        Calculate slippage for a futures trade.
        
        Parameters:
        -----------
        order_price: float
            The intended or quoted price for the order
        execution_price: float
            The actual execution price received
        is_long: bool
            True if this is a buy/long order, False if sell/short
            
        Returns:
        --------
        dict: Slippage details including absolute and percentage values
        """
        absolute_slippage = abs(execution_price - order_price)
        percentage_slippage = (absolute_slippage / order_price) * 100
        
        # For long positions, negative slippage is good (executed at a lower price than ordered)
        # For short positions, positive slippage is good (executed at a higher price than ordered)
        if is_long:
            is_favorable = execution_price < order_price
            direction = -1 if is_favorable else 1
        else:
            is_favorable = execution_price > order_price
            direction = 1 if is_favorable else -1
        
        # Apply direction to slippage values
        absolute_slippage *= direction
        percentage_slippage *= direction
        
        return {
            'absolute_slippage': absolute_slippage,
            'percentage_slippage': percentage_slippage,
            'is_favorable': is_favorable
        }
    
    def calculate_drawdown(self, equity_curve):
        """
        Calculate drawdown metrics from an equity curve.
        
        Parameters:
        -----------
        equity_curve: list of float
            List of equity values over time
            
        Returns:
        --------
        dict: Drawdown metrics including maximum drawdown percentage and duration
        """
        if not equity_curve:
            return {'error': 'Empty equity curve'}
        
        # Convert to numpy array for easier calculations
        equity = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate drawdown in absolute terms
        drawdown = equity - running_max
        
        # Calculate drawdown in percentage terms
        drawdown_pct = drawdown / running_max * 100
        
        # Find the maximum drawdown and its index
        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min()
        max_dd_idx = np.argmin(drawdown)
        
        # Find the peak before the deepest valley
        peak_idx = np.argmax(equity[:max_dd_idx+1]) if max_dd_idx > 0 else 0
        
        # Find recovery index (if any)
        try:
            recovery_idx = np.where(equity[max_dd_idx:] >= equity[peak_idx])[0][0] + max_dd_idx
            recovered = True
            recovery_duration = recovery_idx - max_dd_idx
        except IndexError:
            # No recovery point found
            recovery_idx = len(equity) - 1
            recovered = False
            recovery_duration = None
        
        # Calculate drawdown duration
        drawdown_duration = max_dd_idx - peak_idx
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown_pct,
            'drawdown_duration': drawdown_duration,
            'peak_index': peak_idx,
            'valley_index': max_dd_idx,
            'recovered': recovered,
            'recovery_duration': recovery_duration,
            'drawdown_series': drawdown.tolist(),
            'drawdown_percentage_series': drawdown_pct.tolist()
        }
    
    def calculate_optimal_leverage(self, volatility, target_risk=0.01, max_leverage=20):
        """
        Calculate the optimal leverage based on asset volatility and target risk level.
        
        Parameters:
        -----------
        volatility: float
            The daily volatility of the asset (standard deviation of daily returns)
        target_risk: float
            The target daily risk as a decimal (e.g., 0.01 for 1% daily risk)
        max_leverage: float
            The maximum leverage allowed by the exchange
            
        Returns:
        --------
        float: The optimal leverage level to use
        """
        # Kelly criterion adapted for futures trading
        if volatility <= 0:
            return max_leverage
        
        # Optimal leverage = target risk / volatility
        optimal_leverage = target_risk / volatility
        
        # Cap at maximum allowed leverage
        return min(optimal_leverage, max_leverage)
    
    def calculate_hedging_ratio(self, asset_correlation, asset1_volatility, asset2_volatility):
        """
        Calculate the optimal hedging ratio between two correlated assets.
        
        Parameters:
        -----------
        asset_correlation: float
            The correlation coefficient between the two assets (-1 to 1)
        asset1_volatility: float
            The volatility (standard deviation) of the first asset
        asset2_volatility: float
            The volatility (standard deviation) of the second asset
            
        Returns:
        --------
        float: The optimal hedging ratio (asset2/asset1)
        """
        # Optimal hedging ratio = correlation * (volatility1 / volatility2)
        if asset2_volatility <= 0:
            return 0
            
        hedging_ratio = asset_correlation * (asset1_volatility / asset2_volatility)
        
        return hedging_ratio
    
    def calculate_delta_neutral_position(self, spot_position, futures_price, spot_price):
        """
        Calculate the futures position size needed for a delta-neutral strategy.
        
        Parameters:
        -----------
        spot_position: float
            The size of the spot position (positive for long, negative for short)
        futures_price: float
            The current price of the futures contract
        spot_price: float
            The current spot price of the asset
            
        Returns:
        --------
        float: The required futures position size for delta neutrality
        """
        # For perfect delta neutrality, futures_position = -spot_position * (spot_price / futures_price)
        if futures_price <= 0:
            return 0
            
        futures_position = -spot_position * (spot_price / futures_price)
        
        return futures_position
    
    def calculate_position_risk_reward(self, entry_price, take_profit_price, stop_loss_price, is_long=True):
        """
        Calculate the risk-reward ratio for a planned trade.
        
        Parameters:
        -----------
        entry_price: float
            The planned entry price
        take_profit_price: float
            The target price for taking profit
        stop_loss_price: float
            The stop loss price
        is_long: bool
            True if this is a long position, False if short
            
        Returns:
        --------
        dict: Risk-reward metrics including ratio and absolute values
        """
        if is_long:
            potential_profit = take_profit_price - entry_price
            potential_loss = entry_price - stop_loss_price
        else:
            potential_profit = entry_price - take_profit_price
            potential_loss = stop_loss_price - entry_price
        
        if potential_loss <= 0:
            return {'error': 'Invalid stop loss placement'}
            
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else float('inf')
        
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'potential_profit': potential_profit,
            'potential_loss': potential_loss,
            'profit_percentage': (potential_profit / entry_price) * 100,
            'loss_percentage': (potential_loss / entry_price) * 100
        }
    
    def calculate_position_run_up(self, position, highest_price=None, current_price=None):
        """
        Calculate the maximum run-up (unrealized profit) for a position.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
            - 'margin': float - the margin allocated to this position
            - 'status': str - 'open' or 'closed'
        highest_price: float or None
            The highest price reached since entry (for long) or lowest price (for short)
            If None and position is open, current_price will be used
            If None and position is closed, highest_price must be provided
        current_price: float or None
            The current market price, used only for open positions
            If position is open and highest_price is None, this will be used as highest_price
            
        Returns:
        --------
        dict: Run-up metrics including absolute and percentage values
        """
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        is_open = position['status'] == 'open'
        
        # Determine the price to use for calculation
        if highest_price is None:
            if is_open and current_price is not None:
                # For open positions without historical high/low, use current price
                highest_price = current_price
            else:
                return {'error': 'Must provide highest_price for closed positions or current_price for open positions'}
        
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
            'max_favorable_price': highest_price
        }

    def calculate_position_drawdown(self, position, lowest_price=None, current_price=None):
        """
        Calculate the maximum drawdown (unrealized loss) for a position.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
            - 'margin': float - the margin allocated to this position
            - 'status': str - 'open' or 'closed'
        lowest_price: float or None
            The lowest price reached since entry (for long) or highest price (for short)
            If None and position is open, current_price will be used
            If None and position is closed, lowest_price must be provided
        current_price: float or None
            The current market price, used only for open positions
            If position is open and lowest_price is None, this will be used as lowest_price
            
        Returns:
        --------
        dict: Drawdown metrics including absolute and percentage values
        """
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        is_open = position['status'] == 'open'
        
        # Determine the price to use for calculation
        if lowest_price is None:
            if is_open and current_price is not None:
                # For open positions without historical low/high, use current price
                lowest_price = current_price
            else:
                return {'error': 'Must provide lowest_price for closed positions or current_price for open positions'}
        
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
            'max_unfavorable_price': lowest_price
        }

    def track_position_performance(self, position, price_history=None, current_price=None):
        """
        Track comprehensive performance metrics of a position including run-up and drawdown.
        
        Parameters:
        -----------
        position: dict
            A position dictionary containing at minimum:
            - 'entry_price': float - the entry price
            - 'position_size': float - the size of the position
            - 'is_long': bool - whether it's a long position
            - 'margin': float - the margin allocated to this position
            - 'status': str - 'open' or 'closed'
            - 'entry_timestamp': datetime - when the position was opened
        price_history: list of dict or None
            Historical price data with timestamps, each dict containing:
            - 'timestamp': datetime - the time of the price point
            - 'price': float - the price at that time
            If None, current_price will be used for open positions
        current_price: float or None
            The current market price, used only for open positions when price_history is None
            
        Returns:
        --------
        dict: Comprehensive performance metrics including run-up, drawdown, and duration
        """
        entry_price = position['entry_price']
        position_size = position['position_size']
        is_long = position['is_long']
        margin = position['margin']
        is_open = position['status'] == 'open'
        entry_timestamp = position['entry_timestamp']
        
        # If position is closed and has exit data, use it
        if not is_open and 'exit_price' in position and 'exit_timestamp' in position:
            exit_price = position['exit_price']
            exit_timestamp = position['exit_timestamp']
            duration = (exit_timestamp - entry_timestamp).total_seconds() / 86400  # Duration in days
        else:
            exit_price = current_price if current_price is not None else None
            exit_timestamp = datetime.now()
            duration = (exit_timestamp - entry_timestamp).total_seconds() / 86400  # Duration in days
        
        # Initialize variables for tracking price extremes
        highest_price = entry_price
        lowest_price = entry_price
        
        # Track when max run-up and drawdown occurred
        max_run_up_timestamp = entry_timestamp
        max_drawdown_timestamp = entry_timestamp
        
        # Track price path if history is provided
        price_path = []
        pnl_path = []
        
        if price_history:
            for point in price_history:
                timestamp = point['timestamp']
                price = point['price']
                
                # Skip price points before entry
                if timestamp < entry_timestamp:
                    continue
                    
                # Skip price points after exit for closed positions
                if not is_open and timestamp > exit_timestamp:
                    continue
                
                # Update highest and lowest prices
                if is_long:
                    if price > highest_price:
                        highest_price = price
                        max_run_up_timestamp = timestamp
                    if price < lowest_price:
                        lowest_price = price
                        max_drawdown_timestamp = timestamp
                else:
                    if price < lowest_price:
                        lowest_price = price
                        max_run_up_timestamp = timestamp
                    if price > highest_price:
                        highest_price = price
                        max_drawdown_timestamp = timestamp
                
                # Calculate PnL at this price point
                if is_long:
                    pnl = position_size * (price - entry_price)
                else:
                    pnl = position_size * (entry_price - price)
                
                # Track price and PnL over time
                price_path.append({'timestamp': timestamp, 'price': price})
                pnl_path.append({'timestamp': timestamp, 'pnl': pnl, 'pnl_percentage': (pnl / margin) * 100})
        else:
            # If no price history but current_price is provided for open positions
            if is_open and current_price is not None:
                highest_price = max(entry_price, current_price) if is_long else min(entry_price, current_price)
                lowest_price = min(entry_price, current_price) if is_long else max(entry_price, current_price)
            elif not is_open and exit_price is not None:
                # For closed positions without price history, use exit price
                highest_price = max(entry_price, exit_price) if is_long else min(entry_price, exit_price)
                lowest_price = min(entry_price, exit_price) if is_long else max(entry_price, exit_price)
            else:
                return {'error': 'Must provide price_history, current_price, or exit_price'}
        
        # Calculate run-up and drawdown
        run_up = self.calculate_position_run_up(position, highest_price)
        drawdown = self.calculate_position_drawdown(position, lowest_price)
        
        # Calculate final PnL if position is closed or current PnL if open
        if not is_open and exit_price is not None:
            if is_long:
                final_pnl = position_size * (exit_price - entry_price)
            else:
                final_pnl = position_size * (entry_price - exit_price)
            final_pnl_percentage = (final_pnl / margin) * 100
        elif is_open and current_price is not None:
            if is_long:
                final_pnl = position_size * (current_price - entry_price)
            else:
                final_pnl = position_size * (entry_price - current_price)
            final_pnl_percentage = (final_pnl / margin) * 100
        else:
            final_pnl = None
            final_pnl_percentage = None
        
        # Calculate maximum adverse excursion (MAE) and maximum favorable excursion (MFE)
        mae = drawdown['max_drawdown']
        mfe = run_up['max_run_up']
        mae_percentage = drawdown['max_drawdown_percentage']
        mfe_percentage = run_up['max_run_up_percentage']
        
        # Calculate Profit Factor from Run-up vs Drawdown
        profit_factor = mfe / mae if mae > 0 else float('inf')
        
        # Calculate Risk-adjusted return
        risk_adjusted_return = final_pnl_percentage / mae_percentage if mae_percentage > 0 else float('inf')
        
        return {
            'position_id': position.get('id', None),
            'symbol': position.get('symbol', None),
            'side': 'Long' if is_long else 'Short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'duration_days': duration,
            'max_run_up': mfe,
            'max_run_up_percentage': mfe_percentage,
            'max_run_up_timestamp': max_run_up_timestamp,
            'max_favorable_price': run_up['max_favorable_price'],
            'max_drawdown': mae,
            'max_drawdown_percentage': mae_percentage,
            'max_drawdown_timestamp': max_drawdown_timestamp,
            'max_unfavorable_price': drawdown['max_unfavorable_price'],
            'final_pnl': final_pnl,
            'final_pnl_percentage': final_pnl_percentage,
            'profit_factor': profit_factor,
            'risk_adjusted_return': risk_adjusted_return,
            'price_path': price_path,
            'pnl_path': pnl_path,
            'is_closed': not is_open
        }

    def track_all_positions_performance(self, current_prices=None, price_histories=None):
        """
        Track performance metrics for all positions, both open and closed.
        
        Parameters:
        -----------
        current_prices: dict or None
            A dictionary mapping symbols to their current market prices
            Required for calculating metrics for open positions
        price_histories: dict or None
            A dictionary mapping symbols to their price histories
            Each history should be a list of dictionaries with 'timestamp' and 'price'
            
        Returns:
        --------
        dict: Performance metrics for all positions grouped by status (open/closed)
        """
        open_positions_performance = []
        closed_positions_performance = []
        
        # Process all positions
        for position in self.positions:
            symbol = position.get('symbol', None)
            current_price = None if current_prices is None else current_prices.get(symbol, None)
            price_history = None if price_histories is None else price_histories.get(symbol, None)
            
            # Skip open positions without current price
            if position['status'] == 'open' and current_price is None:
                continue
            
            # Calculate performance metrics for this position
            performance = self.track_position_performance(position, price_history, current_price)
            
            if position['status'] == 'open':
                open_positions_performance.append(performance)
            else:
                closed_positions_performance.append(performance)
        
        # Calculate aggregate metrics
        total_winning_trades = sum(1 for p in closed_positions_performance if p.get('final_pnl', 0) > 0)
        total_losing_trades = sum(1 for p in closed_positions_performance if p.get('final_pnl', 0) <= 0)
        
        total_profit = sum(p.get('final_pnl', 0) for p in closed_positions_performance if p.get('final_pnl', 0) > 0)
        total_loss = sum(p.get('final_pnl', 0) for p in closed_positions_performance if p.get('final_pnl', 0) <= 0)
        
        avg_run_up = np.mean([p.get('max_run_up_percentage', 0) for p in closed_positions_performance]) if closed_positions_performance else 0
        avg_drawdown = np.mean([p.get('max_drawdown_percentage', 0) for p in closed_positions_performance]) if closed_positions_performance else 0
        
        return {
            'open_positions': open_positions_performance,
            'closed_positions': closed_positions_performance,
            'summary': {
                'total_positions': len(self.positions),
                'open_positions': len(open_positions_performance),
                'closed_positions': len(closed_positions_performance),
                'winning_trades': total_winning_trades,
                'losing_trades': total_losing_trades,
                'win_rate': total_winning_trades / len(closed_positions_performance) if closed_positions_performance else 0,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': total_profit + total_loss,
                'profit_factor': abs(total_profit / total_loss) if total_loss != 0 else float('inf'),
                'average_run_up_percentage': avg_run_up,
                'average_drawdown_percentage': avg_drawdown,
                'risk_reward_ratio': abs(avg_run_up / avg_drawdown) if avg_drawdown != 0 else float('inf')
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize the FuturesTrading calculator
    futures = FuturesTrading(maintenance_margin_rate=0.005, taker_fee=0.0004, maker_fee=0.0002)
    
    # Calculate position size based on risk management
    position_sizing = futures.calculate_position_size(
        account_balance=10000,    # $10,000 account
        risk_percentage=1,        # Risking 1% of account
        entry_price=50000,        # BTC at $50,000
        stop_loss_price=49000,    # Stop loss at $49,000
        leverage=5                # Using 5x leverage
    )
    
    print(f"Position Size: {position_sizing['position_size']} BTC")
    print(f"Required Margin: ${position_sizing['required_margin']}")
    print(f"Effective Risk: {position_sizing['effective_risk_percentage']}%")
    
    # Open a position
    btc_position = futures.open_position(
        symbol='BTC-USDT',
        entry_price=50000,
        position_size=position_sizing['position_size'],
        margin=position_sizing['required_margin'],
        leverage=5,
        is_long=True
    )
    
    print(f"Liquidation Price: ${btc_position['liquidation_price']:.2f}")
    
    # Calculate unrealized PnL at current market price
    current_price = 52000
    unrealized_pnl = futures.calculate_unrealized_pnl(btc_position, current_price)
    unrealized_pnl_pct = futures.calculate_unrealized_pnl_percentage(btc_position, current_price)
    
    print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
    print(f"Unrealized PnL %: {unrealized_pnl_pct:.2f}%")
    
    # Calculate effective leverage after price movement
    effective_leverage = futures.calculate_effective_leverage(btc_position, current_price)
    print(f"Effective Leverage: {effective_leverage:.2f}x")
    
    # Calculate ADL risk
    adl_risk = futures.calculate_adl_risk(btc_position, current_price)
    print(f"ADL Risk Level: {adl_risk['risk_level']} (0-5 scale)")
    
    # Calculate risk-reward ratio
    risk_reward = futures.calculate_position_risk_reward(
        entry_price=50000,
        take_profit_price=55000,
        stop_loss_price=49000,
        is_long=True
    )
    print(f"Risk-Reward Ratio: {risk_reward['risk_reward_ratio']:.2f}")
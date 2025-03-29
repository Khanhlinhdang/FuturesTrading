import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionType(Enum):
    SPOT = "spot"
    FUTURES = "futures"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"

maint_lookup_table = [
    (     50_000,  0.4,           0),
    (    250_000,  0.5,          50),
    (  1_000_000,  1.0,       1_300),
    ( 10_000_000,  2.5,      16_300),
    ( 20_000_000,  5.0,     266_300),
    ( 50_000_000, 10.0,   1_266_300),
    (100_000_000, 12.5,   2_516_300),
    (200_000_000, 15.0,   5_016_300),
    (300_000_000, 25.0,  25_016_300),
    (500_000_000, 50.0, 100_016_300),
]

class Position:
    def __init__(
        self,
        exchange: str,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: PositionSide,
        position_type: PositionType,
        leverage: float = 1.0,
        stop_loss: float = None,
        take_profit: float = None,
        open_fee: float = None,
        close_fee: float = None,
        id: str = None,
    ):
        self.id = id if id else str(uuid.uuid4())
        self.exchange = exchange
        self.symbol = symbol
        self.entry_price = entry_price
        self.max_price = entry_price
        self.min_price = entry_price
        self.quantity = quantity
        self.side = side # LONG or SHORT
        self.type = position_type  # spot or futures
        self.leverage = leverage if position_type == PositionType.FUTURES else 1.0
        self.position_size = self.quantity * self.entry_price * self.leverage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.liquidation_price = None
        self.risk_reward_ratio = None
        self.run_up = 0.0
        self.drawdown = 0.0
        self.liquidation_distance = None
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_percentage = 0.0
        self.realized_pnl = 0.0
        self.realized_pnl_percentage = 0.0
        self.status = PositionStatus.OPEN
        self.entry_time = datetime.now()
        self.closed_time = None
        self.open_fee = open_fee
        self.close_fee = close_fee

    def __str__(self):
        return (
            f"Position(id={self.id}, exchange={self.exchange}, symbol={self.symbol}, "
            f"side={self.side}, type={self.type}, entry_price={self.entry_price}, "
            f"quantity={self.quantity}, position_size={self.position_size}, leverage={self.leverage}, "
            f"status={self.status})"
        )

class PositionManager:
    def __init__(
        self,
        maintenance_margin_rate: float = 0.005,
        taker_fee: float = 0.0004,
        maker_fee: float = 0.0002,
    ):
        self.positions = {}
        self.maintenance_margin_rate = maintenance_margin_rate
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee

    def calculate_recommended_capital_by_risk(self, position: Position, total_capital, risk_percentage=1):
        """
        Calculates the recommended capital to allocate for a trade based on risk management parameters.
        Args:
            entry_price (float): The price at which the trade is entered.
            stop_loss_price (float): The price at which the stop-loss is set.
            total_capital (float): The total capital available for trading.
            risk_percentage (float): The percentage of total capital to risk on the trade.
            leverage (float): The leverage used for the trade.
            taker_fee (float): The taker fee rate (as a decimal) charged by the exchange.
        Returns:
            float: The recommended capital to allocate for the trade, rounded to two decimal places.
        Notes:
            - The function calculates the capital to risk based on the specified risk percentage.
            - It accounts for the price difference between the entry price and stop-loss price, 
            as well as the leverage and taker fees.
            - The effective loss per unit is calculated by combining the leveraged loss and 
            the impact of taker fees.
        """
        entry_price, stop_loss_price, leverage, taker_fee = position.entry_price, position.stop_loss, position.leverage, self.taker_fee
        capital_to_risk = total_capital * (risk_percentage / 100)
        price_difference = abs(entry_price - stop_loss_price)
        percent_change_sl = price_difference / entry_price
        leveraged_loss_per_unit = percent_change_sl * leverage
        total_taker_fee = entry_price * leverage * taker_fee * 2
        effective_loss_per_unit = leveraged_loss_per_unit + (total_taker_fee / (entry_price * leverage))
        recommended_capital = capital_to_risk / effective_loss_per_unit
        return recommended_capital

    def calculate_position_size_by_risk(self, position: Position, capital: float, risk_percentage: float) -> float:
        """
        Calculate the optimal position size based on risk management principles
        Args:
            position: The position object for which to calculate the size
            capital: The total available capital
            risk_percentage: The percentage of capital willing to risk (e.g., 1.0 for 1%)
            entry_price: The intended entry price
            stop_loss: The intended stop loss price
            leverage: The intended leverage (default: 1.0)
        Returns:
            The recommended position size (quantity)
        """
        entry_price, stop_loss, leverage = position.entry_price, position.stop_loss, position.leverage
        # Calculate the risk amount in absolute terms
        risk_amount = capital * (risk_percentage / 100)
        # Calculate the price difference between entry and stop loss
        price_difference = abs(entry_price - stop_loss)
        # Calculate the position size (quantity)
        if price_difference == 0:
            raise ValueError("Entry price cannot be equal to stop loss price")
        # Account for leverage in the calculation
        position_size = (risk_amount * leverage) / price_difference
        return position_size / entry_price  # Convert to quantity
    
    def _liq_balance(self,wallet_balance, contract_qty, entry_price):
        for max_position, maint_margin_rate_pct, maint_amount in maint_lookup_table:
            maint_margin_rate = maint_margin_rate_pct / 100
            liq_price = (wallet_balance + maint_amount - contract_qty*entry_price) / (abs(contract_qty) * (maint_margin_rate - (1 if contract_qty>=0 else -1)))
            base_balance = liq_price * abs(contract_qty)
            if base_balance <= max_position:
                break
        return liq_price


    def binance_calculate_liquidation_price(self, entry_price, contract_qty,leverage):
        wallet_balance = abs(contract_qty) * entry_price / leverage
        return self._liq_balance(wallet_balance, contract_qty, entry_price)


    def calculate_liquidation_price(self, entry_price, position_size, leverage, side):
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
        
        if side == PositionSide.LONG:
            bankruptcy_price = entry_price - initial_margin / position_size
            maintenance_margin = position_value * self.maintenance_margin_rate
            liquidation_price = (bankruptcy_price * position_size + maintenance_margin + fee) / position_size
        else:
            bankruptcy_price = entry_price + initial_margin / position_size
            maintenance_margin = position_value * self.maintenance_margin_rate
            liquidation_price = (bankruptcy_price * position_size - maintenance_margin - fee) / position_size
        
        return liquidation_price


    def calculate_open_fee(self, position_size: float, is_taker: bool = True, leverage: float = 1.0) -> float:
        """
        Calculate the fee for opening a position
        
        Args:
            position_size: The size of the position (price * quantity)
            is_taker: Whether the order is a taker order (True) or maker order (False)
            leverage: The leverage used for the position (default: 1.0)
            
        Returns:
            The fee amount for opening the position
        """
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        # Apply fee to the leveraged position size
        leveraged_position_size = position_size * leverage
        return leveraged_position_size * fee_rate

    def calculate_close_fee(self,position: Position, current_price: float,is_taker: bool = True) -> float:
        """
        Calculate the fee for closing a position
        
        Args:
            current_price: The current price at which the position is being closed
            quantity: The quantity being closed
            is_taker: Whether the order is a taker order (True) or maker order (False)
            leverage: The leverage used for the position (default: 1.0)
            
        Returns:
            The fee amount for closing the position
        """
        quantity, leverage = position.quantity, position.leverage
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        close_position_size = quantity * current_price
        # Apply fee to the leveraged closing position size
        leveraged_close_position_size = close_position_size * leverage
        return leveraged_close_position_size * fee_rate


    def open_position(
        self,
        exchange: str,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: PositionSide,
        position_type: PositionType,
        leverage: float = 1.0,
        stop_loss: float = None,
        take_profit: float = None,
        is_taker: bool = True,
        id: str = None,
    ) -> Position:
        """Open a new position"""
        # Validate the side
        if side not in [PositionSide.LONG, PositionSide.SHORT]:
            raise ValueError("Side must be either LONG or SHORT")
            
        # Validate the position type
        if position_type not in [PositionType.SPOT, PositionType.FUTURES]:
            raise ValueError("Position type must be either spot or futures")
            
        # For spot positions, leverage is always 1
        if position_type == PositionType.SPOT and leverage != 1.0:
            leverage = 1.0
            
        # Calculate position size and fees
        position_size = quantity * entry_price
        open_fee = self.calculate_open_fee(
            position_size=position_size,
            is_taker=is_taker,
            leverage=leverage
        )
        
        # Create a new position
        position = Position(
            exchange=exchange,
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            side=side,
            position_type=position_type,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            id=id,
            open_fee=open_fee,
            close_fee=None,  # Will be calculated at close time
        )
        
        # Calculate liquidation price for futures positions
        if position_type == PositionType.FUTURES and leverage > 1:
            position.liquidation_price = self.calculate_liquidation_price(
                entry_price, quantity, leverage, side
            )
            
        # Calculate risk-reward ratio if both stop loss and take profit are set
        if stop_loss and take_profit:
            position.risk_reward_ratio = self.calculate_position_risk_reward(position)
            
        # Store the position
        self.positions[position.id] = position
        
        return position

    def modify_position_stop_loss(self, position: Position, new_stop_loss: float) -> Position:
        """Modify the stop loss level of an existing position"""
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Cannot modify stop loss for {position.status} position")
            
        # Validate the new stop loss based on the position side
        if position.side == PositionSide.LONG and new_stop_loss >= position.entry_price:
            raise ValueError("Stop loss for LONG positions must be below entry price")
        elif position.side == PositionSide.SHORT and new_stop_loss <= position.entry_price:
            raise ValueError("Stop loss for SHORT positions must be above entry price")
            
        # Update the stop loss
        position.stop_loss = new_stop_loss
        
        # Recalculate risk-reward ratio if take profit is set
        if position.take_profit:
            position.risk_reward_ratio = self.calculate_position_risk_reward(position)
            
        return position

    def modify_position_take_profit(self, position: Position, new_take_profit: float) -> Position:
        """Modify the take profit level of an existing position"""
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Cannot modify take profit for {position.status} position")
            
        # Validate the new take profit based on the position side
        if position.side == PositionSide.LONG and new_take_profit <= position.entry_price:
            raise ValueError("Take profit for LONG positions must be above entry price")
        elif position.side == PositionSide.SHORT and new_take_profit >= position.entry_price:
            raise ValueError("Take profit for SHORT positions must be below entry price")
            
        # Update the take profit
        position.take_profit = new_take_profit
        
        # Recalculate risk-reward ratio if stop loss is set
        if position.stop_loss:
            position.risk_reward_ratio = self.calculate_position_risk_reward(position)
            
        return position

    def close_position(self, position: Position, exit_price: float, is_taker: bool = True) -> Position:
        """Close an existing position"""
        
        # Check if the position is already closed
        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Position is already {position.status}")
            
        # Calculate closing fee
        position.close_fee = self.calculate_close_fee(
            position=position,
            current_price=exit_price,
            is_taker=is_taker,
        )
        
        # Calculate realized PnL with fees
        position.realized_pnl = self.calculate_pnl_with_fees(
            position=position,
            exit_price=exit_price,
            open_fee=position.open_fee,
            close_fee=position.close_fee
        )
        position.unrealized_pnl = 0.0  # Reset unrealized PnL on close
        position.unrealized_pnl_percentage = 0.0  # Reset unrealized PnL percentage on close

        # Update position status and closed time
        position.status = PositionStatus.CLOSED
        position.closed_time = datetime.now()
        
        return position

    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get a position by its ID"""
        return self.positions.get(position_id)

    def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())

    def check_position_status(self, position_id: str) -> str:
        """Check the status of a position"""
        if position_id not in self.positions:
            raise ValueError(f"Position with ID {position_id} not found")
            
        return self.get_position_by_id(position_id).status

    def check_stop_loss_hit(self, position: Position, current_price: float) -> bool:
        """Check if the stop loss level has been hit"""
        # Check if the position has a stop loss set
        if not position.stop_loss:
            return False
            
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return False
            
        # Check if stop loss is hit based on position side
        if position.side == PositionSide.LONG and current_price <= position.stop_loss:
            return True
        elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
            return True
            
        return False

    def check_take_profit_hit(self, position: Position, current_price: float) -> bool:
        """Check if the take profit level has been hit"""
        # Check if the position has a take profit set
        if not position.take_profit:
            return False
            
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return False
            
        # Check if take profit is hit based on position side
        if position.side == PositionSide.LONG and current_price >= position.take_profit:
            return True
        elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
            return True
            
        return False

    def calculate_pnl(
        self, 
        entry_price: float, 
        exit_price: float, 
        quantity: float, 
        side: str,
        leverage: float = 1.0
    ) -> float:
        """
        Calculate the raw profit/loss for a position without fees
        
        Args:
            entry_price: The entry price of the position
            exit_price: The exit price (or current price for unrealized PnL)
            quantity: The position quantity
            side: The position side (LONG/SHORT)
            leverage: The position leverage (default: 1.0)
            
        Returns:
            The raw PnL value
        """
        if side == PositionSide.LONG:
            pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - exit_price) * quantity
            
        # Apply leverage for futures positions
        pnl = pnl * leverage
            
        return pnl

    def calculate_pnl_with_fees(
        self, 
        position: Position,
        exit_price: float, 
        open_fee: float = None,
        close_fee: float = None
    ) -> float:
        """
        Calculate the profit/loss for a position including fees
        
        Args:
            entry_price: The entry price of the position
            exit_price: The exit price (or current price for unrealized PnL)
            quantity: The position quantity
            side: The position side (LONG/SHORT)
            leverage: The position leverage (default: 1.0)
            open_fee: The fee paid when opening the position
            close_fee: The fee paid when closing the position
            
        Returns:
            The PnL value including fees
        """
        # Get the raw PnL
        entry_price, quantity, side, leverage = position.entry_price, position.quantity, position.side, position.leverage
        pnl = self.calculate_pnl(entry_price, exit_price, quantity, side, leverage)
        
        # If fees are not provided, calculate them
        if open_fee is None:
            position_size = quantity * entry_price
            open_fee = self.calculate_open_fee(
                position_size=position_size,
                is_taker=True,  # Assuming taker for opening
                leverage=leverage
            )
            
        if close_fee is None:
            close_fee = self.calculate_close_fee(
                position=position,
                current_price=exit_price,
                is_taker=True,  # Assuming taker for closing
            )
        
        # Subtract fees from the PnL
        pnl_with_fees = pnl - open_fee - close_fee
        
        return pnl_with_fees

    def calculate_pnl_percentage(
        self, 
        entry_price: float, 
        exit_price: float, 
        side: str,
        leverage: float = 1.0,
        open_fee: float = 0.0,
        close_fee: float = 0.0
    ) -> float:
        """
        Calculate the profit/loss percentage for a position
        
        Args:
            entry_price: The entry price of the position
            exit_price: The exit price (or current price for unrealized PnL)
            side: The position side (LONG/SHORT)
            leverage: The position leverage (default: 1.0)
            open_fee: The fee paid when opening the position
            close_fee: The fee paid when closing the position
            
        Returns:
            The PnL percentage value
        """
        # Calculate the percentage change
        if side == PositionSide.LONG:
            price_change_percentage = (exit_price - entry_price) / entry_price
        else:  # SHORT
            price_change_percentage = (entry_price - exit_price) / entry_price
            
        # Apply leverage
        raw_pnl_percentage = price_change_percentage * leverage
        
        # Apply fees as a percentage
        fees_percentage = (open_fee + close_fee) / (entry_price * 1.0)  # Assuming quantity = 1.0 for simplicity
        pnl_percentage = raw_pnl_percentage - fees_percentage
        
        return pnl_percentage * 100  # Convert to percentage

    def calculate_unrealized_pnl(
        self, 
        position: Position, 
        current_price: float,
        open_fee: float = None
    ) -> float:
        """
        Calculate the unrealized profit/loss for a position
        
        Args:
            position: The position object
            current_price: The current market price
            open_fee: Optional override for open fee
            
        Returns:
            The unrealized PnL value
        """
        
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return 0.0
            
        # Use provided leverage or position's leverage
        actual_leverage = position.leverage
        
        # Use provided open_fee or position's open_fee
        actual_open_fee = open_fee if open_fee is not None else position.open_fee
        
        # Calculate the unrealized PnL
        unrealized_pnl = self.calculate_pnl(
            position.entry_price, 
            current_price, 
            position.quantity, 
            position.side,
            actual_leverage
        )
        
        # Subtract the open fee if provided
        if actual_open_fee:
            unrealized_pnl -= actual_open_fee
            
        # Update the position's unrealized PnL
        position.unrealized_pnl = unrealized_pnl
        
        # Update max and min prices
        if current_price > position.max_price:
            position.max_price = current_price
        if current_price < position.min_price:
            position.min_price = current_price
            
        # Update run-up and drawdown
        self.calculate_position_run_up(position, current_price)
        self.calculate_position_drawdown(position, current_price)
        
        return unrealized_pnl

    def calculate_unrealized_pnl_percentage(
        self, 
        position: Position, 
        current_price: float,
        open_fee: float = None,
        close_fee: float = None
    ) -> float:
        """
        Calculate the unrealized profit/loss percentage for a position
        
        Args:
            position: The position object
            current_price: The current market price
            open_fee: Optional override for open fee
            close_fee: Estimated fee for closing the position
            
        Returns:
            The unrealized PnL percentage value
        """
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return 0.0
        # Use provided leverage or position's leverage
        actual_leverage = position.leverage
        # Use provided open_fee or position's open_fee
        actual_open_fee = open_fee if open_fee is not None else position.open_fee
        # Estimate close fee if not provided
        if close_fee is None:
            estimated_close_fee = self.calculate_close_fee(
                position=position,
                current_price=current_price,
                is_taker=True,  # Assuming taker for closing
            )
        else:
            estimated_close_fee = close_fee
        # Calculate the unrealized PnL percentage
        unrealized_pnl_percentage = self.calculate_pnl_percentage(
            position.entry_price, 
            current_price, 
            position.side,
            actual_leverage,
            actual_open_fee if actual_open_fee else 0.0,
            estimated_close_fee
        )
        return unrealized_pnl_percentage

    def calculate_position_run_up(self, position: Position, current_price: float) -> float:
        """
        Calculates the run-up of a trading position, which represents the highest 
        unrealized profit point achieved during the position's lifecycle.
        Args:
            position (Position): The trading position object containing details 
                such as entry price, max/min price, quantity, side, and leverage.
            current_price (float): The current market price of the asset.
        Returns:
            float: The highest unrealized profit (run-up) for the position.
        Notes:
            - For LONG positions, the run-up is determined by the maximum price 
              reached since the position was opened.
            - For SHORT positions, the run-up is determined by the minimum price 
              reached since the position was opened.
        """
        # Calculate the current unrealized PnL
        current_pnl = self.calculate_pnl(
            position.entry_price, current_price, position.quantity, position.side, position.leverage
        )
        # For LONG positions, max price gives max profit
        # For SHORT positions, min price gives max profit
        if position.side == PositionSide.LONG:
            max_pnl = self.calculate_pnl(
                position.entry_price, position.max_price, position.quantity, position.side, position.leverage
            )
        else:  # SHORT
            max_pnl = self.calculate_pnl(
                position.entry_price, position.min_price, position.quantity, position.side, position.leverage
            )
        # Run-up is the highest profit point
        position.run_up = max(max_pnl, current_pnl)
        return position.run_up

    def calculate_position_drawdown(self, position: Position, current_price: float) -> float:
        """
        Calculate the drawdown of a trading position based on the current price.
        Drawdown is defined as the lowest unrealized profit and loss (PnL) point 
        reached during the position's lifecycle.
        Args:
            position (Position): The trading position object containing details 
                such as entry price, quantity, side (LONG/SHORT), leverage, 
                and min/max price reached.
            current_price (float): The current market price of the asset.
        Returns:
            float: The drawdown value, representing the lowest unrealized PnL 
            (negative value) for the position.
        """
        # Calculate the current unrealized PnL
        current_pnl = self.calculate_pnl(
            position.entry_price, current_price, position.quantity, position.side, position.leverage
        )
        # For LONG positions, min price gives max loss
        # For SHORT positions, max price gives max loss
        if position.side == PositionSide.LONG:
            min_pnl = self.calculate_pnl(
                position.entry_price, position.min_price, position.quantity, position.side, position.leverage
            )
        else:  # SHORT
            min_pnl = self.calculate_pnl(
                position.entry_price, position.max_price, position.quantity, position.side, position.leverage
            )
        # Drawdown is the lowest loss point (negative value)
        position.drawdown = min(min_pnl, current_pnl)
        return position.drawdown

    def calculate_effective_leverage(self, position: Position, current_price: float) -> float:
        """
        Calculate the effective leverage of a position after price movements.
        
        Parameters:
        -----------
        position: Position
            A position object containing at minimum:
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
        entry_price = position.entry_price
        position_size = position.position_size
        position_value = position_size * entry_price
        initial_margin = position_value / position.leverage
        
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

    def calculate_position_risk_reward(self, position: Position) -> float:
        """
        Calculate the risk-reward ratio for a position
        
        Risk-reward ratio = Potential Reward / Potential Risk
        """
        # Calculate potential risk and reward based on position side
        entry_price = position.entry_price
        stop_loss = position.stop_loss
        take_profit = position.take_profit
        side = position.side
        if side == PositionSide.LONG:
            potential_risk = entry_price - stop_loss
            potential_reward = take_profit - entry_price
        else:  # SHORT
            potential_risk = stop_loss - entry_price
            potential_reward = entry_price - take_profit
            
        # Avoid division by zero
        if potential_risk == 0:
            raise ValueError("Risk cannot be zero (entry price equals stop loss)")
            
        risk_reward_ratio = potential_reward / potential_risk
        
        return risk_reward_ratio

    def calculate_risk_metrics(self, position: Position,current_price) -> Dict:
        """
        Calculate various risk metrics for a position
        
        Returns a dictionary with the following metrics:
        - unrealized_pnl
        - unrealized_pnl_percentage
        - drawdown
        - run_up
        - risk_reward_ratio
        - liquidation_distance (for futures positions)
        """
        # Calculate the risk metrics
        # Compile the metrics
        self.update_position(position,current_price)
        risk_metrics = {
            "unrealized_pnl": position.unrealized_pnl,
            "unrealized_pnl_percentage": position.unrealized_pnl_percentage,
            "drawdown": position.drawdown,
            "run_up": position.run_up,
            "risk_reward_ratio": position.risk_reward_ratio,
            "liquidation_distance": position.liquidation_distance,
        }
        
        return risk_metrics

    def update_position(self, position: Position, current_price: float) -> Position:
        """
        Update a position with the current market price
        
        This calculates and updates all dynamic fields of the position like:
        - unrealized PnL
        - max/min prices
        - run-up
        - drawdown
        """
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return position
            
        # Update the unrealized PnL
        position.unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)
        position.unrealized_pnl_percentage = self.calculate_unrealized_pnl_percentage(position, current_price)
        
        # Update max and min prices
        if current_price > position.max_price:
            position.max_price = current_price
        if current_price < position.min_price:
            position.min_price = current_price
            
        # Update run-up and drawdown
        position.run_up = self.calculate_position_run_up(position, current_price)
        position.drawdown = self.calculate_position_drawdown(position, current_price)

        if position.type == PositionType.FUTURES and position.leverage > 1 and position.liquidation_price:
            if position.side == PositionSide.LONG:
                liquidation_distance = (current_price - position.liquidation_price) / current_price * 100
            else:  # SHORT
                liquidation_distance = (position.liquidation_price - current_price) / current_price * 100
            position.liquidation_distance = liquidation_distance
        return position

    def auto_track_position(self, position: Position, current_price: float) -> Dict:
        """
        Automatically track and manage a position based on current price
        
        This function checks for:
        - Stop loss hit
        - Take profit hit
        - Liquidation price hit (for futures)
        
        Returns a dictionary with action taken and updated position
        """
        
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return {"action": "none", "position": position}
            
        # Update the position
        self.update_position(position, current_price)
        
        # Check for stop loss hit
        if self.check_stop_loss_hit(position, current_price):
            self.close_position(position, current_price)
            return {"action": "stop_loss", "position": position}
            
        # Check for take profit hit
        if self.check_take_profit_hit(position, current_price):
            self.close_position(position, current_price)
            return {"action": "take_profit", "position": position}
            
        # Check for liquidation price hit (for futures)
        if position.type == PositionType.FUTURES and position.liquidation_price:
            if (position.side == PositionSide.LONG and current_price <= position.liquidation_price) or \
               (position.side == PositionSide.SHORT and current_price >= position.liquidation_price):
                position.status = PositionStatus.LIQUIDATED
                position.closed_time = datetime.now()
                position.realized_pnl = -position.position_size / position.leverage  # Full loss of margin
                return {"action": "liquidation", "position": position}
                
        return {"action": "tracking", "position": position}

    def get_position_summary(self, position: Position, current_price: float = None) -> Dict:
        """
        Get a summary of a position
        
        Returns a dictionary with key information about the position
        """
        # Prepare the summary
        summary = {
            "id": position.id,
            "exchange": position.exchange,
            "symbol": position.symbol,
            "side": position.side,
            "type": position.type,
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "position_size": position.position_size,
            "leverage": position.leverage,
            "status": position.status,
            "entry_time": position.entry_time,
            "closed_time": position.closed_time,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "liquidation_price": position.liquidation_price,
            "risk_reward_ratio": position.risk_reward_ratio,
            "open_fee": position.open_fee,
            "close_fee": position.close_fee,
            "current_price": current_price,
            "unrealized_pnl": position.unrealized_pnl,
            "unrealized_pnl_percentage": position.unrealized_pnl_percentage,
            "run_up": position.run_up,
            "drawdown": position.drawdown,
            "realized_pnl": position.realized_pnl,
        }
        
        # Add current metrics if the position is still open and a current price is provided
        if position.status == PositionStatus.OPEN and current_price:
            summary.update({
                "unrealized_pnl": self.calculate_unrealized_pnl(position, current_price),
                "unrealized_pnl_percentage": self.calculate_unrealized_pnl_percentage(position, current_price),
            })
        elif position.status in [PositionStatus.CLOSED, PositionStatus.LIQUIDATED]:
            summary.update({
                "realized_pnl": position.realized_pnl,
                "unrealized_pnl": None,
                "unrealized_pnl_percentage": None,
            })
            
        return summary

    def get_all_position_summary(self, current_price_map: Dict[str, float] = None) -> List[Dict]:
        """
        Get a summary of all positions
        
        Args:
            current_price_map: A dictionary mapping symbols to their current prices
            
        Returns:
            A list of position summaries
        """
        summaries = []
        
        for _, position in self.positions.items():
            # Get the current price for the symbol if available
            current_price = None
            if current_price_map and position.symbol in current_price_map:
                current_price = current_price_map[position.symbol]
                
            # Get the position summary
            summary = self.get_position_summary(position, current_price)
            summaries.append(summary)
            
        return summaries

# Khởi tạo PositionManager với các tham số của sàn giao dịch
position_manager = PositionManager(
    maintenance_margin_rate=0.005,
    taker_fee=0.0004,
    maker_fee=0.0002
)

# Mở một vị thế LONG trên BTC-USDT
btc_position = position_manager.open_position(
    exchange="binance",
    symbol="BTC-USDT",
    entry_price=50000,
    quantity=0.5,
    side=PositionSide.LONG,
    position_type=PositionType.FUTURES,
    leverage=5.0,
    stop_loss=48000,
    take_profit=55000,
    is_taker=True
)
position_manager.update_position(btc_position, current_price=52000)
# In ra thông tin vị thế
print(position_manager.get_position_summary(btc_position))
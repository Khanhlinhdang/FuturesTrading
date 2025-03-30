import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionType(Enum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"

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
        entry_time: float,
        quantity: float,
        side: PositionSide,
        position_type: PositionType,
        leverage: float = 1.0,
        stop_loss: float = None,
        take_profit: float = None,
        open_fee: float = None,
        close_fee: float = None,
        id: str = None,
        strategy_id: str = None,
        total_capital: float = 0.0,
    ):
        self.id = id if id else str(uuid.uuid4())
        self.strategy_id = strategy_id
        self.exchange = exchange
        self.symbol = symbol
        self.entry_price = entry_price
        self.max_price = entry_price
        self.min_price = entry_price
        self.current_price = entry_price
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
        self.entry_time = entry_time
        self.closed_time = None
        self.total_capital = total_capital
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
        total_capital: float = 1000,
    ):
        self.positions: Dict[str, Position] = {}
        self.maintenance_margin_rate = maintenance_margin_rate
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.total_capital = total_capital

    def setup_variables(self, total_capital: float, maintenance_margin_rate: float, taker_fee: float, maker_fee: float):
        """Set up the initial variables for the PositionManager"""
        self.positions: Dict[str, Position] = {}
        self.maintenance_margin_rate = maintenance_margin_rate
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.total_capital = total_capital

    @property
    def total_capital(self) -> float:
        """Calculate the total capital available for trading"""
        return self.balance
    @total_capital.setter
    def total_capital(self, value: float):
        """Set the total capital available for trading"""
        self.balance = value

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
        return round(liquidation_price, 4)

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
        return round(leveraged_position_size * fee_rate,2)

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
        return round(leveraged_close_position_size * fee_rate, 2)


    def open_position(
        self,
        exchange: str,
        symbol: str,
        entry_price: float,
        entry_time: float,
        quantity: float,
        side: PositionSide,
        position_type: PositionType,
        leverage: float = 1.0,
        stop_loss: float = None,
        take_profit: float = None,
        is_taker: bool = True,
        id: str = None,
        strategy_id: str = None,
    ) -> Position:
        """
        Open a new trading position.
        Args:
            exchange (str): The name of the exchange where the position is opened.
            symbol (str): The trading pair symbol (e.g., "BTC/USDT").
            entry_price (float): The price at which the position is entered.
            entry_time (float): The timestamp of when the position is opened.
            quantity (float): The quantity of the asset being traded.
            side (PositionSide): The side of the position, either LONG or SHORT.
            position_type (PositionType): The type of position, either SPOT or FUTURES.
            leverage (float, optional): The leverage applied to the position. Defaults to 1.0.
            stop_loss (float, optional): The stop-loss price. Defaults to None.
            take_profit (float, optional): The take-profit price. Defaults to None.
            is_taker (bool, optional): Whether the position is opened as a taker. Defaults to True.
            id (str, optional): A unique identifier for the position. Defaults to None.
            strategy_id (str, optional): The identifier for the strategy associated with the position. Defaults to None.
        Returns:
            Position: The newly created position object.
        Raises:
            ValueError: If the side is not LONG or SHORT.
            ValueError: If the position type is not SPOT or FUTURES.
        Notes:
            - For SPOT positions, leverage is always set to 1.0.
            - Calculates the position size, open fees, and (if applicable) the liquidation price.
            - If both stop_loss and take_profit are provided, calculates the risk-reward ratio.
            - Stores the position in the internal positions dictionary.
        """
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
            entry_time=entry_time,
            quantity=quantity,
            side=side,
            position_type=position_type,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            id=id,
            strategy_id=strategy_id,
            open_fee=open_fee,
            close_fee=None,  # Will be calculated at close time
            total_capital=self.total_capital
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
        """
        Modify the stop loss level of an existing position.
        Args:
            position (Position): The position object to modify. Must have a status of OPEN.
            new_stop_loss (float): The new stop loss level to set. For LONG positions, 
                                   it must be below the entry price. For SHORT positions, 
                                   it must be above the entry price.
        Returns:
            Position: The updated position object with the modified stop loss level.
        Raises:
            ValueError: If the position is not OPEN.
            ValueError: If the new stop loss level is invalid based on the position side.
        Notes:
            - The method updates the stop loss level of the position and recalculates 
              the risk-reward ratio if a take profit level is set.
            - The `update_position` method is called to reflect the changes in the position.
        """
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

        self.update_position(position, position.current_price)
        
        # Recalculate risk-reward ratio if take profit is set
        if position.take_profit:
            position.risk_reward_ratio = self.calculate_position_risk_reward(position)
            
        return position

    def modify_position_take_profit(self, position: Position, new_take_profit: float) -> Position:
        """
        Modify the take profit level of an existing position.
        Args:
            position (Position): The position object to be modified. It must have a status of OPEN.
            new_take_profit (float): The new take profit level to set. For LONG positions, it must be 
                                     greater than the entry price. For SHORT positions, it must be 
                                     less than the entry price.
        Returns:
            Position: The updated position object with the modified take profit level.
        Raises:
            ValueError: If the position is not OPEN.
            ValueError: If the new take profit level is invalid based on the position side 
                        (e.g., below entry price for LONG or above entry price for SHORT).
        Notes:
            - The method updates the position's take profit level and recalculates the risk-reward 
              ratio if a stop loss is set.
            - The `update_position` method is called to reflect the changes in the position.
        """
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
        self.update_position(position, position.current_price)
        # Recalculate risk-reward ratio if stop loss is set
        if position.stop_loss:
            position.risk_reward_ratio = self.calculate_position_risk_reward(position)
            
        return position

    def close_position(self, position: Position, exit_price: float, current_time: int, is_taker: bool = True) -> Position:
        """
        Close an existing position.
        Args:
            position (Position): The position object to be closed. Must have a status of PositionStatus.OPEN.
            exit_price (float): The price at which the position is being closed.
            current_time (int): The timestamp at which the position is closed.
            is_taker (bool, optional): Indicates whether the closing order is a taker order. Defaults to True.
        Returns:
            Position: The updated position object with updated status, fees, realized PnL, and other attributes.
        Raises:
            ValueError: If the position is not in an open state.
        Notes:
            - Updates the position's status to PositionStatus.CLOSED.
            - Calculates and updates the closing fee, realized PnL, and total capital.
            - Resets unrealized PnL and unrealized PnL percentage to zero upon closing.
        """
        """Close an existing position"""
        
        # Check if the position is already closed
        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Position is already {position.status}")
        position.closed_time = current_time
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
        self.total_capital += position.realized_pnl
        position.total_capital = self.total_capital + position.realized_pnl - position.open_fee - position.close_fee
        position.unrealized_pnl = 0.0  # Reset unrealized PnL on close
        position.unrealized_pnl_percentage = 0.0  # Reset unrealized PnL percentage on close
        # Update position status and closed time
        position.status = PositionStatus.CLOSED
        
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
        """
        Check if the stop loss level has been hit for a given position.
        Args:
            position (Position): The trading position to check. It contains details 
                such as stop loss level, position side (LONG/SHORT), and status.
            current_price (float): The current market price of the asset.
        Returns:
            bool: True if the stop loss level has been hit and the position is still open, 
            otherwise False.
        Behavior:
            - If the position does not have a stop loss set, returns False.
            - If the position is not open, returns False.
            - For LONG positions, returns True if the current price is less than or equal 
              to the stop loss level.
            - For SHORT positions, returns True if the current price is greater than or 
              equal to the stop loss level.
        """
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
        """
        Check if the take profit level has been hit for a given position.
        Args:
            position (Position): The trading position to check.
            current_price (float): The current market price of the asset.
        Returns:
            bool: True if the take profit level has been hit and the position is still open, 
                  False otherwise.
        Notes:
            - For a LONG position, the take profit is hit if the current price is greater 
              than or equal to the take profit level.
            - For a SHORT position, the take profit is hit if the current price is less 
              than or equal to the take profit level.
            - If the position does not have a take profit set or is not open, the function 
              will return False.
        """
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

    def calculate_pnl(self, position: Position,current_price=None) -> float:
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
        entry_price, quantity, side, leverage = position.entry_price, position.quantity, position.side, position.leverage
        
        if not current_price:
            current_price = position.current_price
        if side == PositionSide.LONG:
            pnl = (current_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - current_price) * quantity
        # Apply leverage for futures positions
        pnl = pnl * leverage
        return round(pnl, 2)

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
        entry_price, quantity, leverage = position.entry_price, position.quantity, position.leverage
        pnl = self.calculate_pnl(position,exit_price)
        
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
        
        return round(pnl_with_fees, 2)

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
        
        return round(pnl_percentage * 100, 2)     # Convert to percentage

    def calculate_unrealized_pnl(self, position: Position, current_price: float,open_fee: float = None) -> float:
        """
        Calculate the unrealized profit and loss (PnL) for a given position based on the current price.
        Args:
            position (Position): The trading position for which to calculate the unrealized PnL.
            current_price (float): The current market price of the asset.
            open_fee (float, optional): The fee incurred when opening the position. If not provided, 
                the position's `open_fee` attribute will be used.
        Returns:
            float: The calculated unrealized PnL for the position.
        Notes:
            - If the position is not open (`PositionStatus.OPEN`), the unrealized PnL will be 0.0.
            - The unrealized PnL is adjusted by subtracting the open fee if provided.
            - Updates the position's `unrealized_pnl`, `max_price`, and `min_price` attributes.
            - Also updates the position's run-up and drawdown metrics using the current price.
        """
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return 0.0
        
        # Use provided open_fee or position's open_fee
        actual_open_fee = open_fee if open_fee is not None else position.open_fee
        
        # Calculate the unrealized PnL
        unrealized_pnl = self.calculate_pnl(position,current_price)
        
        # Subtract the open fee if provided
        if actual_open_fee:
            unrealized_pnl -= actual_open_fee
            
        # Update the position's unrealized PnL
        position.unrealized_pnl = unrealized_pnl
        
        return round(unrealized_pnl, 2)

    def calculate_unrealized_pnl_percentage(self, 
        position: Position, 
        current_price: float,
        open_fee: float = None,
        close_fee: float = None) -> float:
        """
        Calculate the unrealized profit and loss (PnL) percentage for a given position.

        Args:
            position (Position): The trading position for which the unrealized PnL percentage is calculated.
            current_price (float): The current market price of the asset.
            open_fee (float, optional): The fee incurred when opening the position. If not provided, 
                the position's `open_fee` will be used.
            close_fee (float, optional): The fee estimated for closing the position. If not provided, 
                it will be calculated using the `calculate_close_fee` method.

        Returns:
            float: The unrealized PnL percentage. Returns 0.0 if the position is not open.
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
        return round(unrealized_pnl_percentage, 2)

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
            position, current_price
        )
        # For LONG positions, max price gives max profit
        # For SHORT positions, min price gives max profit
        if position.side == PositionSide.LONG:
            max_pnl = self.calculate_pnl(
                position, position.max_price
            )
        else:  # SHORT
            max_pnl = self.calculate_pnl(
                position, position.min_price
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
        position.current_price = current_price
        current_pnl = self.calculate_pnl(position,current_price)

        # For LONG positions, min price gives max loss
        # For SHORT positions, max price gives max loss
        if position.side == PositionSide.LONG:
            min_pnl = self.calculate_pnl(
                position, position.min_price
            )
        else:  # SHORT
            min_pnl = self.calculate_pnl(
                position, position.max_price
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
        
        return round(effective_leverage, 2)

    def calculate_position_risk_reward(self, position: Position) -> float:
        """
        Calculate the risk-reward ratio for a given trading position.
        This method computes the potential risk and reward based on the entry price,
        stop loss, and take profit levels of the position. It also considers the 
        position side (LONG or SHORT) to determine the appropriate calculations.
        Args:
            position (Position): The trading position containing the entry price, 
                                 stop loss, take profit, and side (LONG or SHORT).
        Returns:
            float: The calculated risk-reward ratio.
        Raises:
            ValueError: If the potential risk is zero (entry price equals stop loss),
                        which would result in a division by zero.
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
        
        return round(risk_reward_ratio, 2)

    def calculate_risk_metrics(self, position: Position,current_price) -> Dict:
        """
        Calculate and return risk metrics for a given trading position.
        This method updates the position with the current price and computes
        various risk-related metrics such as unrealized profit and loss (PnL),
        drawdown, run-up, risk-reward ratio, and liquidation distance.
        Args:
            position (Position): The trading position object containing details
                about the current position.
            current_price (float): The current market price of the asset.
        Returns:
            Dict: A dictionary containing the following risk metrics:
                - "unrealized_pnl" (float): The unrealized profit or loss.
                - "unrealized_pnl_percentage" (float): The unrealized PnL as a percentage.
                - "drawdown" (float): The maximum observed loss from a peak.
                - "run_up" (float): The maximum observed gain from a trough.
                - "risk_reward_ratio" (float): The ratio of potential reward to risk.
                - "liquidation_distance" (float): The distance to the liquidation price.
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
        Updates the details of a given trading position based on the current market price.
        Args:
            position (Position): The trading position to update. Must be an instance of the Position class.
            current_price (float): The current market price of the asset.
        Returns:
            Position: The updated trading position with recalculated attributes.
        Updates:
            - `current_price`: Sets the current price of the position.
            - `unrealized_pnl`: Recalculates the unrealized profit and loss.
            - `unrealized_pnl_percentage`: Recalculates the unrealized profit and loss as a percentage.
            - `max_price`: Updates the maximum price reached by the position.
            - `min_price`: Updates the minimum price reached by the position.
            - `run_up`: Recalculates the position's run-up (maximum gain from entry price).
            - `drawdown`: Recalculates the position's drawdown (maximum loss from entry price).
            - `liquidation_distance`: For leveraged futures positions, calculates the percentage distance to the liquidation price.
        Notes:
            - If the position is not open (`PositionStatus.OPEN`), it is returned without modification.
            - For futures positions with leverage greater than 1, the liquidation distance is calculated based on the position's side (LONG or SHORT).
        """
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return position
        # Update the current price
        position.current_price = current_price
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
        # Calculate liquidation distance for futures positions

        if position.type == PositionType.FUTURES and position.leverage > 1 and position.liquidation_price:
            if position.side == PositionSide.LONG:
                liquidation_distance = (current_price - position.liquidation_price) / current_price * 100
            else:  # SHORT
                liquidation_distance = (position.liquidation_price - current_price) / current_price * 100
            position.liquidation_distance = liquidation_distance
        return position


    def auto_track_position(self, position: Position, current_price: float, current_time: int) -> Dict:
        """
        Automatically tracks the status of a trading position and determines the appropriate action 
        based on the current price and position parameters.

        Args:
            position (Position): The trading position to be tracked.
            current_price (float): The current market price of the asset.
            current_time (int): The current time for tracking purposes.

        Returns:
            Dict: A dictionary containing the action taken and the updated position object. 
                  Possible actions include:
                  - "none": No action taken as the position is not open.
                  - "stop_loss": Position closed due to stop loss being hit.
                  - "take_profit": Position closed due to take profit being hit.
                  - "liquidation": Position liquidated due to hitting the liquidation price.
                  - "tracking": Position is still being tracked with no action taken.

        Notes:
            - The function updates the position details based on the current price.
            - For futures positions, it checks if the liquidation price is hit and handles it accordingly.
            - Realized PnL is calculated as a full loss of margin in case of liquidation.
        """
        # Check if the position is still open
        if position.status != PositionStatus.OPEN:
            return {"action": "none", "position": position}
        # Update the position
        self.update_position(position, current_price)
        # Check for stop loss hit
        if self.check_stop_loss_hit(position, current_price):
            self.close_position(position, current_price,current_time)
            return {"action": "stop_loss", "position": position}
        # Check for take profit hit
        if self.check_take_profit_hit(position, current_price):
            self.close_position(position, current_price,current_time)
            return {"action": "take_profit", "position": position}
        # Check for liquidation price hit (for futures)
        if position.type == PositionType.FUTURES and position.liquidation_price:
            if (position.side == PositionSide.LONG and current_price <= position.liquidation_price) or \
               (position.side == PositionSide.SHORT and current_price >= position.liquidation_price):
                position.status = PositionStatus.LIQUIDATED
                position.closed_time = current_time
                position.realized_pnl = -position.position_size / position.leverage  # Full loss of margin
                return {"action": "liquidation", "position": position}
        position.total_capital = self.total_capital + position.unrealized_pnl - position.open_fee
        return {"action": "tracking", "position": position}

    def auto_track_all_positions_by_symbol(self, symbol: str, current_price: float, current_time: int) -> List[Dict]:
        """
        Automatically tracks all positions for a given trading symbol and determines the appropriate action
        based on the current price and position parameters.
        Args:
            symbol (str): The trading symbol to track.
            current_price (float): The current market price of the asset.
            current_time (float): The current time for tracking purposes.
        Returns:
            List[List]: A list of list information of each position.
        """
        # Get all positions for the given symbol
        positions = [pos for pos in self.get_all_positions() if pos.symbol == symbol]
        for position in positions:
            self.auto_track_position(position, current_price, current_time)
        # return self.get_all_position_summary_by_symbol(symbol)
    
    def auto_track_all_positions_by_strategy(self, strategy_id: str, current_price: float, current_time: int) -> List[Dict]:
        """
        Automatically tracks all positions for a given strategy ID and determines the appropriate action
        based on the current price and position parameters.
        Args:
            strategy_id (str): The strategy ID to track.
            current_price (float): The current market price of the asset.
            current_time (float): The current time for tracking purposes.
        Returns:
            List[List]: A list of list information of each position.
        """
        # Get all positions for the given strategy ID
        positions = [pos for pos in self.get_all_positions() if pos.strategy_id == strategy_id]
        for position in positions:
            self.auto_track_position(position, current_price, current_time)
        # return self.get_all_position_summary_by_strategy(strategy_id)
    
    def get_all_position_summary_by_strategy(self, strategy_id: str) -> List[List]:
        """
        Retrieves a summary of all positions managed by the position manager for a specific symbol.
        
        Args:
            symbol (str): The trading symbol for which to retrieve position summaries.
            current_price_map (Dict[str, float], optional): A dictionary mapping 
                symbols to their current prices. If provided, the current price 
                for each position's symbol will be used in the summary. Defaults to None.
                
        Returns:
            List[List]: A list of dictionaries, where each dictionary contains 
                the summary of a position for the specified symbol.
        """
        summaries = []
        
        for _, position in self.positions.items():
            if position.strategy_id == strategy_id:
                summary = self.get_position_summary(position)
                summaries.insert(0,summary) 
        return summaries

    def get_last_position_by_strategy(self, strategy_id: str) -> Optional[Position]:
        """
        Retrieves the last position for a given trading symbol.
        
        Args:
            strategy_id (str): The strategy ID for which to retrieve the last position.
            
        Returns:
            Position: The last position object for the specified strategy, or None if no positions exist.
        """
        positions = [pos for pos in self.get_all_positions() if pos.strategy_id == strategy_id]
        if positions:
            return positions[-1]
        return None


    def get_position_summary(self, position: Position, current_price: float = None) -> List:
        """
        Generates a summary of the given position, including its details and calculated metrics.
        Args:
            position (Position): The position object containing details about the trade.
            current_price (float, optional): The current market price of the asset. Defaults to None.
        Returns:
            List: A dictionary containing the position summary with the following keys:
                - id (str): The unique identifier of the position.
                - exchange (str): The exchange where the position is held.
                - symbol (str): The trading symbol of the asset.
                - side (str): The side of the position (e.g., 'long' or 'short').
                - type (str): The type of position (e.g., 'market' or 'limit').
                - entry_price (float): The price at which the position was entered.
                - quantity (float): The quantity of the asset in the position.
                - position_size (float): The size of the position in terms of value.
                - leverage (float): The leverage used for the position.
                - status (str): The current status of the position (e.g., 'open', 'closed', 'liquidated').
                - entry_time (datetime): The timestamp when the position was opened.
                - closed_time (datetime): The timestamp when the position was closed (if applicable).
                - stop_loss (float): The stop-loss price for the position.
                - take_profit (float): The take-profit price for the position.
                - liquidation_price (float): The liquidation price for the position.
                - risk_reward_ratio (float): The risk-reward ratio of the position.
                - open_fee (float): The fee incurred when opening the position.
                - close_fee (float): The fee incurred when closing the position.
                - current_price (float): The current market price of the asset (if provided).
                - unrealized_pnl (float): The unrealized profit or loss of the position.
                - unrealized_pnl_percentage (float): The unrealized profit or loss as a percentage.
                - run_up (float): The maximum favorable price movement since the position was opened.
                - drawdown (float): The maximum adverse price movement since the position was opened.
                - realized_pnl (float): The realized profit or loss of the position (if applicable).
        Notes:
            - If the position is open and a current price is provided, the unrealized PnL and 
              unrealized PnL percentage are recalculated based on the current price.
            - If the position is closed or liquidated, the realized PnL is included, and unrealized 
              PnL metrics are set to None.
        """
        # Prepare the summary
        summary = {
            "id": position.id,
            "exchange": position.exchange,
            "symbol": position.symbol,
            "side": position.side.value,
            "type": position.type.value,
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "position_size": position.position_size,
            "leverage": position.leverage,
            "status": position.status.value,
            "entry_time": position.entry_time,
            "closed_time": position.closed_time,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "liquidation_price": position.liquidation_price,
            "risk_reward_ratio": position.risk_reward_ratio,
            "open_fee": position.open_fee,
            "close_fee": position.close_fee,
            "current_price": current_price if current_price else position.current_price,
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
        # ["","Symbol", "Type", "Status", "Open Time", "Close Time", "Entry Price","Close Price", "Quantity", "Profit","Fee", "Cum.Profit", "Run-up", "Drawdown"]
        if position.closed_time:
            close_time = datetime.fromtimestamp(position.closed_time).strftime("%d.%m.%Y-%H:%M:%S")
        else:
            close_time = ""
        if position.realized_pnl:
            realized_pnl = position.realized_pnl
        else:
            realized_pnl = position.unrealized_pnl
        fee = position.open_fee 
        if position.close_fee:
            fee += position.close_fee
        return ["", position.symbol, position.side.value, position.status.value, datetime.fromtimestamp(position.entry_time).strftime("%d.%m.%Y-%H:%M:%S"), close_time, position.entry_price, position.current_price, position.quantity, realized_pnl, fee, position.total_capital, position.run_up, position.drawdown]
    
    def get_all_position_summary_by_symbol(self, symbol: str) -> List[List]:
        """
        Retrieves a summary of all positions managed by the position manager for a specific symbol.
        
        Args:
            symbol (str): The trading symbol for which to retrieve position summaries.
            current_price_map (Dict[str, float], optional): A dictionary mapping 
                symbols to their current prices. If provided, the current price 
                for each position's symbol will be used in the summary. Defaults to None.
                
        Returns:
            List[List]: A list of dictionaries, where each dictionary contains 
                the summary of a position for the specified symbol.
        """
        summaries = []
        
        for _, position in self.positions.items():
            if position.symbol == symbol:
                # Get the current price for the symbol if available
                summary = self.get_position_summary(position)
                summaries.insert(0,summary) 
        return summaries  # Return the summaries without reversing the order
    
    
    def get_all_position_summary(self, current_price_map: Dict[str, float] = None) -> List[Dict]:
        """
        Retrieves a summary of all positions managed by the position manager.
        Args:
            current_price_map (Dict[str, float], optional): A dictionary mapping 
                symbols to their current prices. If provided, the current price 
                for each position's symbol will be used in the summary. Defaults to None.
        Returns:
            List[Dict]: A list of dictionaries, where each dictionary contains 
                the summary of a position. The summary includes details such as 
                the position's symbol, quantity, and other relevant information.
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
    def clear_all_positions_by_symbol(self, symbol: str) -> None:
        """
        Clears all positions for a given trading symbol.
        Args:
            symbol (str): The trading symbol for which to clear positions.
        """
        # Remove all positions for the given symbol
        self.positions = {k: v for k, v in self.positions.items() if v.symbol != symbol}
    def clear_all_positions_by_strategy(self, strategy_id: str) -> None:
        """
        Clears all positions for a given trading strategy.
        Args:
            strategy_id (str): The trading strategy for which to clear positions.
        """
        # Remove all positions for the given strategy
        self.positions = {k: v for k, v in self.positions.items() if v.strategy_id != strategy_id}
    def clear_all_positions(self) -> None:
        """
        Clears all positions managed by the position manager.
        """
        # Remove all positions
        self.positions = {}


# Khởi tạo PositionManager với các tham số của sàn giao dịch
position_manager = PositionManager(
    maintenance_margin_rate=0.005,
    taker_fee=0.0004,
    maker_fee=0.0002,
    total_capital=10000
)


# Mở một vị thế LONG trên BTC-USDT
btc_position = position_manager.open_position(
    exchange="binance",
    symbol="BTC-USDT",
    entry_price=50000,
    entry_time=1633072800,  # Thời gian mở vị thế (timestamp)
    quantity=0.5,
    side=PositionSide.LONG,
    position_type=PositionType.FUTURES,
    leverage=5.0,
    stop_loss=48000,
    take_profit=55000,
    is_taker=True
)


# Cập nhật vị thế với giá hiện tại
# position_manager.auto_track_position(btc_position, current_price=51000)
# In ra thông tin vị thế
position_manager.modify_position_stop_loss(btc_position, new_stop_loss=49000)
position_manager.modify_position_take_profit(btc_position, new_take_profit=54000)
print(position_manager.get_position_summary(btc_position))

# position_manager.auto_track_position(btc_position, current_price=54001)
# print(position_manager.get_position_summary(btc_position))
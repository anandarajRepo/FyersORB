# strategy/order_manager.py

"""
Order Management System for ORB Strategy
Handles actual order placement with Fyers broker
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from fyers_apiv3 import fyersModel

from config.settings import FyersConfig, SignalType
from models.trading_models import Position, ORBSignal

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages actual order placement with Fyers broker"""

    def __init__(self, fyers_config: FyersConfig):
        self.fyers_config = fyers_config

        # Initialize Fyers model for order placement
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_config.client_id,
            token=fyers_config.access_token,
            log_path=""
        )

        # Order tracking
        self.placed_orders: Dict[str, Dict[str, Any]] = {}

    async def place_entry_order(self, position: Position) -> bool:
        """
        Place entry order for a new position

        Args:
            position: Position object with entry details

        Returns:
            bool: True if order placed successfully
        """
        try:
            # Determine order side
            side = 1 if position.signal_type == SignalType.LONG else -1

            # Prepare order data
            order_data = {
                "symbol": self._get_fyers_symbol(position.symbol),
                "qty": abs(position.quantity),
                "type": 2,  # Market order
                "side": side,
                "productType": "INTRADAY",  # Intraday for ORB
                "limitPrice": 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }

            logger.info(f"Placing entry order for {position.symbol}: "
                        f"Side={side}, Qty={abs(position.quantity)}, Price={position.entry_price}")

            # Place order via Fyers API
            response = self.fyers.place_order(data=order_data)

            if response and response.get('s') == 'ok':
                order_id = response.get('id')
                position.order_id = order_id

                # Track the order
                self.placed_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'ENTRY',
                    'position': position,
                    'response': response,
                    'timestamp': datetime.now()
                }

                logger.info(f"Entry order placed successfully - Order ID: {order_id}")
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Entry order failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error placing entry order for {position.symbol}: {e}")
            return False

    async def place_stop_loss_order(self, position: Position) -> bool:
        """
        Place stop loss order for position protection

        Args:
            position: Position object with stop loss details

        Returns:
            bool: True if order placed successfully
        """
        try:
            # Determine order side (opposite of entry)
            side = -1 if position.signal_type == SignalType.LONG else 1

            # Prepare stop loss order data
            order_data = {
                "symbol": self._get_fyers_symbol(position.symbol),
                "qty": abs(position.quantity),
                "type": 3,  # Stop loss order
                "side": side,
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": round(position.stop_loss),
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }

            logger.info(f"Placing stop loss order for {position.symbol}: "
                        f"Side={side}, Qty={abs(position.quantity)}, Stop={position.stop_loss}")

            # Place order via Fyers API
            response = self.fyers.place_order(data=order_data)

            if response and response.get('s') == 'ok':
                order_id = response.get('id')
                position.sl_order_id = order_id

                # Track the order
                self.placed_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'STOP_LOSS',
                    'position': position,
                    'response': response,
                    'timestamp': datetime.now()
                }

                logger.info(f"Stop loss order placed successfully - Order ID: {order_id}")
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Stop loss order failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error placing stop loss order for {position.symbol}: {e}")
            return False

    async def place_target_order(self, position: Position) -> bool:
        """
        Place target order for profit taking

        Args:
            position: Position object with target details

        Returns:
            bool: True if order placed successfully
        """
        try:
            # Determine order side (opposite of entry)
            side = -1 if position.signal_type == SignalType.LONG else 1

            # Prepare target order data
            order_data = {
                "symbol": self._get_fyers_symbol(position.symbol),
                "qty": abs(position.quantity),
                "type": 1,  # Limit order
                "side": side,
                "productType": "INTRADAY",
                "limitPrice": round(position.target_price),
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }

            logger.info(f"Placing target order for {position.symbol}: "
                        f"Side={side}, Qty={abs(position.quantity)}, Target={position.target_price}")

            # Place order via Fyers API
            response = self.fyers.place_order(data=order_data)

            if response and response.get('s') == 'ok':
                order_id = response.get('id')
                position.target_order_id = order_id

                # Track the order
                self.placed_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'TARGET',
                    'position': position,
                    'response': response,
                    'timestamp': datetime.now()
                }

                logger.info(f"Target order placed successfully - Order ID: {order_id}")
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Target order failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error placing target order for {position.symbol}: {e}")
            return False

    async def place_exit_order(self, position: Position, exit_price: float) -> bool:
        """
        Place market order to exit position

        Args:
            position: Position to exit
            exit_price: Current market price (for logging)

        Returns:
            bool: True if order placed successfully
        """
        try:
            # Determine order side (opposite of entry)
            side = -1 if position.signal_type == SignalType.LONG else 1

            # Prepare market exit order
            order_data = {
                "symbol": self._get_fyers_symbol(position.symbol),
                "qty": abs(position.quantity),
                "type": 2,  # Market order
                "side": side,
                "productType": "INTRADAY",
                "limitPrice": 0,  # Use current price as limit
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }

            logger.info(f"Placing exit order for {position.symbol}: "
                        f"Side={side}, Qty={abs(position.quantity)}, Price={exit_price}")

            # Place order via Fyers API
            response = self.fyers.place_order(data=order_data)

            if response and response.get('s') == 'ok':
                order_id = response.get('id')

                # Track the order
                self.placed_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'EXIT',
                    'position': position,
                    'response': response,
                    'timestamp': datetime.now()
                }

                logger.info(f"Exit order placed successfully - Order ID: {order_id}")

                # Cancel any pending SL/Target orders
                await self.cancel_pending_orders(position)

                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Exit order failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error placing exit order for {position.symbol}: {e}")
            return False

    async def modify_stop_loss(self, position: Position, new_stop: float) -> bool:
        """
        Modify stop loss order (for trailing stops)

        Args:
            position: Position with stop loss order
            new_stop: New stop loss price

        Returns:
            bool: True if modification successful
        """
        try:
            if not position.sl_order_id:
                logger.warning(f"No stop loss order to modify for {position.symbol}")
                return False

            # Prepare modification data
            modify_data = {
                "id": position.sl_order_id,
                "type": 3,  # Stop loss order
                "stopPrice": new_stop,
                "qty": abs(position.quantity),
                "limitPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }

            logger.info(f"Modifying stop loss for {position.symbol}: "
                        f"Order ID={position.sl_order_id}, New Stop={new_stop}")

            # Modify order via Fyers API
            response = self.fyers.modify_order(data=modify_data)

            if response and response.get('s') == 'ok':
                logger.info(f"Stop loss modified successfully")
                position.current_stop_loss = new_stop
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Stop loss modification failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error modifying stop loss for {position.symbol}: {e}")
            return False

    async def cancel_pending_orders(self, position: Position) -> bool:
        """
        Cancel all pending orders for a position

        Args:
            position: Position with orders to cancel

        Returns:
            bool: True if cancellation successful
        """
        try:
            cancelled_count = 0

            # Cancel stop loss order
            if position.sl_order_id:
                if await self.cancel_order(position.sl_order_id):
                    cancelled_count += 1
                    logger.info(f"Cancelled SL order: {position.sl_order_id}")

            # Cancel target order
            if position.target_order_id:
                if await self.cancel_order(position.target_order_id):
                    cancelled_count += 1
                    logger.info(f"Cancelled target order: {position.target_order_id}")

            logger.info(f"Cancelled {cancelled_count} pending orders for {position.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling pending orders for {position.symbol}: {e}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order

        Args:
            order_id: Fyers order ID to cancel

        Returns:
            bool: True if cancellation successful
        """
        try:
            cancel_data = {"id": order_id}

            response = self.fyers.cancel_order(data=cancel_data)

            if response and response.get('s') == 'ok':
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.warning(f"Order cancellation failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a placed order

        Args:
            order_id: Fyers order ID

        Returns:
            Order status dictionary or None
        """
        try:
            # Query order status from Fyers
            response = self.fyers.orderbook()

            if response and response.get('s') == 'ok':
                orders = response.get('orderBook', [])

                for order in orders:
                    if order.get('id') == order_id:
                        return order

            return None

        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None

    def _get_fyers_symbol(self, symbol: str) -> str:
        """
        Convert display symbol to Fyers trading symbol format

        Args:
            symbol: Display symbol (e.g., 'RELIANCE')

        Returns:
            Fyers format symbol (e.g., 'NSE:RELIANCE-EQ')
        """
        from config.symbols import convert_to_fyers_format

        fyers_symbol = convert_to_fyers_format(symbol)
        if not fyers_symbol:
            # Fallback to default NSE equity format
            fyers_symbol = f"NSE:{symbol}-EQ"

        return fyers_symbol

    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of all broker positions

        Returns:
            Dictionary with position information
        """
        try:
            response = self.fyers.positions()

            if response and response.get('s') == 'ok':
                positions = response.get('netPositions', [])

                return {
                    'success': True,
                    'positions': positions,
                    'count': len(positions)
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Unknown error'),
                    'positions': [],
                    'count': 0
                }

        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'positions': [],
                'count': 0
            }

    def verify_broker_connection(self) -> bool:
        """
        Verify connection to broker API

        Returns:
            bool: True if connected successfully
        """
        try:
            response = self.fyers.get_profile()

            if response and response.get('s') == 'ok':
                logger.info("Broker connection verified")
                return True
            else:
                logger.error("Broker connection failed")
                return False

        except Exception as e:
            logger.error(f"Broker connection verification error: {e}")
            return False
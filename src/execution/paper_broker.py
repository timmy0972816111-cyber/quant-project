from __future__ import annotations

from datetime import datetime
from src.execution.base_broker import BaseBroker


class PaperBroker(BaseBroker):
    """
    最簡化 paper broker：
    - 假設市價單立即成交
    - 不處理撮合簿
    - 不處理滑價
    - 僅更新本地持倉與現金
    """

    def __init__(self, initial_cash=1_000_000):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions = {}   # {symbol: quantity}
        self.orders = []      # list of order dicts
        self.connected = False
        self._next_order_id = 1

    def connect(self):
        self.connected = True
        print("PaperBroker connected")

    def disconnect(self):
        self.connected = False
        print("PaperBroker disconnected")

    def get_account_info(self):
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
        }

    def get_positions(self):
        return dict(self.positions)

    def get_open_orders(self):
        return [o for o in self.orders if o["status"] not in ("filled", "cancelled")]

    def place_order(self, symbol, side, quantity, order_type="MKT", price=None):
        if not self.connected:
            raise RuntimeError("Broker is not connected")

        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        fill_price = float(price) if price is not None else None

        order = {
            "order_id": self._next_order_id,
            "timestamp": datetime.now(),
            "symbol": symbol,
            "side": side,
            "quantity": int(quantity),
            "order_type": order_type,
            "price": fill_price,
            "status": "filled",
        }
        self._next_order_id += 1

        current_position = self.positions.get(symbol, 0)

        if side == "BUY":
            self.positions[symbol] = current_position + int(quantity)
            if fill_price is not None:
                self.cash -= fill_price * quantity

        elif side == "SELL":
            new_position = current_position - int(quantity)
            if new_position < 0:
                raise ValueError(
                    f"PaperBroker 不允許賣超。symbol={symbol}, current={current_position}, sell={quantity}"
                )
            self.positions[symbol] = new_position
            if fill_price is not None:
                self.cash += fill_price * quantity

        self.orders.append(order)
        return order

    def cancel_order(self, order_id):
        for order in self.orders:
            if order["order_id"] == order_id and order["status"] != "filled":
                order["status"] = "cancelled"
                return True
        return False
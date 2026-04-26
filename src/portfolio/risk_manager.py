class RiskManager:
    """
    最簡化版風控：
    - 限制單筆下單口數 / 股數
    - 預留未來可擴充單日最大虧損、最大持倉等
    """

    def __init__(self, max_order_qty=1, allow_short=False):
        self.max_order_qty = int(max_order_qty)
        self.allow_short = allow_short

    def check_order(self, side, quantity, current_position=0):
        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        if quantity > self.max_order_qty:
            raise ValueError(
                f"下單數量超過限制：quantity={quantity}, max_order_qty={self.max_order_qty}"
            )

        if side.upper() == "SELL" and (not self.allow_short) and quantity > current_position:
            raise ValueError(
                f"目前設定不允許賣超：current_position={current_position}, sell_qty={quantity}"
            )

        return True
class PositionManager:
    """
    負責跟 broker 同步持倉狀態
    """

    def __init__(self, broker):
        self.broker = broker

    def sync_positions(self):
        return self.broker.get_positions()

    def get_position(self, symbol):
        positions = self.sync_positions()
        return positions.get(symbol, 0)

    def has_position(self, symbol):
        return self.get_position(symbol) != 0
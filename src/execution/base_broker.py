from abc import ABC, abstractmethod


class BaseBroker(ABC):
    @abstractmethod
    def connect(self):
        """建立連線"""
        pass

    @abstractmethod
    def disconnect(self):
        """中斷連線"""
        pass

    @abstractmethod
    def get_account_info(self):
        """取得帳戶資訊"""
        pass

    @abstractmethod
    def get_positions(self):
        """取得目前持倉"""
        pass

    @abstractmethod
    def place_order(self, symbol, side, quantity, order_type="MKT", price=None):
        """送出訂單"""
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        """取消訂單"""
        pass

    @abstractmethod
    def get_open_orders(self):
        """查詢未成交訂單"""
        pass
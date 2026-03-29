from tau2.domains.burger.data_model import BurgerDB, BurgerOrder, MenuItem, OrderStatus
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class BurgerTools(ToolKitBase):
    """Simple tools for a burger ordering domain."""

    db: BurgerDB

    def __init__(self, db: BurgerDB) -> None:
        super().__init__(db)

    def _find_menu_item_by_name(self, burger_name: str) -> MenuItem:
        for menu_item in self.db.menu_items.values():
            if menu_item.name.lower() == burger_name.lower():
                return menu_item
        raise ValueError(f"Burger '{burger_name}' not found")

    @is_tool(ToolType.READ)
    def get_menu(self) -> list[MenuItem]:
        """
        Get the currently available burger menu.
        """
        return [item for item in self.db.menu_items.values() if item.available]

    @is_tool(ToolType.WRITE)
    def place_order(
        self, customer_name: str, burger_name: str, quantity: int, pickup_time: str
    ) -> BurgerOrder:
        """
        Place a pickup order for a burger.

        Args:
            customer_name: The name that should be attached to the order.
            burger_name: The exact burger name from the menu, such as 'Classic Burger'.
            quantity: Number of burgers to order. Must be between 1 and 10.
            pickup_time: Requested pickup time, such as '12:30 PM'.

        Returns:
            The created burger order.

        Raises:
            ValueError: If the burger does not exist, is unavailable, quantity is invalid, or pickup time is blank.
        """
        menu_item = self._find_menu_item_by_name(burger_name)
        if not menu_item.available:
            raise ValueError(f"Burger '{burger_name}' is currently unavailable")
        if quantity < 1 or quantity > 10:
            raise ValueError("Quantity must be between 1 and 10")
        if not pickup_time.strip():
            raise ValueError("Pickup time must not be empty")

        order_id = f"BURGER-{len(self.db.orders) + 1:03d}"
        order = BurgerOrder(
            order_id=order_id,
            customer_name=customer_name,
            item_id=menu_item.item_id,
            item_name=menu_item.name,
            quantity=quantity,
            pickup_time=pickup_time,
            total_price=round(menu_item.price * quantity, 2),
            status="confirmed",
        )
        self.db.orders[order_id] = order
        return order

    def assert_order_status(self, order_id: str, expected_status: OrderStatus) -> bool:
        """
        Check whether an order exists with the expected status.
        """
        if order_id not in self.db.orders:
            raise ValueError(f"Order '{order_id}' not found")
        return self.db.orders[order_id].status == expected_status

    def assert_order_matches(
        self,
        order_id: str,
        customer_name: str,
        burger_name: str,
        quantity: int,
        pickup_time: str,
    ) -> bool:
        """
        Check whether an order matches the expected details.
        """
        if order_id not in self.db.orders:
            raise ValueError(f"Order '{order_id}' not found")
        order = self.db.orders[order_id]
        return (
            order.customer_name == customer_name
            and order.item_name == burger_name
            and order.quantity == quantity
            and order.pickup_time == pickup_time
        )

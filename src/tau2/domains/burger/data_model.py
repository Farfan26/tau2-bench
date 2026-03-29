from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from tau2.domains.burger.utils import BURGER_DB_PATH
from tau2.environment.db import DB

OrderStatus = Literal["confirmed"]


class MenuItem(BaseModel):
    item_id: str = Field(description="Unique identifier for the burger menu item")
    name: str = Field(description="Display name of the burger")
    price: float = Field(description="Price of a single burger")
    available: bool = Field(description="Whether the burger can currently be ordered")


class BurgerOrder(BaseModel):
    order_id: str = Field(description="Unique identifier for the order")
    customer_name: str = Field(description="Name attached to the order")
    item_id: str = Field(description="Identifier of the burger ordered")
    item_name: str = Field(description="Name of the burger ordered")
    quantity: int = Field(description="Number of burgers ordered")
    pickup_time: str = Field(description="Requested pickup time")
    total_price: float = Field(description="Total order price")
    status: OrderStatus = Field(description="Current order status")


class BurgerDB(DB):
    menu_items: Dict[str, MenuItem] = Field(
        description="Burger menu indexed by menu item id"
    )
    orders: Dict[str, BurgerOrder] = Field(
        description="Placed burger orders indexed by order id"
    )

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_menu_items": len(self.menu_items),
            "num_orders": len(self.orders),
        }


def get_db():
    return BurgerDB.load(BURGER_DB_PATH)

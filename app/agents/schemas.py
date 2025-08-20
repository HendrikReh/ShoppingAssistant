"""Pydantic schemas for Price and Stock agents."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"


class TransactionType(str, Enum):
    """Stock transaction types."""
    IN = "IN"  # Stock added
    OUT = "OUT"  # Stock sold
    RESERVED = "RESERVED"  # Stock reserved for cart
    CANCELLED = "CANCELLED"  # Reservation cancelled


class WarehouseLocation(str, Enum):
    """Warehouse locations."""
    MAIN = "MAIN"
    EAST = "EAST"
    WEST = "WEST"
    CENTRAL = "CENTRAL"


class ProductPrice(BaseModel):
    """Product pricing information."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    product_id: str
    product_title: str
    base_price: Decimal = Field(decimal_places=2)
    current_price: Decimal = Field(decimal_places=2)
    min_price: Optional[Decimal] = Field(None, decimal_places=2)
    max_price: Optional[Decimal] = Field(None, decimal_places=2)
    currency: Currency = Currency.USD
    last_updated: datetime = Field(default_factory=datetime.now)
    price_history: Optional[List[Dict[str, Any]]] = None
    retailer_prices: Optional[Dict[str, Decimal]] = None
    
    @property
    def discount_percentage(self) -> Optional[float]:
        """Calculate discount percentage from base price."""
        if self.base_price and self.current_price < self.base_price:
            return float((self.base_price - self.current_price) / self.base_price * 100)
        return None
    
    @property
    def is_on_sale(self) -> bool:
        """Check if product is on sale."""
        return self.current_price < self.base_price


class PriceHistory(BaseModel):
    """Historical price record."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    product_id: str
    price: Decimal = Field(decimal_places=2)
    retailer: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class StockLevel(BaseModel):
    """Product inventory information."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    product_id: str
    product_title: str
    quantity_available: int = Field(ge=0)
    quantity_reserved: int = Field(ge=0, default=0)
    reorder_level: int = Field(ge=0, default=10)
    max_stock_level: int = Field(gt=0, default=100)
    warehouse_location: WarehouseLocation = WarehouseLocation.MAIN
    last_restocked: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.now)
    
    @property
    def quantity_total(self) -> int:
        """Total quantity including reserved."""
        return self.quantity_available + self.quantity_reserved
    
    @property
    def is_in_stock(self) -> bool:
        """Check if product is in stock."""
        return self.quantity_available > 0
    
    @property
    def is_low_stock(self) -> bool:
        """Check if stock is below reorder level."""
        return self.quantity_available <= self.reorder_level
    
    @property
    def stock_percentage(self) -> float:
        """Calculate stock level as percentage of max."""
        if self.max_stock_level > 0:
            return (self.quantity_total / self.max_stock_level) * 100
        return 0.0


class StockTransaction(BaseModel):
    """Stock transaction record."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    product_id: str
    transaction_type: TransactionType
    quantity: int = Field(gt=0)
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PriceUpdate(BaseModel):
    """Request to update product price."""
    product_id: str
    new_price: Decimal = Field(decimal_places=2)
    retailer: Optional[str] = None
    reason: Optional[str] = None


class StockUpdate(BaseModel):
    """Request to update stock level."""
    product_id: str
    quantity: int
    transaction_type: TransactionType
    warehouse_location: WarehouseLocation = WarehouseLocation.MAIN
    reason: Optional[str] = None


class PriceSearchResult(BaseModel):
    """Result from price search/comparison."""
    product_id: str
    product_title: str
    local_price: Decimal = Field(decimal_places=2)
    web_prices: Dict[str, Decimal] = Field(default_factory=dict)
    best_price: Decimal = Field(decimal_places=2)
    best_retailer: str
    savings: Optional[Decimal] = Field(None, decimal_places=2)
    last_checked: datetime = Field(default_factory=datetime.now)


class StockAvailability(BaseModel):
    """Stock availability across warehouses."""
    product_id: str
    product_title: str
    total_available: int
    total_reserved: int
    warehouses: Dict[str, int]  # warehouse -> quantity
    is_available: bool
    estimated_restock: Optional[datetime] = None
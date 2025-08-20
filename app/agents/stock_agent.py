"""Stock Agent for managing product inventory and stock levels."""

import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict

from .schemas import (
    StockLevel,
    StockTransaction,
    StockUpdate,
    StockAvailability,
    TransactionType,
    WarehouseLocation
)

logger = logging.getLogger(__name__)


class StockAgent:
    """Agent for managing product inventory and stock transactions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the Stock Agent.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or Path("app/databases/inventory.db")
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create inventory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inventory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    product_title TEXT NOT NULL,
                    quantity_available INTEGER NOT NULL DEFAULT 0,
                    quantity_reserved INTEGER DEFAULT 0,
                    reorder_level INTEGER DEFAULT 10,
                    max_stock_level INTEGER DEFAULT 100,
                    warehouse_location TEXT DEFAULT 'MAIN',
                    last_restocked TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(product_id, warehouse_location)
                )
            """)
            
            # Create stock_transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    transaction_type TEXT CHECK(transaction_type IN ('IN', 'OUT', 'RESERVED', 'CANCELLED')),
                    quantity INTEGER NOT NULL,
                    warehouse_location TEXT DEFAULT 'MAIN',
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_warehouse ON inventory(warehouse_location)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_product ON stock_transactions(product_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON stock_transactions(timestamp)")
            
            conn.commit()
            logger.info(f"Inventory database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def add_product_stock(self, stock_level: StockLevel) -> StockLevel:
        """Add a new product to inventory.
        
        Args:
            stock_level: Stock level data
            
        Returns:
            Created stock level with ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO inventory (
                    product_id, product_title, quantity_available,
                    quantity_reserved, reorder_level, max_stock_level,
                    warehouse_location, last_restocked
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stock_level.product_id,
                stock_level.product_title,
                stock_level.quantity_available,
                stock_level.quantity_reserved,
                stock_level.reorder_level,
                stock_level.max_stock_level,
                stock_level.warehouse_location.value,
                stock_level.last_restocked
            ))
            
            conn.commit()
            stock_level.id = cursor.lastrowid
            
            # Record initial stock transaction
            if stock_level.quantity_available > 0:
                self._record_transaction(
                    stock_level.product_id,
                    TransactionType.IN,
                    stock_level.quantity_available,
                    stock_level.warehouse_location,
                    "Initial stock"
                )
            
            logger.info(f"Added stock for product {stock_level.product_id} in {stock_level.warehouse_location}")
            return stock_level
    
    def get_stock_level(
        self,
        product_id: str,
        warehouse: Optional[WarehouseLocation] = None
    ) -> Optional[StockLevel]:
        """Get stock level for a product.
        
        Args:
            product_id: Product ID
            warehouse: Specific warehouse location
            
        Returns:
            Stock level data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if warehouse:
                cursor.execute("""
                    SELECT * FROM inventory
                    WHERE product_id = ? AND warehouse_location = ?
                """, (product_id, warehouse.value))
            else:
                # Get from main warehouse by default
                cursor.execute("""
                    SELECT * FROM inventory
                    WHERE product_id = ? AND warehouse_location = ?
                """, (product_id, WarehouseLocation.MAIN.value))
            
            row = cursor.fetchone()
            if row:
                stock_data = dict(row)
                stock_data['warehouse_location'] = WarehouseLocation(stock_data['warehouse_location'])
                return StockLevel(**stock_data)
        
        return None
    
    def get_total_stock(self, product_id: str) -> StockAvailability:
        """Get total stock across all warehouses.
        
        Args:
            product_id: Product ID
            
        Returns:
            Stock availability summary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM inventory WHERE product_id = ?
            """, (product_id,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return StockAvailability(
                    product_id=product_id,
                    product_title="Unknown Product",
                    total_available=0,
                    total_reserved=0,
                    warehouses={},
                    is_available=False
                )
            
            total_available = 0
            total_reserved = 0
            warehouses = {}
            product_title = ""
            
            for row in rows:
                stock_data = dict(row)
                total_available += stock_data['quantity_available']
                total_reserved += stock_data['quantity_reserved']
                warehouses[stock_data['warehouse_location']] = stock_data['quantity_available']
                product_title = stock_data['product_title']
            
            return StockAvailability(
                product_id=product_id,
                product_title=product_title,
                total_available=total_available,
                total_reserved=total_reserved,
                warehouses=warehouses,
                is_available=total_available > 0
            )
    
    def update_stock(self, stock_update: StockUpdate) -> Optional[StockLevel]:
        """Update stock level for a product.
        
        Args:
            stock_update: Stock update request
            
        Returns:
            Updated stock level or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current stock
            current_stock = self.get_stock_level(
                stock_update.product_id,
                stock_update.warehouse_location
            )
            
            if not current_stock:
                return None
            
            # Calculate new quantities based on transaction type
            if stock_update.transaction_type == TransactionType.IN:
                new_available = current_stock.quantity_available + stock_update.quantity
                new_reserved = current_stock.quantity_reserved
                last_restocked = datetime.now()
            elif stock_update.transaction_type == TransactionType.OUT:
                new_available = max(0, current_stock.quantity_available - stock_update.quantity)
                new_reserved = current_stock.quantity_reserved
                last_restocked = current_stock.last_restocked
            elif stock_update.transaction_type == TransactionType.RESERVED:
                new_available = max(0, current_stock.quantity_available - stock_update.quantity)
                new_reserved = current_stock.quantity_reserved + stock_update.quantity
                last_restocked = current_stock.last_restocked
            elif stock_update.transaction_type == TransactionType.CANCELLED:
                new_available = current_stock.quantity_available + stock_update.quantity
                new_reserved = max(0, current_stock.quantity_reserved - stock_update.quantity)
                last_restocked = current_stock.last_restocked
            else:
                return None
            
            # Update inventory
            cursor.execute("""
                UPDATE inventory
                SET quantity_available = ?, quantity_reserved = ?,
                    last_restocked = ?, last_updated = CURRENT_TIMESTAMP
                WHERE product_id = ? AND warehouse_location = ?
            """, (
                new_available,
                new_reserved,
                last_restocked,
                stock_update.product_id,
                stock_update.warehouse_location.value
            ))
            
            conn.commit()
            
            # Record transaction
            self._record_transaction(
                stock_update.product_id,
                stock_update.transaction_type,
                stock_update.quantity,
                stock_update.warehouse_location,
                stock_update.reason
            )
            
            logger.info(
                f"Updated stock for {stock_update.product_id}: "
                f"{stock_update.transaction_type.value} {stock_update.quantity} units"
            )
            
            return self.get_stock_level(stock_update.product_id, stock_update.warehouse_location)
    
    def _record_transaction(
        self,
        product_id: str,
        transaction_type: TransactionType,
        quantity: int,
        warehouse: WarehouseLocation,
        reason: Optional[str] = None
    ):
        """Record a stock transaction.
        
        Args:
            product_id: Product ID
            transaction_type: Type of transaction
            quantity: Quantity involved
            warehouse: Warehouse location
            reason: Transaction reason
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stock_transactions (
                    product_id, transaction_type, quantity,
                    warehouse_location, reason
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                product_id,
                transaction_type.value,
                quantity,
                warehouse.value,
                reason
            ))
            conn.commit()
    
    def reserve_stock(
        self,
        product_id: str,
        quantity: int,
        warehouse: Optional[WarehouseLocation] = None
    ) -> bool:
        """Reserve stock for a customer order.
        
        Args:
            product_id: Product ID
            quantity: Quantity to reserve
            warehouse: Preferred warehouse
            
        Returns:
            True if reservation successful
        """
        # Try to reserve from specified warehouse or find best warehouse
        if warehouse:
            stock = self.get_stock_level(product_id, warehouse)
            if stock and stock.quantity_available >= quantity:
                update = StockUpdate(
                    product_id=product_id,
                    quantity=quantity,
                    transaction_type=TransactionType.RESERVED,
                    warehouse_location=warehouse,
                    reason="Customer reservation"
                )
                return self.update_stock(update) is not None
        else:
            # Find warehouse with sufficient stock
            availability = self.get_total_stock(product_id)
            for wh_location, wh_quantity in availability.warehouses.items():
                if wh_quantity >= quantity:
                    update = StockUpdate(
                        product_id=product_id,
                        quantity=quantity,
                        transaction_type=TransactionType.RESERVED,
                        warehouse_location=WarehouseLocation(wh_location),
                        reason="Customer reservation"
                    )
                    return self.update_stock(update) is not None
        
        return False
    
    def cancel_reservation(
        self,
        product_id: str,
        quantity: int,
        warehouse: WarehouseLocation = WarehouseLocation.MAIN
    ) -> bool:
        """Cancel a stock reservation.
        
        Args:
            product_id: Product ID
            quantity: Quantity to unreserve
            warehouse: Warehouse location
            
        Returns:
            True if cancellation successful
        """
        update = StockUpdate(
            product_id=product_id,
            quantity=quantity,
            transaction_type=TransactionType.CANCELLED,
            warehouse_location=warehouse,
            reason="Reservation cancelled"
        )
        return self.update_stock(update) is not None
    
    def get_low_stock_products(self, warehouse: Optional[WarehouseLocation] = None) -> List[StockLevel]:
        """Get products with low stock levels.
        
        Args:
            warehouse: Filter by warehouse
            
        Returns:
            List of products below reorder level
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if warehouse:
                cursor.execute("""
                    SELECT * FROM inventory
                    WHERE quantity_available <= reorder_level
                    AND warehouse_location = ?
                    ORDER BY quantity_available ASC
                """, (warehouse.value,))
            else:
                cursor.execute("""
                    SELECT * FROM inventory
                    WHERE quantity_available <= reorder_level
                    ORDER BY quantity_available ASC
                """)
            
            products = []
            for row in cursor.fetchall():
                stock_data = dict(row)
                stock_data['warehouse_location'] = WarehouseLocation(stock_data['warehouse_location'])
                products.append(StockLevel(**stock_data))
            
            return products
    
    def get_out_of_stock_products(self, warehouse: Optional[WarehouseLocation] = None) -> List[StockLevel]:
        """Get products that are out of stock.
        
        Args:
            warehouse: Filter by warehouse
            
        Returns:
            List of out-of-stock products
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if warehouse:
                cursor.execute("""
                    SELECT * FROM inventory
                    WHERE quantity_available = 0
                    AND warehouse_location = ?
                """, (warehouse.value,))
            else:
                cursor.execute("""
                    SELECT * FROM inventory
                    WHERE quantity_available = 0
                """)
            
            products = []
            for row in cursor.fetchall():
                stock_data = dict(row)
                stock_data['warehouse_location'] = WarehouseLocation(stock_data['warehouse_location'])
                products.append(StockLevel(**stock_data))
            
            return products
    
    def get_transaction_history(
        self,
        product_id: str,
        days: int = 30,
        limit: Optional[int] = None
    ) -> List[StockTransaction]:
        """Get transaction history for a product.
        
        Args:
            product_id: Product ID
            days: Number of days of history
            limit: Maximum number of records
            
        Returns:
            List of stock transactions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT * FROM stock_transactions
                WHERE product_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (product_id, since_date))
            
            transactions = []
            for row in cursor.fetchall():
                trans_data = dict(row)
                trans_data['transaction_type'] = TransactionType(trans_data['transaction_type'])
                # Remove warehouse_location from transaction as it's not in schema
                trans_data.pop('warehouse_location', None)
                transactions.append(StockTransaction(**trans_data))
            
            return transactions
    
    def bulk_restock(self, restock_list: List[Dict[str, Any]]) -> int:
        """Bulk restock multiple products.
        
        Args:
            restock_list: List of dicts with product_id, quantity, warehouse
            
        Returns:
            Number of products restocked
        """
        restocked_count = 0
        
        for item in restock_list:
            update = StockUpdate(
                product_id=item['product_id'],
                quantity=item['quantity'],
                transaction_type=TransactionType.IN,
                warehouse_location=WarehouseLocation(item.get('warehouse', 'MAIN')),
                reason=item.get('reason', 'Bulk restock')
            )
            
            if self.update_stock(update):
                restocked_count += 1
        
        logger.info(f"Bulk restocked {restocked_count} products")
        return restocked_count
    
    def transfer_stock(
        self,
        product_id: str,
        quantity: int,
        from_warehouse: WarehouseLocation,
        to_warehouse: WarehouseLocation
    ) -> bool:
        """Transfer stock between warehouses.
        
        Args:
            product_id: Product ID
            quantity: Quantity to transfer
            from_warehouse: Source warehouse
            to_warehouse: Destination warehouse
            
        Returns:
            True if transfer successful
        """
        # Remove from source warehouse
        out_update = StockUpdate(
            product_id=product_id,
            quantity=quantity,
            transaction_type=TransactionType.OUT,
            warehouse_location=from_warehouse,
            reason=f"Transfer to {to_warehouse.value}"
        )
        
        if not self.update_stock(out_update):
            return False
        
        # Add to destination warehouse
        in_update = StockUpdate(
            product_id=product_id,
            quantity=quantity,
            transaction_type=TransactionType.IN,
            warehouse_location=to_warehouse,
            reason=f"Transfer from {from_warehouse.value}"
        )
        
        return self.update_stock(in_update) is not None
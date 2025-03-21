"""PostgreSQL database connection manager with connection pooling."""

import os
from typing import List, Dict, Any, Optional, Union
import psycopg2
from psycopg2 import pool, OperationalError, InterfaceError, InternalError, DatabaseError
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
import logging
import time
from urllib.parse import urlparse

from .schema_init import initialize_schema

logger = logging.getLogger(__name__)


class PostgresManager:
    """Manages PostgreSQL database connections and query execution with connection pooling."""
    
    # Class-level variable to track schema initialization
    _schema_initialized = False

    # Class-level variable to hold the connection pool
    _connection_pool = None
    
    def __init__(
        self,
        min_connections: int = 1,
        max_connections: int = 100,
        retry_attempts: int = 3,
        retry_delay: int = 2
    ):
        """Initialize the database connection manager.
        
        Args:
            min_connections: Minimum number of connections in the pool
            max_connections: Maximum number of connections in the pool
            retry_attempts: Number of retry attempts for failed queries
            retry_delay: Delay in seconds between retries
        """
        schema = os.getenv('DB_SCHEMA', 'darwin_pro')
        if not schema:
            raise ValueError("DB_SCHEMA environment variable is required")
        self.schema = schema.lower()
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Parse connection parameters
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            parsed = urlparse(database_url)
            self._conn_params = {
                'dbname': parsed.path[1:],
                'user': parsed.username,
                'password': parsed.password,
                'host': parsed.hostname,
                'port': str(parsed.port or '5432')
            }
        else:
            self._conn_params = {
                'dbname': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT', '5432')
            }
        
        self._validate_connection_params()
        self._initialize_connection_pool()
        # Check and initialize schema only if it hasn't been done yet
        if not PostgresManager._schema_initialized:
            self.ensure_schema_exists()
            PostgresManager._schema_initialized = True

    def _initialize_connection_pool(self) -> None:
        """Initialize the connection pool if it hasn't been created yet."""
        if PostgresManager._connection_pool is None:
            try:
                PostgresManager._connection_pool = pool.SimpleConnectionPool(
                    self.min_connections,
                    self.max_connections,
                    **self._conn_params
                )
                logger.info("Created database connection pool")
            except Exception as e:
                logger.warning(f"Failed to create connection pool: {str(e)}")
                raise ConnectionError(f"Failed to create connection pool: {str(e)}")

    def _validate_connection_params(self) -> None:
        """Validate that all required connection parameters are present."""
        required_params = ['dbname', 'user', 'password', 'host']
        missing_params = [param for param in required_params if not self._conn_params.get(param)]
        
        if missing_params:
            raise ValueError(
                f"Missing required database connection parameters: {', '.join(missing_params)}"
            )

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        # Check if the pool is initialized and open
        if PostgresManager._connection_pool is None or PostgresManager._connection_pool.closed:
            logger.info("Connection pool is closed or not initialized. Creating new pool.")
            self._initialize_connection_pool()

        connection = None
        try:
            connection = PostgresManager._connection_pool.getconn()
            yield connection
        except Exception as e:
            logger.warning(f"Failed to get connection from pool: {str(e)}")
            raise
        finally:
            if connection:
                PostgresManager._connection_pool.putconn(connection)

    @contextmanager
    def transaction(self):
        """Create a new database transaction with automatic commit/rollback."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.warning(f"Transaction failed: {str(e)}")
                raise

    def execute_query(
        self,
        query: str,
        params: Optional[Union[tuple, List[tuple]]] = None,
        fetch_all: bool = True,
        as_dict: bool = True,
        retry_count: int = None,
        use_executemany: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute a database query with retry mechanism.
        
        Args:
            query: SQL query to execute
            params: Query parameters to bind
            fetch_all: Whether to fetch all results or just one
            as_dict: Whether to return results as dictionaries
            retry_count: Number of retry attempts (defaults to self.retry_attempts)
            use_executemany: If True, executes in batch mode using execute_values
        """
        attempts = 0
        retry_count = retry_count or self.retry_attempts

        while attempts < retry_count:
            try:
                with self.transaction() as conn:
                    cursor_factory = RealDictCursor if as_dict else None
                    with conn.cursor(cursor_factory=cursor_factory) as cur:
                        # cur.execute(query, params)
                        if use_executemany and isinstance(params, list):
                            execute_values(cur, query, params)
                        else:
                            cur.execute(query, params)
                        
                        if not cur.description:
                            return cur.rowcount
                            
                        if fetch_all:
                            return cur.fetchall()
                        result = cur.fetchone()
                        return [result] if result else []

            except (OperationalError, InterfaceError, InternalError, DatabaseError) as e:
                attempts += 1
                if attempts >= retry_count:
                    logger.warning(f"Query failed after {retry_count} attempts: {str(e)}")
                    raise
                logger.warning(f"Query attempt {attempts} failed, retrying... Error: {str(e)}")
                time.sleep(self.retry_delay)
                self._initialize_connection_pool()

    def execute_batch(
        self,
        query: str,
        params_list: List[tuple],
        page_size: int = 1000,
        retry_count: int = None
    ) -> int:
        """Execute a batch operation with retry mechanism."""
        attempts = 0
        retry_count = retry_count or self.retry_attempts
        total_affected = 0

        while attempts < retry_count:
            try:
                with self.transaction() as conn:
                    with conn.cursor() as cur:
                        for i in range(0, len(params_list), page_size):
                            batch_params = params_list[i:i + page_size]
                            cur.executemany(query, batch_params)
                            total_affected += cur.rowcount
                return total_affected

            except (OperationalError, InterfaceError, InternalError, DatabaseError) as e:
                attempts += 1
                if attempts >= retry_count:
                    logger.warning(f"Batch execution failed after {retry_count} attempts: {str(e)}")
                    raise
                logger.warning(f"Batch attempt {attempts} failed, retrying... Error: {str(e)}")
                time.sleep(self.retry_delay)
                self._initialize_connection_pool()

    def ensure_schema_exists(self) -> None:
        """Create the schema and initialize all objects if they don't exist."""
        try:
            # Create a direct connection for schema operations
            conn = psycopg2.connect(**self._conn_params)
            conn.autocommit = True  # Set autocommit immediately
            try:
                # Check if schema exists
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s",
                        (self.schema,)
                    )
                    schema_exists = cur.fetchone() is not None

                if not schema_exists:
                    initialize_schema(conn, self.schema)
                    logger.info(f"Initialized schema '{self.schema}'")
                else:
                    logger.info(f"Schema '{self.schema}' already exists")
            finally:
                conn.close()
                        
        except Exception as e:
            logger.warning(f"Failed to ensure schema exists: {str(e)}")
            raise

    def __del__(self):
        """Cleanup connection pool on deletion."""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            
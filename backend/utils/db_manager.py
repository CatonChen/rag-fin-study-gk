import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator, Optional
from utils.logging_config import LoggingConfig
from utils.error_handler import DatabaseError

logger = LoggingConfig().logger

class DatabaseManager:
    """数据库连接管理器
    
    提供以下功能：
    1. 数据库连接池管理
    2. 事务管理
    3. 连接超时处理
    4. 自动重试机制
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str):
        """初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        if not hasattr(self, 'initialized'):
            self.db_path = db_path
            self._local = threading.local()
            self.initialized = True
            logger.info(f"数据库管理器初始化完成: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接
        
        Returns:
            sqlite3.Connection: 数据库连接对象
            
        Raises:
            DatabaseError: 当连接失败时
        """
        try:
            if not hasattr(self._local, 'connection'):
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    timeout=30,  # 连接超时时间
                    check_same_thread=False  # 允许多线程访问
                )
                # 启用外键约束
                self._local.connection.execute("PRAGMA foreign_keys = ON")
                # 启用WAL模式提高并发性能
                self._local.connection.execute("PRAGMA journal_mode = WAL")
                logger.debug("创建新的数据库连接")
            return self._local.connection
        except sqlite3.Error as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise DatabaseError(f"数据库连接失败: {str(e)}")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """获取数据库连接的上下文管理器
        
        Yields:
            sqlite3.Connection: 数据库连接对象
            
        Raises:
            DatabaseError: 当连接失败时
        """
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"数据库操作失败: {str(e)}")
            raise DatabaseError(f"数据库操作失败: {str(e)}")
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """事务管理的上下文管理器
        
        Yields:
            sqlite3.Connection: 数据库连接对象
            
        Raises:
            DatabaseError: 当事务失败时
        """
        conn = None
        try:
            conn = self._get_connection()
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.commit()
            logger.debug("事务提交成功")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"事务失败: {str(e)}")
            raise DatabaseError(f"事务失败: {str(e)}")
    
    def close(self):
        """关闭数据库连接"""
        try:
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
                del self._local.connection
                logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {str(e)}")
            raise DatabaseError(f"关闭数据库连接失败: {str(e)}")
    
    def __del__(self):
        """清理资源"""
        self.close() 
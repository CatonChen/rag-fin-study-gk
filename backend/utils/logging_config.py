from dataclasses import dataclass
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

@dataclass
class LoggingConfig:
    """日志配置类"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def configure(self):
        """配置日志"""
        logging.basicConfig(
            level=getattr(logging, self.level),
            format=self.format
        )
        
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        return logging.getLogger(__name__) 
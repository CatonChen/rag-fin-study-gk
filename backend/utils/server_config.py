from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

@dataclass
class ServerConfig:
    """服务器配置类"""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    api_prefix: str = "/api"
    api_version: str = "v1"
    
    @property
    def server_url(self) -> str:
        """获取服务器URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def api_url(self) -> str:
        """获取API URL前缀"""
        return f"{self.api_prefix}/{self.api_version}"

@dataclass
class CORSConfig:
    """跨域配置类"""
    origins: list = None
    methods: list = None
    headers: list = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.origins is None:
            self.origins = os.getenv("CORS_ORIGINS", "*").split(",")
        if self.methods is None:
            self.methods = ["*"]
        if self.headers is None:
            self.headers = ["*"] 
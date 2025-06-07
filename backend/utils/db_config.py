from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

@dataclass
class DBConfig:
    """数据库配置类"""
    db_path: str = os.getenv("DB_PATH", "db/financial_terms_zhipu.db")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "db/vector_store")
    collection_name: str = "financial_terms"
    
    @property
    def index_path(self) -> str:
        """获取FAISS索引文件路径"""
        return f"{self.db_path}.index"
    
    @property
    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return all([
            self.db_path,
            self.vector_db_path,
            self.collection_name
        ]) 
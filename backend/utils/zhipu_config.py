from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ZhipuModelType(Enum):
    """智谱AI模型类型"""
    EMBEDDING = "embedding"  # 嵌入模型
    LLM = "llm"  # 大语言模型

class ZhipuEmbeddingModel(Enum):
    """智谱AI嵌入模型"""
    EMBEDDING_2 = "embedding-2"  # 第二代嵌入模型
    EMBEDDING_3 = "embedding-3"  # 第三代嵌入模型

class ZhipuLLMModel(Enum):
    """智谱AI大语言模型"""
    GLM_4 = "glm-4"  # GLM-4
    GLM_4_PLUS = "glm-4-plus"  # GLM-4-Plus
    GLM_4V = "glm-4v"  # GLM-4V

@dataclass
class ZhipuConfig:
    """智谱AI配置类"""
    model_type: ZhipuModelType
    model_name: str
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.api_key is None:
            self.api_key = os.getenv("ZHIPU_API_KEY")
            if not self.api_key:
                raise ValueError("ZHIPU_API_KEY not found in environment variables")

@dataclass
class ZhipuEmbeddingConfig(ZhipuConfig):
    """智谱AI嵌入模型配置"""
    def __post_init__(self):
        super().__post_init__()
        if self.model_type != ZhipuModelType.EMBEDDING:
            raise ValueError("Model type must be EMBEDDING")
        if self.model_name not in [model.value for model in ZhipuEmbeddingModel]:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")

@dataclass
class ZhipuLLMConfig(ZhipuConfig):
    """智谱AI大语言模型配置"""
    temperature: float = 0.7
    top_p: float = 0.7
    max_tokens: int = 2048
    
    def __post_init__(self):
        super().__post_init__()
        if self.model_type != ZhipuModelType.LLM:
            raise ValueError("Model type must be LLM")
        if self.model_name not in [model.value for model in ZhipuLLMModel]:
            raise ValueError(f"Unsupported LLM model: {self.model_name}") 
from typing import Union
from zhipuai import ZhipuAI
from .zhipu_config import (
    ZhipuConfig,
    ZhipuEmbeddingConfig,
    ZhipuLLMConfig,
    ZhipuModelType
)

class ZhipuFactory:
    """智谱AI工厂类"""
    
    @staticmethod
    def create_client(config: Union[ZhipuConfig, ZhipuEmbeddingConfig, ZhipuLLMConfig]) -> ZhipuAI:
        """创建智谱AI客户端
        
        Args:
            config: 智谱AI配置对象
            
        Returns:
            ZhipuAI: 智谱AI客户端实例
        """
        return ZhipuAI(api_key=config.api_key)
    
    @staticmethod
    def create_embedding(config: ZhipuEmbeddingConfig) -> ZhipuAI:
        """创建嵌入模型实例
        
        Args:
            config: 嵌入模型配置对象
            
        Returns:
            ZhipuAI: 智谱AI客户端实例
        """
        if config.model_type != ZhipuModelType.EMBEDDING:
            raise ValueError("Model type must be EMBEDDING")
        return ZhipuFactory.create_client(config)
    
    @staticmethod
    def create_llm(config: ZhipuLLMConfig) -> ZhipuAI:
        """创建大语言模型实例
        
        Args:
            config: 大语言模型配置对象
            
        Returns:
            ZhipuAI: 智谱AI客户端实例
        """
        if config.model_type != ZhipuModelType.LLM:
            raise ValueError("Model type must be LLM")
        return ZhipuFactory.create_client(config) 
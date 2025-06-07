from typing import Any, List
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr
import zhipuai
import os
import time

class ZhipuAIEmbedding(BaseEmbedding):
    _api_key: str = PrivateAttr()
    _model: str = PrivateAttr()
    _client: Any = PrivateAttr()
    _timeout: int = PrivateAttr(default=30)  # 默认30秒超时

    def __init__(
        self,
        api_key: str = None,
        model: str = "embedding-3",  # 使用最新的模型版本
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # 尝试多种方式获取API密钥
        self._api_key = (
            api_key or 
            os.getenv("ZHIPUAI_API_KEY") or 
            os.environ.get("ZHIPUAI_API_KEY")
        )
        if not self._api_key:
            raise ValueError("未找到ZHIPUAI_API_KEY环境变量，请确保已设置")
            
        self._model = model
        self._timeout = timeout
        self._client = zhipuai.ZhipuAI(api_key=self._api_key)

    def _get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        start_time = time.time()
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=text,
                timeout=self._timeout
            )
            if not response or not response.data:
                raise ValueError("API返回数据为空")
            return response.data[0].embedding
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"获取嵌入向量时出错 (耗时: {elapsed_time:.2f}秒): {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        return self._get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入向量"""
        start_time = time.time()
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
                timeout=self._timeout
            )
            if not response or not response.data:
                raise ValueError("API返回数据为空")
            return [item.embedding for item in response.data]
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"批量获取嵌入向量时出错 (耗时: {elapsed_time:.2f}秒): {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        return self._get_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询文本的嵌入向量"""
        return self._get_embedding(query) 
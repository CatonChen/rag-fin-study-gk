import faiss
import numpy as np
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
import logging
import sqlite3
from utils.zhipu_config import ZhipuLLMConfig, ZhipuModelType
from utils.zhipu_factory import ZhipuFactory
from utils.db_config import DBConfig
from utils.db_manager import DatabaseManager
from utils.logging_config import LoggingConfig
from utils.error_handler import ModelError, ValidationError

# 初始化配置
db_config = DBConfig()
logging_config = LoggingConfig()
logger = logging_config.logger

load_dotenv()

class FinancialStdService:
    """金融术语标准化服务
    
    提供以下功能：
    1. 金融术语标准化
    2. 术语相似度匹配
    3. 术语验证
    4. 术语分类
    
    特点：
    - 使用智谱AI的GLM-4-Plus模型进行术语标准化
    - 使用FAISS向量数据库进行相似度匹配
    - 支持相似度阈值过滤
    - 提供术语元数据查询
    """
    
    def __init__(
        self,
        model_name: str = "glm-4-plus",
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 2048
    ):
        """初始化标准化服务
        
        Args:
            model_name: 模型名称，默认使用智谱AI的GLM-4-Plus
            temperature: 温度参数，控制输出的随机性
            top_p: 核采样参数，控制输出的多样性
            max_tokens: 最大生成token数
        """
        # 初始化LLM模型
        self.llm_config = ZhipuLLMConfig(
            model_type=ZhipuModelType.LLM,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        self.client = ZhipuFactory.create_llm(self.llm_config)
        
        # 初始化数据库管理器
        self.db_manager = DatabaseManager(db_config.db_path)
        
        # 加载FAISS索引
        self.index = faiss.read_index(db_config.index_path)
        
        # 加载术语向量和元数据
        self._load_term_vectors()
        
        logger.info(f"初始化标准化服务完成，使用模型：{model_name}")

    def _load_term_vectors(self):
        """加载术语向量和元数据
        
        Raises:
            ModelError: 当加载失败时
        """
        try:
            logger.debug("开始加载术语向量和元数据")
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 加载术语向量
                cursor.execute("SELECT term, vector FROM term_vectors")
                self.term_vectors = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor.fetchall()}
                
                # 加载术语元数据
                cursor.execute("SELECT term, type, definition FROM term_metadata")
                self.term_metadata = {row[0]: {"type": row[1], "definition": row[2]} for row in cursor.fetchall()}
                
            logger.info(f"术语向量和元数据加载完成，共 {len(self.term_vectors)} 个术语")
        except Exception as e:
            logger.error(f"加载术语向量和元数据失败: {str(e)}")
            raise ModelError(f"加载术语向量和元数据失败: {str(e)}")

    async def standardize(
        self,
        text: str,
        options: Dict[str, Any],
        term_types: Dict[str, bool],
        zhipu_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """标准化金融术语
        
        Args:
            text: 输入文本
            options: 处理选项
            term_types: 术语类型配置
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 标准化结果
            
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            logger.info(f"开始标准化处理: {text[:100]}...")
            if not text.strip():
                raise ValidationError("输入文本不能为空")
                
            # 标准化处理
            standardized_terms = await self._standardize_terms(text)
            
            # 过滤术语
            filtered_terms = self._filter_terms(standardized_terms, term_types)
            
            logger.info(f"标准化处理完成，共处理 {len(filtered_terms)} 个术语")
            return {
                "standardized_terms": filtered_terms
            }
        except Exception as e:
            logger.error(f"标准化处理失败: {str(e)}")
            if isinstance(e, (ValidationError, ModelError)):
                raise e
            raise ModelError(f"术语标准化失败: {str(e)}")
    
    async def _standardize_terms(self, text: str) -> List[Dict[str, Any]]:
        """标准化术语
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 标准化后的术语列表
            
        Raises:
            ModelError: 当标准化处理失败时
        """
        prompt = f"""请对以下文本中的金融术语进行标准化处理。
        请以JSON格式返回结果，包含原始术语、标准化术语、术语类型和置信度。
        
        文本：{text}
        """
        
        try:
            logger.debug("调用模型进行术语标准化")
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语标准化专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result.get("terms", [])
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"术语标准化处理失败: {str(e)}")
    
    def _filter_terms(
        self,
        terms: List[Dict[str, Any]],
        term_types: Dict[str, bool]
    ) -> List[Dict[str, Any]]:
        """过滤术语
        
        Args:
            terms: 术语列表
            term_types: 术语类型配置
            
        Returns:
            List[Dict[str, Any]]: 过滤后的术语列表
            
        Raises:
            ModelError: 当术语过滤失败时
        """
        try:
            logger.debug(f"开始过滤术语，原始数量: {len(terms)}")
            if not term_types.get("allFinancialTerms", False):
                return terms
                
            filtered = [
                term for term in terms
                if term["type"] in term_types
            ]
            logger.debug(f"术语过滤完成，过滤后数量: {len(filtered)}")
            return filtered
        except Exception as e:
            logger.error(f"术语过滤失败: {str(e)}")
            raise ModelError(f"术语过滤失败: {str(e)}")
    
    async def search_similar_terms(
        self,
        term: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """搜索相似术语
        
        Args:
            term: 输入术语
            top_k: 返回的最相似术语数量
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Dict[str, Any]]: 相似术语列表
            
        Raises:
            ValidationError: 当输入术语无效时
            ModelError: 当搜索失败时
        """
        try:
            logger.info(f"开始搜索相似术语: {term}")
            if not term.strip():
                raise ValidationError("输入术语不能为空")
                
            # 获取输入术语的向量表示
            prompt = f"""请将以下金融术语转换为向量表示。
            请以JSON格式返回结果，包含向量表示。
            
            术语：{term}
            """
            
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语向量化专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            query_vector = np.array(result.get("vector", []), dtype=np.float32)
            
            if len(query_vector) == 0:
                raise ModelError("无法获取术语的向量表示")
                
            # 使用FAISS进行相似度搜索
            query_vector = query_vector.reshape(1, -1)
            distances, indices = self.index.search(query_vector, top_k)
            
            # 构建结果
            similar_terms = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示没有找到匹配
                    continue
                    
                similarity = 1 - distance  # 将距离转换为相似度
                if similarity < similarity_threshold:
                    continue
                    
                term = list(self.term_vectors.keys())[idx]
                metadata = self.term_metadata.get(term, {})
                
                similar_terms.append({
                    "term": term,
                    "similarity": float(similarity),
                    "type": metadata.get("type", ""),
                    "definition": metadata.get("definition", "")
                })
            
            logger.info(f"相似术语搜索完成，找到 {len(similar_terms)} 个结果")
            return similar_terms
        except Exception as e:
            logger.error(f"相似术语搜索失败: {str(e)}")
            if isinstance(e, ValidationError):
                raise e
            raise ModelError(f"相似术语搜索失败: {str(e)}")

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'index'):
                del self.index
                logger.info("FAISS索引已清理")
            if hasattr(self, 'client'):
                del self.client
                logger.info("嵌入模型客户端已清理")
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
                logger.info("数据库连接已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
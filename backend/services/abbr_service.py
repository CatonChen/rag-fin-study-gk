from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
from services.std_service import FinancialStdService
from utils.zhipu_config import ZhipuLLMConfig, ZhipuModelType
from utils.zhipu_factory import ZhipuFactory
from utils.logging_config import LoggingConfig
from utils.db_config import DBConfig
from utils.error_handler import ModelError, ValidationError

# 初始化配置
logging_config = LoggingConfig()
db_config = DBConfig()
logger = logging_config.logger

load_dotenv()

class FinancialAbbrService:
    """金融术语缩写扩展服务
    
    提供以下功能：
    1. 金融术语缩写扩展
    2. 缩写定义查询
    3. 缩写标准化
    
    特点：
    - 使用智谱GLM-4模型进行缩写扩展
    - 支持上下文感知的扩展
    - 提供标准化验证
    - 支持多种扩展方法
    """
    
    def __init__(
        self,
        model_name: str = "glm-4-plus",
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 2048
    ):
        """初始化金融术语缩写服务
        
        Args:
            model_name: 模型名称，默认使用智谱AI的GLM-4-Plus
            temperature: 温度参数，控制输出的随机性
            top_p: 核采样参数，控制输出的多样性
            max_tokens: 最大生成token数
        """
        # 初始化标准化服务
        self.std_service = FinancialStdService()
        
        # 初始化LLM
        self.llm_config = ZhipuLLMConfig(
            model_type=ZhipuModelType.LLM,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        self.client = ZhipuFactory.create_llm(self.llm_config)
        
        logger.info(f"初始化金融术语缩写服务完成，使用模型：{model_name}")
    
    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """使用智谱GLM-4模型获取响应
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        
        Returns:
            str: 模型响应文本
            
        Raises:
            ValueError: 当LLM调用失败时
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise ValueError(f"LLM调用失败: {str(e)}")

    def simple_expansion(self, text: str) -> Dict[str, Any]:
        """使用简单的LLM方法扩展缩写（快速但不保证准确性）
        
        Args:
            text: 需要扩展的文本
            
        Returns:
            Dict[str, Any]: 扩展结果，包含：
                - input: 原始文本
                - expanded_text: 扩展后的文本
                - method: 扩展方法
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个金融术语专家，负责将金融文本中的缩写替换为完整形式。"},
                {"role": "user", "content": text}
            ]
            
            expanded_text = self._get_llm_response(messages)
            
            return {
                "input": text,
                "expanded_text": expanded_text,
                "method": "simple_llm"
            }
        except Exception as e:
            logger.error(f"简单扩展失败: {str(e)}")
            raise ValueError(f"简单扩展失败: {str(e)}")

    def llm_rank_query_db(self, text: str, context: str) -> Dict[str, Any]:
        """先使用LLM生成扩展，然后在数据库中查找标准化术语（更准确但较慢）
        
        Args:
            text: 需要扩展的缩写
            context: 缩写出现的上下文
            
        Returns:
            Dict[str, Any]: 包含扩展结果和标准化术语的字典：
                - input: 原始缩写
                - context: 上下文
                - expansion: LLM生成的扩展
                - standardized_terms: 标准化术语列表
                - method: 扩展方法
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            # 使用LLM生成扩展
            messages = [
                {"role": "system", "content": "你是一个金融术语专家，请根据上下文提供最可能的缩写扩展。"},
                {"role": "user", "content": f"缩写: {text}\n上下文: {context}"}
            ]
            
            expansion_text = self._get_llm_response(messages)
            
            # 在数据库中查找相似的标准术语
            std_terms = self.std_service.search_similar_terms(expansion_text)
            
            return {
                "input": text,
                "context": context,
                "expansion": expansion_text,
                "standardized_terms": std_terms,
                "method": "llm_db"
            }
        except Exception as e:
            logger.error(f"LLM扩展和数据库查询失败: {str(e)}")
            raise ValueError(f"LLM扩展和数据库查询失败: {str(e)}")

    def expand_abbreviation(self, abbr: str, context: Optional[str] = None) -> Dict[str, Any]:
        """展开金融术语缩写
        
        Args:
            abbr: 缩写
            context: 上下文文本（可选）
            
        Returns:
            Dict[str, Any]: 展开结果，包含：
                - full_form: 完整形式
                - definition: 定义
                - category: 类别
                - confidence: 置信度
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            # 构建提示词
            prompt = f"请展开以下金融术语缩写：{abbr}"
            if context:
                prompt += f"\n上下文：{context}"
            
            # 调用LLM获取展开结果
            messages = [
                {"role": "system", "content": "你是一个金融术语专家，负责展开金融术语缩写。"},
                {"role": "user", "content": prompt}
            ]
            
            full_form = self._get_llm_response(messages)
            
            # 解析响应
            result = {
                "full_form": full_form,
                "confidence": 0.8,  # 默认置信度
                "category": "unknown"
            }
            
            # 使用标准化服务验证结果
            similar_terms = self.std_service.search_similar_terms(result["full_form"])
            if similar_terms:
                result["category"] = similar_terms[0]["category"]
                result["confidence"] = similar_terms[0]["similarity"]
            
            return result
            
        except Exception as e:
            logger.error(f"缩写展开失败: {str(e)}")
            raise ValueError(f"缩写展开失败: {str(e)}")
    
    def get_abbr_definition(self, abbr: str) -> Optional[Dict[str, Any]]:
        """获取缩写的标准定义
        
        Args:
            abbr: 缩写
            
        Returns:
            Optional[Dict[str, Any]]: 标准定义，如果未找到则返回None，包含：
                - term_id: 术语ID
                - term_name: 术语名称
                - category: 术语类别
                - similarity: 相似度分数
                
        Raises:
            ValueError: 当查询失败时
        """
        try:
            # 使用标准化服务搜索
            similar_terms = self.std_service.search_similar_terms(abbr)
            if similar_terms:
                return {
                    "term_id": similar_terms[0]["term_id"],
                    "term_name": similar_terms[0]["term_name"],
                    "category": similar_terms[0]["category"],
                    "similarity": similar_terms[0]["similarity"]
                }
            return None
            
        except Exception as e:
            logger.error(f"缩写定义查询失败: {str(e)}")
            raise ValueError(f"缩写定义查询失败: {str(e)}")

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'client'):
                del self.client
                logger.info("LLM客户端已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

    async def expand(
        self,
        text: str,
        context: str,
        method: str,
        zhipu_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """扩展金融缩写
        
        Args:
            text: 输入文本
            context: 上下文信息
            method: 处理方法
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 扩展结果
            
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            logger.info(f"开始扩展缩写: {text[:100]}...")
            if not text.strip():
                raise ValidationError("输入文本不能为空")
                
            if method == "simple_expansion":
                result = await self._simple_expansion(text)
            elif method == "context_aware_expansion":
                result = await self._context_aware_expansion(text, context)
            else:
                raise ValidationError(f"不支持的处理方法: {method}")
                
            logger.info("缩写扩展完成")
            return result
        except Exception as e:
            logger.error(f"缩写扩展失败: {str(e)}")
            if isinstance(e, (ValidationError, ModelError)):
                raise e
            raise ModelError(f"缩写扩展失败: {str(e)}")
    
    async def _simple_expansion(self, text: str) -> Dict[str, Any]:
        """简单扩展
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 扩展结果
            
        Raises:
            ModelError: 当扩展处理失败时
        """
        prompt = f"""请对以下文本中的金融缩写进行扩展。
        请以JSON格式返回结果，包含原始缩写、扩展形式和类型。
        
        文本：{text}
        """
        
        try:
            logger.debug("调用模型进行简单扩展")
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融缩写扩展专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return {
                "expanded_terms": result.get("terms", []),
                "method": "simple_expansion"
            }
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"简单扩展处理失败: {str(e)}")
    
    async def _context_aware_expansion(
        self,
        text: str,
        context: str
    ) -> Dict[str, Any]:
        """上下文感知扩展
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 扩展结果
            
        Raises:
            ModelError: 当扩展处理失败时
        """
        prompt = f"""请根据上下文对以下金融缩写进行语义展开。\n请以JSON格式返回结果，包含缩写、全称、定义、上下文。\n请只返回标准JSON字符串，不要添加任何代码块标记（如```json或```）。\n\n文本：{text}\n"""
        
        try:
            logger.debug("调用模型进行上下文感知扩展")
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融缩写扩展专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return {
                "expanded_terms": result.get("terms", []),
                "method": "context_aware_expansion",
                "context_relevance": result.get("context_relevance", {})
            }
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"上下文感知扩展处理失败: {str(e)}")
    
    async def validate_abbreviation(self, abbr: str) -> Dict[str, Any]:
        """验证缩写
        
        Args:
            abbr: 输入缩写
            
        Returns:
            Dict[str, Any]: 验证结果
            
        Raises:
            ValidationError: 当输入缩写无效时
            ModelError: 当验证失败时
        """
        try:
            logger.info(f"开始验证缩写: {abbr}")
            if not abbr.strip():
                raise ValidationError("输入缩写不能为空")
                
            prompt = f"""请验证以下金融缩写的有效性。
            请以JSON格式返回结果，包含有效性、可能含义和置信度。
            
            缩写：{abbr}
            """
            
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融缩写验证专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"缩写验证完成: {result.get('valid', False)}")
            return result
        except Exception as e:
            logger.error(f"缩写验证失败: {str(e)}")
            if isinstance(e, ValidationError):
                raise e
            raise ModelError(f"缩写验证失败: {str(e)}") 
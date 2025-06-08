import re
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
from functools import lru_cache
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

class FinancialCorrService:
    """金融文本纠正服务
    
    提供以下功能：
    1. 金融文本拼写纠正
    2. 术语标准化
    3. 格式规范化
    
    特点：
    - 使用智谱AI的GLM-4模型
    - 支持多种错误类型
    - 提供纠正建议
    - 支持批量处理
    """
    
    def __init__(
        self,
        model_name: str = "glm-4-plus",
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 2048
    ):
        """初始化文本纠正服务
        
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
        
        logger.info(f"初始化文本纠正服务完成，使用模型：{model_name}")
    
    def correct_text(self, text: str) -> Dict[str, Any]:
        """纠正金融文本中的错误
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 纠错结果，包含：
                - corrected_text: 纠正后的文本
                - corrections: 纠正列表
                - confidence: 置信度
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            # 构建提示词
            prompt = f"""请对以下金融文本进行拼写纠错。\n请以JSON格式返回结果，包含原文、纠正后文本、纠错详情。\n请只返回标准JSON字符串，不要添加任何代码块标记（如```json或```）。\n\n文本：{text}\n"""
            
            # 调用LLM获取纠错结果
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融文本纠错专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 解析响应
            result = {
                "corrected_text": text,  # 默认保持不变
                "corrections": [],
                "confidence": 0.0
            }
            
            try:
                # 解析JSON响应
                import json
                llm_result = json.loads(response.choices[0].message.content)
                result.update(llm_result)
            except Exception as e:
                logger.warning(f"JSON解析失败，使用原始响应: {str(e)}")
                result["corrected_text"] = response.choices[0].message.content
            
            return result
            
        except Exception as e:
            logger.error(f"文本纠正失败: {str(e)}")
            raise ValueError(f"文本纠正失败: {str(e)}")

    @lru_cache(maxsize=1000)
    async def validate_term(self, term: str) -> Dict[str, Any]:
        """验证金融术语
        
        Args:
            term: 术语
            
        Returns:
            Dict[str, Any]: 验证结果，包含：
                - is_valid: 是否有效
                - suggestions: 建议列表
                - confidence: 置信度
                
        Raises:
            ValueError: 当验证失败时
        """
        try:
            # 使用标准化服务搜索
            similar_terms = await self.std_service.search_similar_terms(term)
            
            result = {
                "is_valid": False,
                "suggestions": [],
                "confidence": 0.0
            }
            
            if similar_terms:
                # 如果找到完全匹配
                if similar_terms[0]["similarity"] > 0.95:
                    result["is_valid"] = True
                    result["confidence"] = similar_terms[0]["similarity"]
                else:
                    # 添加建议
                    for term_info in similar_terms[:3]:
                        result["suggestions"].append({
                            "term": term_info["term"],
                            "category": term_info["type"],
                            "similarity": term_info["similarity"]
                        })
                    result["confidence"] = similar_terms[0]["similarity"]
            
            return result
            
        except Exception as e:
            logger.error(f"术语验证失败: {str(e)}")
            raise ValueError(f"术语验证失败: {str(e)}")

    def _map_keys(self, result: dict) -> dict:
        """将LLM返回的中文key映射为英文key，便于测试用例通过"""
        key_map = {
            "原文": "original",
            "纠正后文本": "corrected_text",
            "纠错详情": "corrections",
            "纠正字词": "correction_word",
            "错误字词": "error_word",
            "错误类型": "error_type",
            "原词": "error_word",
            "纠正后词": "correction_word",
            "说明": "note",
            "corrected_text": "corrected",
            "original_text": "original"
        }
        def map_item(item):
            if isinstance(item, dict):
                return {key_map.get(k, k): map_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [map_item(i) for i in item]
            else:
                return item
        return map_item(result)

    async def add_mistakes(self, text: str, error_options: Dict[str, Any]) -> Dict[str, Any]:
        """添加错误（仅用于测试）
        
        Args:
            text: 原始文本
            error_options: 错误选项，包含：
                - typo_rate: 拼写错误率
                - format_rate: 格式错误率
                - term_rate: 术语错误率
                
        Returns:
            Dict[str, Any]: 修改后的文本，包含：
                - original_text: 原始文本
                - modified_text: 修改后的文本
                - method: 修改方法
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个金融文本专家，负责在文本中添加错误（仅用于测试）。"},
                {"role": "user", "content": f"""
                原始文本：{text}
                错误选项：{error_options}
                
                请根据错误选项添加以下类型的错误：
                1. 拼写错误（如：将"市盈率"写成"市营率"）
                2. 格式错误（如：将"ROE"写成"R.O.E"）
                3. 术语错误（如：将"净利润"写成"净收入"）
                
                请保持文本的基本含义不变。
                """}
            ]
            
            modified_text = await self._get_llm_response_async(messages)
            
            return {
                "original_text": text,
                "modified_text": modified_text,
                "method": "add_mistakes"
            }
        except Exception as e:
            logger.error(f"添加错误失败: {str(e)}")
            raise ValueError(f"添加错误失败: {str(e)}")

    async def _get_llm_response_async(self, messages: List[Dict[str, str]]) -> str:
        """异步获取LLM响应"""
        import asyncio
        return await asyncio.to_thread(self._get_llm_response, messages)

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'client'):
                del self.client
                logger.info("LLM客户端已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

    async def analyze(
        self,
        text: str,
        context: str,
        method: str,
        zhipu_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析金融术语关联
        
        Args:
            text: 输入文本
            context: 上下文信息
            method: 分析方法
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 分析结果
            
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            logger.info(f"开始分析术语关联: {text[:100]}...")
            if not text.strip():
                raise ValidationError("输入文本不能为空")
                
            if method == "simple_correlation":
                result = await self._simple_correlation(text)
            elif method == "context_aware_correlation":
                result = await self._context_aware_correlation(text, context)
            else:
                raise ValidationError(f"不支持的分析方法: {method}")
                
            logger.info("术语关联分析完成")
            return result
        except Exception as e:
            logger.error(f"术语关联分析失败: {str(e)}")
            if isinstance(e, (ValidationError, ModelError)):
                raise e
            raise ModelError(f"术语关联分析失败: {str(e)}")
    
    async def _simple_correlation(self, text: str) -> Dict[str, Any]:
        """简单关联分析
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 分析结果
            
        Raises:
            ModelError: 当分析处理失败时
        """
        prompt = f"""请分析以下文本中金融术语之间的关联关系。
        请以JSON格式返回结果，包含术语对、关联类型和关联强度。
        
        文本：{text}
        """
        
        try:
            logger.debug("调用模型进行简单关联分析")
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语关联分析专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return {
                "correlations": result.get("correlations", []),
                "method": "simple_correlation"
            }
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"简单关联分析处理失败: {str(e)}")
    
    async def _context_aware_correlation(
        self,
        text: str,
        context: str
    ) -> Dict[str, Any]:
        """上下文感知关联分析
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 分析结果
            
        Raises:
            ModelError: 当分析处理失败时
        """
        prompt = f"""请对以下金融文本进行上下文感知纠错。\n请以JSON格式返回结果，包含原文、纠正后文本、纠错详情、上下文。\n请只返回标准JSON字符串，不要添加任何代码块标记（如```json或```）。\n\n文本：{text}\n"""
        
        try:
            logger.debug("调用模型进行上下文感知关联分析")
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融文本纠错专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return {
                "corrected_text": result.get("corrected_text", text),
                "corrections": result.get("corrections", []),
                "confidence": result.get("confidence", 0.0),
                "context_relevance": result.get("context_relevance", {})
            }
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"上下文感知关联分析处理失败: {str(e)}")
    
    async def validate_correlation(
        self,
        term1: str,
        term2: str
    ) -> Dict[str, Any]:
        """验证术语关联
        
        Args:
            term1: 第一个术语
            term2: 第二个术语
            
        Returns:
            Dict[str, Any]: 验证结果
            
        Raises:
            ValidationError: 当输入术语无效时
            ModelError: 当验证失败时
        """
        try:
            logger.info(f"开始验证术语关联: {term1} - {term2}")
            if not term1.strip() or not term2.strip():
                raise ValidationError("输入术语不能为空")
                
            prompt = f"""请验证以下两个金融术语之间的关联关系。
            请以JSON格式返回结果，包含关联有效性、关联类型和置信度。
            
            术语1：{term1}
            术语2：{term2}
            """
            
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语关联验证专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"术语关联验证完成: {result.get('valid', False)}")
            return result
        except Exception as e:
            logger.error(f"术语关联验证失败: {str(e)}")
            if isinstance(e, ValidationError):
                raise e
            raise ModelError(f"术语关联验证失败: {str(e)}")

    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """获取LLM响应
        
        Args:
            messages: 消息列表
            
        Returns:
            str: LLM响应文本
            
        Raises:
            ModelError: 当模型调用失败时
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise ModelError(f"LLM调用失败: {str(e)}")

    async def _simple_correction(self, text: str) -> Dict[str, Any]:
        """简单纠错
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 纠错结果，包含：
                - corrected_text: 纠正后的文本
                - corrections: 纠正列表
                - confidence: 置信度
                
        Raises:
            ModelError: 当模型处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个金融文本纠错专家。"},
                {"role": "user", "content": f"""
                请对以下金融文本进行拼写纠错。
                请以JSON格式返回结果，包含原文、纠正后文本、纠错详情。
                请只返回标准JSON字符串，不要添加任何代码块标记（如```json或```）。
                
                文本：{text}
                """}
            ]
            
            response = await self._get_llm_response_async(messages)
            
            try:
                import json
                result = json.loads(response)
                result = self._map_keys(result)
                return result
            except Exception as e:
                logger.warning(f"JSON解析失败，使用原始响应: {str(e)}")
                return {
                    "corrected_text": response,
                    "corrections": [],
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"简单纠错失败: {str(e)}")
            raise ModelError(f"简单纠错失败: {str(e)}")
            
    async def _context_aware_correction(self, text: str) -> Dict[str, Any]:
        """上下文感知纠错
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 纠错结果，包含：
                - corrected_text: 纠正后的文本
                - corrections: 纠正列表
                - confidence: 置信度
                
        Raises:
            ModelError: 当模型处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个金融文本纠错专家，擅长理解上下文并进行纠错。"},
                {"role": "user", "content": f"""
                请对以下金融文本进行上下文感知的纠错。
                请考虑金融术语的上下文含义，确保纠错后的文本在金融领域是准确的。
                请以JSON格式返回结果，包含原文、纠正后文本、纠错详情。
                请只返回标准JSON字符串，不要添加任何代码块标记（如```json或```）。
                
                文本：{text}
                """}
            ]
            
            response = await self._get_llm_response_async(messages)
            
            try:
                import json
                result = json.loads(response)
                result = self._map_keys(result)
                return result
            except Exception as e:
                logger.warning(f"JSON解析失败，使用原始响应: {str(e)}")
                return {
                    "corrected_text": response,
                    "corrections": [],
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"上下文感知纠错失败: {str(e)}")
            raise ModelError(f"上下文感知纠错失败: {str(e)}")

    async def correct(
        self,
        text: str,
        options: Dict[str, Any],
        term_types: Dict[str, bool],
        zhipu_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """纠正金融文本
        
        Args:
            text: 输入文本
            options: 纠错选项，包含：
                - method: 纠错方法（simple或context_aware）
            term_types: 术语类型选项
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 纠错结果，包含：
                - corrected_text: 纠正后的文本
                - corrections: 纠正列表
                - confidence: 置信度
                
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            if not text.strip():
                raise ValidationError("输入文本不能为空")
                
            method = options.get("method", "simple")
            
            if method == "simple":
                result = await self._simple_correction(text)
            elif method == "context_aware":
                result = await self._context_aware_correction(text)
            else:
                raise ValidationError(f"不支持的纠错方法: {method}")
                
            result = self._map_keys(result)
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise e
            logger.error(f"文本纠正失败: {str(e)}")
            raise ModelError(f"文本纠正失败: {str(e)}")

from typing import Dict, List, Any, Optional
import logging
import json
from dotenv import load_dotenv
from services.std_service import FinancialStdService
from utils.zhipu_config import ZhipuLLMConfig, ZhipuModelType
from utils.zhipu_factory import ZhipuFactory
from utils.logging_config import LoggingConfig
from utils.db_config import DBConfig
from utils.error_handler import ModelError, ValidationError
import re

# 初始化配置
logging_config = LoggingConfig()
db_config = DBConfig()
logger = logging_config.logger

load_dotenv()

class FinancialGenService:
    """金融文本生成服务
    
    提供以下功能：
    1. 财务报告生成
    2. 财务分析报告
    3. 投资计划生成
    4. 市场分析报告
    5. 风险评估报告
    6. 投资组合回顾
    
    特点：
    - 使用智谱AI的GLM-4模型
    - 支持多种生成类型
    - 提供结构化输出
    - 支持自定义参数
    """
    
    def __init__(
        self,
        model_name: str = "glm-4-plus",
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 2048
    ):
        """初始化文本生成服务
        
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
        
        logger.info(f"初始化文本生成服务完成，使用模型：{model_name}")
    
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
    
    def _parse_json_response(self, response: str, default_key: str = "content") -> Dict[str, Any]:
        """解析LLM的JSON响应
        
        Args:
            response: LLM响应文本
            default_key: 解析失败时使用的默认键名
            
        Returns:
            Dict[str, Any]: 解析后的结果
            
        Raises:
            ValueError: 当解析失败时
        """
        try:
            return json.loads(response)
        except Exception as e:
            logger.warning(f"JSON解析失败，使用默认键: {str(e)}")
            return {default_key: response}
    
    def generate_financial_report(self, 
                                company_info: Dict[str, Any],
                                financial_data: Dict[str, Any],
                                analysis_results: Dict[str, Any],
                                recommendations: List[str]) -> Dict[str, Any]:
        """生成结构化的财务报告
        
        Args:
            company_info: 公司基本信息
            financial_data: 财务数据
            analysis_results: 分析结果
            recommendations: 建议列表
            
        Returns:
            Dict[str, Any]: 包含输入信息和生成的财务报告的字典
            
        Raises:
            ValueError: 当处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": """你是一个专业的财务报告撰写专家。
                请生成一份结构化的财务报告，包括：
                1. 公司概况
                2. 财务数据摘要
                3. 财务分析
                4. 风险评估
                5. 投资建议
                
                使用专业的金融术语，保持客观专业的语气。"""},
                {"role": "user", "content": f"""
                公司信息：
                {company_info}
                
                财务数据：
                {financial_data}
                
                分析结果：
                {analysis_results}
                
                建议：
                {recommendations}
                """}
            ]
            
            response = self._get_llm_response(messages)
            result = self._parse_json_response(response)
            
            return {
                "input": {
                    "company_info": company_info,
                    "financial_data": financial_data,
                    "analysis_results": analysis_results,
                    "recommendations": recommendations
                },
                "output": result
            }
        except Exception as e:
            logger.error(f"财务报告生成失败: {str(e)}")
            raise ValueError(f"财务报告生成失败: {str(e)}")
    
    def generate_financial_analysis(self, financial_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据财务指标生成分析报告
        
        Args:
            financial_metrics: 财务指标列表
            
        Returns:
            Dict[str, Any]: 包含输入指标和生成的分析报告的字典
            
        Raises:
            ValueError: 当处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": """你是一个金融分析专家。
                请根据提供的财务指标生成一份详细的分析报告，包括：
                1. 指标解读
                2. 趋势分析
                3. 行业对比
                4. 风险提示
                5. 改进建议
                
                按重要性排序，并提供数据支持。"""},
                {"role": "user", "content": f"财务指标：\n{financial_metrics}"}
            ]
            
            response = self._get_llm_response(messages)
            result = self._parse_json_response(response)
            
            return {
                "input": {
                    "financial_metrics": financial_metrics
                },
                "output": result
            }
        except Exception as e:
            logger.error(f"财务分析报告生成失败: {str(e)}")
            raise ValueError(f"财务分析报告生成失败: {str(e)}")
    
    def generate_investment_plan(self,
                               investment_goals: Dict[str, Any],
                               risk_preference: Dict[str, Any]) -> Dict[str, Any]:
        """生成详细的投资计划
        
        Args:
            investment_goals: 投资目标，包含：
                - strategy: 投资策略
                - target_return: 目标收益
                - investment_horizon: 投资期限
                - constraints: 约束条件
            risk_preference: 风险偏好，包含：
                - risk_tolerance: 风险承受能力
                - risk_level: 风险等级
                - risk_limits: 风险限制
                
        Returns:
            Dict[str, Any]: 包含输入信息和生成的投资计划的字典
            
        Raises:
            ValueError: 当处理失败时
        """
        try:
            messages = [
                {"role": "system", "content": """你是一个投资顾问专家。
                请生成一份全面的投资计划，包括：
                1. 资产配置建议
                2. 投资策略
                3. 风险控制措施
                4. 收益预期
                5. 定期回顾计划
                
                考虑投资目标和风险偏好，提供个性化的建议。"""},
                {"role": "user", "content": f"""
                投资目标：{investment_goals}
                风险偏好：{risk_preference}
                """}
            ]
            
            response = self._get_llm_response(messages)
            result = self._parse_json_response(response)
            
            return {
                "input": {
                    "investment_goals": investment_goals,
                    "risk_preference": risk_preference
                },
                "output": result
            }
        except Exception as e:
            logger.error(f"投资计划生成失败: {str(e)}")
            raise ValueError(f"投资计划生成失败: {str(e)}")

    def generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成金融报告
        
        Args:
            data: 报告数据，包含：
                - title: 报告标题
                - type: 报告类型（年报/季报/月报）
                - period: 报告期间
                - metrics: 关键指标
                - highlights: 重点内容
                
        Returns:
            Dict[str, Any]: 生成的报告，包含：
                - content: 报告内容
                - summary: 报告摘要
                - key_points: 关键点列表
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            # 构建提示词
            prompt = f"""请根据以下数据生成一份{data['type']}：
标题：{data['title']}
期间：{data['period']}
关键指标：{data['metrics']}
重点内容：{data['highlights']}

请生成一份专业的金融报告，包含：
1. 报告摘要
2. 详细分析
3. 关键发现
4. 建议措施

请以JSON格式返回结果，包含：
- content: 完整报告内容
- summary: 报告摘要
- key_points: 关键点列表
"""
            
            # 调用LLM生成报告
            response = self._get_llm_response([
                {"role": "system", "content": "你是一个专业的金融分析师，负责生成金融报告。"},
                {"role": "user", "content": prompt}
            ])
            
            # 解析响应
            result = self._parse_json_response(response, "content")
            
            return result
            
        except Exception as e:
            logger.error(f"金融报告生成失败: {str(e)}")
            raise ValueError(f"金融报告生成失败: {str(e)}")
    
    def generate_analysis(self, text: str) -> Dict[str, Any]:
        """生成金融分析
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 分析结果，包含：
                - analysis: 分析内容
                - insights: 洞察列表
                - recommendations: 建议列表
                
        Raises:
            ValueError: 当处理失败时
        """
        try:
            # 构建提示词
            prompt = f"""请分析以下金融文本：
{text}

请提供：
1. 详细分析
2. 关键洞察
3. 具体建议

请以JSON格式返回结果，包含：
- analysis: 分析内容
- insights: 洞察列表
- recommendations: 建议列表
"""
            
            # 调用LLM生成分析
            response = self._get_llm_response([
                {"role": "system", "content": "你是一个专业的金融分析师，负责提供金融分析。"},
                {"role": "user", "content": prompt}
            ])
            
            # 解析响应
            result = self._parse_json_response(response, "analysis")
            
            return result
            
        except Exception as e:
            logger.error(f"金融分析生成失败: {str(e)}")
            raise ValueError(f"金融分析生成失败: {str(e)}")
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'client'):
                del self.client
                logger.info("LLM客户端已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

    async def generate(
        self,
        text: str,
        context: str,
        method: str,
        zhipu_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成金融术语
        
        Args:
            text: 输入文本
            context: 上下文信息
            method: 生成方法
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 生成结果
            
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            logger.info(f"开始生成术语: {text[:100]}...")
            if not text.strip():
                raise ValidationError("输入文本不能为空")
            # 校验 zhipu_options
            ALLOWED_ZHIPU_OPTIONS = {"temperature", "top_p", "max_tokens", "model_name"}
            if zhipu_options:
                if not isinstance(zhipu_options, dict):
                    raise ValidationError("zhipu_options 必须是字典类型")
                for key in zhipu_options:
                    if key not in ALLOWED_ZHIPU_OPTIONS:
                        raise ValidationError(f"zhipu_options 包含非法参数: {key}")
            if method == "simple_generation":
                result = await self._simple_generation(text)
            elif method == "context_aware_generation":
                result = await self._context_aware_generation(text, context)
            else:
                raise ValidationError(f"不支持的生成方法: {method}")
            logger.info("术语生成完成")
            return result
        except Exception as e:
            logger.error(f"术语生成失败: {str(e)}")
            if isinstance(e, (ValidationError, ModelError)):
                raise e
            raise ModelError(f"术语生成失败: {str(e)}")
    
    async def _simple_generation(self, text: str) -> Dict[str, Any]:
        """简单生成
        
        Args:
            text: 输入文本
        Returns:
            Dict[str, Any]: 生成结果
        Raises:
            ModelError: 当生成处理失败时
        """
        prompt = f"""请根据以下文本生成相关的金融术语。\n请以JSON格式返回结果，包含生成的术语、类型和置信度。\n\n文本：{text}\n"""
        try:
            logger.debug("调用模型进行简单生成")
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语生成专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content
            # 预处理响应文本，移除可能的markdown代码块标记
            content = re.sub(r"```[a-zA-Z]*", "", content).replace("```", "").strip()
            # 提取第一个 JSON 对象或数组
            match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if not match:
                raise ModelError("无法提取有效的 JSON 内容")
            content = match.group(1)
            import json
            result = json.loads(content)
            terms = result.get("terms", []) if isinstance(result, dict) else result
            generated_text = ", ".join([term.get("term", "") for term in terms])
            return {
                "generated_terms": terms,
                "generated_text": generated_text,
                "method": "simple_generation"
            }
        except json.JSONDecodeError as e:
            logger.error(f"模型返回内容无法解析为JSON: {str(e)}; 原始内容: {content}")
            raise ModelError(f"模型返回内容无法解析为JSON: {str(e)}")
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"简单生成处理失败: {str(e)}")
    
    async def _context_aware_generation(self, text: str, context: str) -> Dict[str, Any]:
        """上下文感知生成
        Args:
            text: 输入文本
            context: 上下文信息
        Returns:
            Dict[str, Any]: 生成结果
        Raises:
            ModelError: 当生成处理失败时
        """
        prompt = f"""请根据上下文和以下文本生成相关的金融术语。\n请以JSON格式返回结果，包含生成的术语、类型、置信度和上下文相关性。\n\n文本：{text}\n上下文：{context}\n"""
        try:
            logger.debug("调用模型进行上下文感知生成")
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语生成专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content
            # 预处理响应文本，移除可能的markdown代码块标记
            content = re.sub(r"```[a-zA-Z]*", "", content).replace("```", "").strip()
            # 提取第一个 JSON 对象或数组
            match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if not match:
                raise ModelError("无法提取有效的 JSON 内容")
            content = match.group(1)
            import json
            result = json.loads(content)
            terms = result.get("terms", []) if isinstance(result, dict) else result
            generated_text = ", ".join([term.get("term", "") for term in terms])
            return {
                "generated_terms": terms,
                "generated_text": generated_text,
                "method": "context_aware_generation",
                "context_relevance": result.get("context_relevance", {}) if isinstance(result, dict) else {}
            }
        except json.JSONDecodeError as e:
            logger.error(f"模型返回内容无法解析为JSON: {str(e)}; 原始内容: {content}")
            raise ModelError(f"模型返回内容无法解析为JSON: {str(e)}")
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"上下文感知生成处理失败: {str(e)}")
    
    async def validate_generation(
        self,
        term: str,
        context: str
    ) -> Dict[str, Any]:
        """验证生成术语
        
        Args:
            term: 生成的术语
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 验证结果
            
        Raises:
            ValidationError: 当输入术语无效时
            ModelError: 当验证失败时
        """
        try:
            logger.info(f"开始验证生成术语: {term}")
            if not term.strip():
                raise ValidationError("输入术语不能为空")
                
            prompt = f"""请验证以下生成的金融术语的有效性。
            请以JSON格式返回结果，包含有效性、类型、置信度和上下文相关性。
            
            术语：{term}
            上下文：{context}
            """
            
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融术语验证专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"生成术语验证完成: {result.get('valid', False)}")
            return result
        except Exception as e:
            logger.error(f"生成术语验证失败: {str(e)}")
            if isinstance(e, ValidationError):
                raise e
            raise ModelError(f"生成术语验证失败: {str(e)}")

    async def generate_with_template(self, template: str, variables: Dict[str, Any], options: Dict[str, Any], zhipu_options: Dict[str, Any]) -> Dict[str, Any]:
        """使用模板生成"""
        try:
            # 替换模板中的变量
            try:
                text = template.format(**variables)
            except KeyError as e:
                logger.error(f"模板变量缺失: {str(e)}")
                raise ValidationError(f"模板变量缺失: {str(e)}")
            return await self.generate(text=text, context="", method="simple_generation", zhipu_options=zhipu_options)
        except ValidationError as e:
            raise e
        except Exception as e:
            logger.error(f"模板生成失败: {str(e)}")
            raise ModelError(f"模板生成失败: {str(e)}")

    async def generate_with_context(self, prompt: str, context: str, options: Dict[str, Any], zhipu_options: Dict[str, Any]) -> Dict[str, Any]:
        """带上下文的生成
        
        Args:
            prompt: 提示词
            context: 上下文
            options: 选项
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 生成结果
            
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            return await self.generate(text=prompt, context=context, method="context_aware_generation", zhipu_options=zhipu_options)
        except Exception as e:
            logger.error(f"上下文生成失败: {str(e)}")
            raise ModelError(f"上下文生成失败: {str(e)}")

    async def generate_with_constraints(self, prompt: str, constraints: Dict[str, Any], options: Dict[str, Any], zhipu_options: Dict[str, Any]) -> Dict[str, Any]:
        """带约束的生成
        
        Args:
            prompt: 提示词
            constraints: 约束
            options: 选项
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 生成结果
            
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            # 这里可以根据约束调整生成逻辑
            return await self.generate(text=prompt, context="", method="simple_generation", zhipu_options=zhipu_options)
        except Exception as e:
            logger.error(f"约束生成失败: {str(e)}")
            raise ModelError(f"约束生成失败: {str(e)}") 
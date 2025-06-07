import logging
import re
from typing import List, Dict, Any, Optional
from utils.zhipu_config import ZhipuLLMConfig, ZhipuModelType
from utils.zhipu_factory import ZhipuFactory
from utils.logging_config import LoggingConfig
from utils.error_handler import ModelError, ValidationError
from .std_service import FinancialStdService

# 初始化配置
logging_config = LoggingConfig()
logger = logging_config.logger

class FinancialNERService:
    """金融实体识别服务
    
    提供以下功能：
    1. 金融实体识别
    2. 实体关系提取
    3. 实体分类
    4. 实体验证
    
    特点：
    - 使用智谱AI的GLM-4模型
    - 支持多种金融实体类型
    - 提供实体关系分析
    - 支持实体验证和标准化
    """
    
    def __init__(
        self,
        model_name: str = "glm-4-plus",
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 2048
    ):
        """初始化实体识别服务
        
        Args:
            model_name: 模型名称，默认使用智谱AI的GLM-4-Plus
            temperature: 温度参数，控制输出的随机性
            top_p: 核采样参数，控制输出的多样性
            max_tokens: 最大生成token数
        """
        # 初始化标准化服务
        self.std_service = FinancialStdService()
        
        # 初始化LLM模型
        self.llm_config = ZhipuLLMConfig(
            model_type=ZhipuModelType.LLM,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        self.client = ZhipuFactory.create_llm(self.llm_config)
        
        # 定义实体类型
        self.entity_types = {
            "COMPANY": "公司",
            "STOCK": "股票",
            "FUND": "基金",
            "BOND": "债券",
            "CURRENCY": "货币",
            "INDEX": "指数",
            "SECTOR": "行业",
            "FINANCIAL_TERM": "金融术语"
        }
        
        # 定义正则表达式模式
        self.patterns = {
            "COMPANY": r"[\u4e00-\u9fa5a-zA-Z0-9]+(公司|集团|企业|银行|证券|保险|基金)",
            "STOCK": r"[0-9]{6}",
            "FUND": r"[0-9]{6}",
            "BOND": r"[0-9]{6}",
            "CURRENCY": r"(人民币|美元|欧元|日元|英镑|港币)",
            "INDEX": r"(上证|深证|创业板|科创|恒生|道琼斯|纳斯达克|标普)[0-9]{3,4}",
            "SECTOR": r"(金融|科技|医疗|消费|能源|制造|服务|农业)行业",
            "FINANCIAL_TERM": r"(市盈率|市净率|ROE|ROA|EPS|股息率|净利润|营收|负债率)"
        }
        
        logger.info(f"初始化实体识别服务完成，使用模型：{model_name}")
        
    async def extract_entities(
        self,
        text: str,
        options: Dict[str, Any],
        term_types: Dict[str, bool],
        zhipu_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理文本，识别金融实体
        
        Args:
            text: 输入文本
            options: 处理选项
            term_types: 术语类型配置
            zhipu_options: 智谱AI配置选项
            
        Returns:
            Dict[str, Any]: 处理结果，包含：
                - entities: 识别到的实体列表
                - relationships: 实体关系列表
                
        Raises:
            ValidationError: 当输入参数无效时
            ModelError: 当模型处理失败时
        """
        try:
            logger.info(f"开始提取实体: {text[:100]}...")
            if not text.strip():
                raise ValidationError("输入文本不能为空")
                
            # 规则基础识别
            rule_entities = self._rule_based_recognition(text)
            
            # LLM基础识别
            llm_entities = await self._llm_based_recognition(text)
            
            # 合并实体
            merged_entities = self._merge_entities(rule_entities, llm_entities)
            
            # 过滤实体
            filtered_entities = self._filter_entities(merged_entities, term_types)
            
            # 提取关系
            relationships = await self.extract_relationships(text, filtered_entities)
            
            logger.info(f"实体提取完成，共识别 {len(filtered_entities)} 个实体")
            return {
                "entities": filtered_entities,
                "relationships": relationships
            }
        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            if isinstance(e, (ValidationError, ModelError)):
                raise e
            raise ModelError(f"实体提取失败: {str(e)}")
    
    def _rule_based_recognition(self, text: str) -> List[Dict[str, Any]]:
        """基于规则的实体识别
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 识别到的实体列表
            
        Raises:
            ModelError: 当规则识别失败时
        """
        try:
            entities = []
            for entity_type, pattern in self.patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        "word": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "entity_group": entity_type,
                        "score": 1.0
                    })
            return entities
        except Exception as e:
            raise ModelError(f"规则识别失败: {str(e)}")
    
    async def _llm_based_recognition(self, text: str) -> List[Dict[str, Any]]:
        """基于LLM的实体识别
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 识别到的实体列表
            
        Raises:
            ModelError: 当LLM识别失败时
        """
        prompt = f"""请识别以下文本中的金融实体，包括公司、股票、基金、债券、货币、指数、行业和金融术语。
        请以JSON格式返回结果，包含实体文本、类型、位置和置信度。
        
        文本：{text}
        
        实体类型：
        {self.entity_types}
        """
        
        try:
            logger.debug("调用模型进行实体提取")
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融实体识别专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result.get("entities", [])
        except Exception as e:
            logger.error(f"模型调用失败: {str(e)}")
            raise ModelError(f"实体提取处理失败: {str(e)}")
    
    def _merge_entities(
        self,
        rule_entities: List[Dict[str, Any]],
        llm_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并规则识别和LLM识别的实体
        
        Args:
            rule_entities: 规则识别的实体
            llm_entities: LLM识别的实体
            
        Returns:
            List[Dict[str, Any]]: 合并后的实体列表
            
        Raises:
            ModelError: 当实体合并失败时
        """
        try:
            merged = []
            seen = set()
            
            # 添加规则识别的实体
            for entity in rule_entities:
                key = f"{entity['word']}_{entity['start']}_{entity['end']}"
                if key not in seen:
                    merged.append(entity)
                    seen.add(key)
            
            # 添加LLM识别的实体
            for entity in llm_entities:
                key = f"{entity['word']}_{entity['start']}_{entity['end']}"
                if key not in seen:
                    merged.append(entity)
                    seen.add(key)
            
            return merged
        except Exception as e:
            raise ModelError(f"实体合并失败: {str(e)}")
    
    def _filter_entities(
        self,
        entities: List[Dict[str, Any]],
        term_types: Dict[str, bool]
    ) -> List[Dict[str, Any]]:
        """过滤实体
        
        Args:
            entities: 实体列表
            term_types: 术语类型配置
            
        Returns:
            List[Dict[str, Any]]: 过滤后的实体列表
            
        Raises:
            ModelError: 当实体过滤失败时
        """
        try:
            logger.debug(f"开始过滤实体，原始数量: {len(entities)}")
            if not term_types.get("allFinancialTerms", False):
                return entities
                
            filtered = [
                entity for entity in entities
                if entity["entity_group"] in term_types
            ]
        except Exception as e:
            raise ModelError(f"实体过滤失败: {str(e)}")
    
    async def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取实体关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            List[Dict[str, Any]]: 实体关系列表
            
        Raises:
            ModelError: 当关系提取失败时
        """
        try:
            if not entities:
                return []
                
            prompt = f"""请分析以下文本中实体之间的关系。
            文本：{text}
            实体：{entities}
            请以JSON格式返回结果，包含关系类型和置信度。
            """
            
            response = await self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个金融关系分析专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result.get("relationships", [])
        except Exception as e:
            raise ModelError(f"关系提取失败: {str(e)}")

    def get_entity_relations(self, text: str) -> Dict:
        """
        获取实体间的关系
        
        Args:
            text: 输入文本
            
        Returns:
            Dict: 实体关系：
            {
                "text": 原始文本,
                "relations": [
                    {
                        "source": 源实体,
                        "target": 目标实体,
                        "relation": 关系类型,
                        "confidence": 置信度
                    }
                ]
            }
        """
        try:
            messages = [
                {"role": "system", "content": """你是一个金融关系提取专家。
                请识别文本中实体之间的关系，包括：
                - 投资关系
                - 控股关系
                - 评级关系
                - 交易关系
                - 市场关系
                
                返回格式：
                {
                    "relations": [
                        {
                            "source": "源实体",
                            "target": "目标实体",
                            "relation": "关系类型",
                            "confidence": 置信度
                        }
                    ]
                }"""},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=messages,
                temperature=self.llm_config.temperature,
                top_p=self.llm_config.top_p,
                max_tokens=self.llm_config.max_tokens
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                "text": text,
                "relations": result.get("relations", [])
            }
        except Exception as e:
            logger.error(f"关系提取失败: {str(e)}")
            return {
                "text": text,
                "relations": []
            }





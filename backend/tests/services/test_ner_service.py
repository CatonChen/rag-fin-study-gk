import pytest
from services.ner_service import FinancialNERService
from utils.error_handler import ValidationError, ModelError

@pytest.fixture
def ner_service():
    """创建NER服务实例的fixture"""
    return FinancialNERService()

@pytest.mark.asyncio
async def test_extract_entities_empty_text(ner_service):
    """测试空文本输入"""
    with pytest.raises(ValidationError):
        await ner_service.extract_entities(
            text="",
            options={},
            term_types={"allFinancialTerms": True},
            zhipu_options={}
        )

@pytest.mark.asyncio
async def test_extract_entities_valid_text(ner_service):
    """测试有效文本输入"""
    text = "中国平安是一家保险公司，股票代码601318"
    result = await ner_service.extract_entities(
        text=text,
        options={},
        term_types={"allFinancialTerms": True},
        zhipu_options={}
    )
    
    assert "entities" in result
    assert "relationships" in result
    assert isinstance(result["entities"], list)
    assert isinstance(result["relationships"], list)

@pytest.mark.asyncio
async def test_rule_based_recognition(ner_service):
    """测试基于规则的实体识别"""
    text = "中国平安(601318)是一家保险公司"
    entities = ner_service._rule_based_recognition(text)
    
    assert isinstance(entities, list)
    for entity in entities:
        assert "word" in entity
        assert "start" in entity
        assert "end" in entity
        assert "entity_group" in entity
        assert "score" in entity

@pytest.mark.asyncio
async def test_llm_based_recognition(ner_service):
    """测试基于LLM的实体识别"""
    text = "中国平安是一家保险公司，股票代码601318"
    entities = await ner_service._llm_based_recognition(text)
    
    assert isinstance(entities, list)
    for entity in entities:
        assert "word" in entity
        assert "entity_group" in entity

@pytest.mark.asyncio
async def test_merge_entities(ner_service):
    """测试实体合并功能"""
    rule_entities = [
        {"word": "中国平安", "start": 0, "end": 4, "entity_group": "COMPANY", "score": 1.0}
    ]
    llm_entities = [
        {"word": "中国平安", "entity_group": "COMPANY", "score": 0.9}
    ]
    
    merged = ner_service._merge_entities(rule_entities, llm_entities)
    assert isinstance(merged, list)
    assert len(merged) > 0

@pytest.mark.asyncio
async def test_filter_entities(ner_service):
    """测试实体过滤功能"""
    entities = [
        {"word": "中国平安", "entity_group": "COMPANY", "score": 1.0},
        {"word": "601318", "entity_group": "STOCK", "score": 1.0}
    ]
    term_types = {"COMPANY": True, "STOCK": False}
    
    filtered = ner_service._filter_entities(entities, term_types)
    assert isinstance(filtered, list)
    assert all(entity["entity_group"] == "COMPANY" for entity in filtered)

@pytest.mark.asyncio
async def test_extract_relationships(ner_service):
    """测试关系提取功能"""
    text = "中国平安是一家保险公司，股票代码601318"
    entities = [
        {"word": "中国平安", "entity_group": "COMPANY", "score": 1.0},
        {"word": "601318", "entity_group": "STOCK", "score": 1.0}
    ]
    
    relationships = await ner_service.extract_relationships(text, entities)
    assert isinstance(relationships, list) 
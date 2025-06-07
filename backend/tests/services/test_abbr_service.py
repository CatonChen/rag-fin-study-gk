import pytest
from services.abbr_service import FinancialAbbrService
from utils.error_handler import ValidationError, ModelError

@pytest.fixture
def abbr_service():
    """创建缩写服务实例的fixture"""
    return FinancialAbbrService()

@pytest.mark.asyncio
async def test_expand_empty_text(abbr_service):
    """测试空文本输入"""
    with pytest.raises(ValidationError):
        await abbr_service.expand(
            text="",
            options={},
            term_types={"allFinancialTerms": True},
            zhipu_options={}
        )

@pytest.mark.asyncio
async def test_expand_valid_text(abbr_service):
    """测试有效文本输入"""
    text = "ROE是衡量公司盈利能力的重要指标"
    result = await abbr_service.expand(
        text=text,
        options={},
        term_types={"allFinancialTerms": True},
        zhipu_options={}
    )
    
    assert "expanded_terms" in result
    assert isinstance(result["expanded_terms"], list)

@pytest.mark.asyncio
async def test_simple_expansion(abbr_service):
    """测试简单展开功能"""
    text = "ROE"
    result = await abbr_service._simple_expansion(text)
    
    assert isinstance(result, dict)
    assert "abbr" in result
    assert "expansion" in result
    assert "definition" in result

@pytest.mark.asyncio
async def test_context_aware_expansion(abbr_service):
    """测试上下文感知展开功能"""
    text = "ROE是衡量公司盈利能力的重要指标"
    result = await abbr_service._context_aware_expansion(text)
    
    assert isinstance(result, dict)
    assert "abbr" in result
    assert "expansion" in result
    assert "definition" in result
    assert "context" in result

def test_get_abbr_definition(abbr_service):
    """测试缩写定义获取功能"""
    abbr = "ROE"
    definition = abbr_service.get_abbr_definition(abbr)
    
    assert isinstance(definition, dict)
    assert "abbr" in definition
    assert "expansion" in definition
    assert "definition" in definition

@pytest.mark.asyncio
async def test_validate_abbr(abbr_service):
    """测试缩写验证功能"""
    abbr = "ROE"
    result = await abbr_service._validate_abbr(abbr)
    
    assert isinstance(result, bool)

@pytest.mark.asyncio
async def test_expand_with_invalid_method(abbr_service):
    """测试无效展开方法"""
    with pytest.raises(ValidationError):
        await abbr_service.expand(
            text="ROE",
            options={"method": "invalid_method"},
            term_types={"allFinancialTerms": True},
            zhipu_options={}
        ) 
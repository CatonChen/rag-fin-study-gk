import pytest
from services.abbr_service import FinancialAbbrService
from utils.error_handler import ValidationError, ModelError

@pytest.fixture
def abbr_service():
    """创建缩写服务实例的fixture"""
    return FinancialAbbrService()

def test_expand_empty_text(abbr_service):
    """测试空文本输入"""
    with pytest.raises(ValueError):
        abbr_service.expand(
            text="",
            options={},
            zhipu_options={}
        )

def test_expand_valid_text(abbr_service):
    """测试有效文本输入"""
    text = "ROE是衡量公司盈利能力的重要指标"
    result = abbr_service.expand(
        text=text,
        options={},
        zhipu_options={}
    )
    
    assert isinstance(result, dict)
    assert "abbr" in result
    assert "expansion" in result
    assert "definition" in result

def test_simple_expansion(abbr_service):
    """测试简单展开功能"""
    text = "ROE"
    result = abbr_service._simple_expansion(text)
    
    assert isinstance(result, dict)
    assert "abbr" in result
    assert "expansion" in result
    assert "definition" in result

def test_context_aware_expansion(abbr_service):
    """测试上下文感知展开功能"""
    text = "ROE是衡量公司盈利能力的重要指标"
    result = abbr_service._context_aware_expansion(text)
    
    assert isinstance(result, dict)
    assert "abbr" in result
    assert "expansion" in result
    assert "definition" in result
    assert "context" in result

@pytest.mark.asyncio
async def test_get_abbr_definition(abbr_service):
    """测试缩写定义获取功能"""
    abbr = "ROE"
    definition = await abbr_service.get_abbr_definition(abbr)
    
    assert definition is None or isinstance(definition, dict)
    if definition:
        assert "abbr" in definition
        assert "expansion" in definition
        assert "definition" in definition

def test_validate_abbr(abbr_service):
    """测试缩写验证功能"""
    abbr = "ROE"
    result = abbr_service._validate_abbr(abbr)
    
    assert isinstance(result, bool)

def test_expand_with_invalid_method(abbr_service):
    """测试无效展开方法"""
    with pytest.raises(ValueError):
        abbr_service.expand(
            text="ROE",
            options={"method": "invalid_method"},
            zhipu_options={}
        ) 
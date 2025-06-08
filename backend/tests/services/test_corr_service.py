import pytest
from services.corr_service import FinancialCorrService
from utils.error_handler import ValidationError, ModelError

@pytest.fixture
def corr_service():
    """创建纠错服务实例的fixture"""
    return FinancialCorrService()

@pytest.mark.asyncio
async def test_correct_empty_text(corr_service):
    """测试空文本输入"""
    with pytest.raises(ValidationError):
        await corr_service.correct(
            text="",
            options={},
            term_types={"allFinancialTerms": True},
            zhipu_options={}
        )

@pytest.mark.asyncio
async def test_correct_valid_text(corr_service):
    """测试有效文本输入"""
    text = "ROE是衡量公司盈利能力的重药指标"
    result = await corr_service.correct(
        text=text,
        options={},
        term_types={"allFinancialTerms": True},
        zhipu_options={}
    )
    assert "corrected" in result
    assert "original" in result
    assert "corrections" in result

@pytest.mark.asyncio
async def test_simple_correction(corr_service):
    """测试简单纠错功能"""
    text = "ROE是重药指标"
    result = await corr_service._simple_correction(text)
    assert isinstance(result, dict)
    assert "original" in result
    assert "corrected_text" in result
    assert "corrections" in result

@pytest.mark.asyncio
async def test_context_aware_correction(corr_service):
    """测试上下文感知纠错功能"""
    text = "ROE是衡量公司盈利能力的重药指标"
    result = await corr_service._context_aware_correction(text)
    assert isinstance(result, dict)
    assert "original" in result
    assert "corrected_text" in result
    assert "corrections" in result

@pytest.mark.asyncio
async def test_validate_term(corr_service):
    """测试术语验证功能"""
    term = "ROE"
    result = await corr_service.validate_term(term)
    
    assert isinstance(result, dict)
    assert "is_valid" in result
    assert "suggestions" in result
    assert "confidence" in result

@pytest.mark.asyncio
async def test_add_mistakes(corr_service):
    """测试添加错误功能"""
    text = "ROE是衡量公司盈利能力的重要指标"
    error_options = {
        "error_types": ["spelling", "format"],
        "error_rate": 0.1
    }
    result = await corr_service.add_mistakes(text, error_options)
    assert isinstance(result, dict)
    assert "original_text" in result
    assert "modified_text" in result
    assert "method" in result

@pytest.mark.asyncio
async def test_correct_with_invalid_method(corr_service):
    """测试无效纠错方法"""
    with pytest.raises(ValidationError):
        await corr_service.correct(
            text="ROE是重要指标",
            options={"method": "invalid_method"},
            term_types={"allFinancialTerms": True},
            zhipu_options={}
        ) 
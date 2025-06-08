import pytest
from services.std_service import FinancialStdService
from utils.error_handler import ValidationError, ModelError

@pytest.fixture
def std_service():
    """创建标准化服务实例的fixture"""
    return FinancialStdService()

@pytest.mark.asyncio
async def test_standardize_empty_text(std_service):
    """测试空文本输入"""
    with pytest.raises(ValidationError):
        await std_service.standardize(
            text="",
            options={},
            term_types={"allFinancialTerms": True},
            zhipu_options={}
        )

@pytest.mark.asyncio
async def test_standardize_valid_text(std_service):
    """测试有效文本输入"""
    text = "ROE是衡量公司盈利能力的重要指标"
    result = await std_service.standardize(
        text=text,
        options={},
        term_types={"allFinancialTerms": True},
        zhipu_options={}
    )
    
    assert "standardized_terms" in result
    assert isinstance(result["standardized_terms"], list)

@pytest.mark.asyncio
async def test_standardize_terms(std_service):
    """测试术语标准化功能"""
    text = "ROE是衡量公司盈利能力的重要指标"
    terms = await std_service._standardize_terms(text)
    
    assert isinstance(terms, list)
    for term in terms:
        assert "original" in term
        assert "standardized" in term
        assert "type" in term
        assert "confidence" in term

def test_filter_terms(std_service):
    """测试术语过滤功能"""
    terms = [
        {"original": "ROE", "standardized": "净资产收益率", "type": "FINANCIAL_TERM", "confidence": 0.9},
        {"original": "EPS", "standardized": "每股收益", "type": "FINANCIAL_TERM", "confidence": 0.9}
    ]
    term_types = {"FINANCIAL_TERM": True}
    
    filtered = std_service._filter_terms(terms, term_types)
    assert isinstance(filtered, list)
    assert all(term["type"] == "FINANCIAL_TERM" for term in filtered)

@pytest.mark.asyncio
async def test_search_similar_terms(std_service):
    """测试相似术语搜索功能"""
    term = "ROE"
    similar_terms = await std_service.search_similar_terms(
        term=term,
        top_k=5,
        similarity_threshold=0.7
    )
    
    assert isinstance(similar_terms, list)
    assert len(similar_terms) <= 5
    for similar_term in similar_terms:
        assert "term" in similar_term
        assert "similarity" in similar_term
        assert "type" in similar_term
        assert "definition" in similar_term
        assert isinstance(similar_term["similarity"], float)
        assert 0 <= similar_term["similarity"] <= 1

@pytest.mark.asyncio
async def test_search_similar_terms_empty_input(std_service):
    """测试空输入搜索"""
    with pytest.raises(ValidationError):
        await std_service.search_similar_terms("")

@pytest.mark.asyncio
async def test_search_similar_terms_low_threshold(std_service):
    """测试低相似度阈值"""
    term = "ROE"
    similar_terms = await std_service.search_similar_terms(
        term=term,
        top_k=5,
        similarity_threshold=0.1  # 使用很低的阈值，应该返回更多结果
    )
    
    assert isinstance(similar_terms, list)
    assert len(similar_terms) <= 5 
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
        assert "metadata" in similar_term

def test_load_term_vectors(std_service):
    """测试术语向量加载功能"""
    std_service._load_term_vectors()
    assert hasattr(std_service, "term_vectors")
    assert hasattr(std_service, "term_metadata")
    assert isinstance(std_service.term_vectors, dict)
    assert isinstance(std_service.term_metadata, dict) 
import pytest
from services.gen_service import FinancialGenService
from utils.error_handler import ValidationError, ModelError

@pytest.fixture
def gen_service():
    """创建生成服务实例的fixture"""
    return FinancialGenService()

@pytest.mark.asyncio
async def test_generate_empty_prompt(gen_service):
    """测试空提示词输入"""
    with pytest.raises(ValidationError):
        await gen_service.generate(
            text="",
            context="",
            method="simple_generation",
            zhipu_options={}
        )

@pytest.mark.asyncio
async def test_generate_valid_prompt(gen_service):
    """测试有效提示词输入"""
    text = "请解释什么是ROE"
    result = await gen_service.generate(
        text=text,
        context="",
        method="simple_generation",
        zhipu_options={}
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)

@pytest.mark.asyncio
async def test_generate_with_template(gen_service):
    """测试使用模板生成"""
    template = "请解释{term}的含义"
    variables = {"term": "ROE"}
    result = await gen_service.generate_with_template(
        template=template,
        variables=variables,
        options={},
        zhipu_options={}
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)

@pytest.mark.asyncio
async def test_generate_with_context(gen_service):
    """测试带上下文的生成"""
    prompt = "请解释什么是ROE"
    context = "在金融分析中，ROE是一个重要指标"
    result = await gen_service.generate_with_context(
        prompt=prompt,
        context=context,
        options={},
        zhipu_options={}
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)

@pytest.mark.asyncio
async def test_generate_with_constraints(gen_service):
    """测试带约束的生成"""
    prompt = "请解释什么是ROE"
    constraints = {
        "max_length": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }
    result = await gen_service.generate_with_constraints(
        prompt=prompt,
        constraints=constraints,
        options={},
        zhipu_options={}
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)
    assert len(result["generated_text"]) <= 100

@pytest.mark.asyncio
async def test_generate_with_invalid_options(gen_service):
    """测试无效选项"""
    with pytest.raises(ValidationError):
        await gen_service.generate(
            text="请解释什么是ROE",
            context="",
            method="simple_generation",
            zhipu_options={"invalid_option": True}
        )

@pytest.mark.asyncio
async def test_generate_with_invalid_template(gen_service):
    """测试无效模板"""
    with pytest.raises(ValidationError):
        await gen_service.generate_with_template(
            template="无效模板{invalid_var}",
            variables={},
            options={},
            zhipu_options={}
        ) 
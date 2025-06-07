from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from services.ner_service import FinancialNERService
from services.std_service import FinancialStdService
from services.abbr_service import FinancialAbbrService
from services.corr_service import FinancialCorrService
from services.gen_service import FinancialGenService
from typing import List, Dict, Optional, Literal, Union, Any
from utils.server_config import ServerConfig, CORSConfig
from utils.logging_config import LoggingConfig
from utils.db_config import DBConfig
from utils.db_manager import DatabaseManager
from utils.error_handler import (
    APIError,
    ValidationError,
    ModelError,
    DatabaseError,
    error_handler
)
from dotenv import load_dotenv
import uvicorn

# 加载环境变量
load_dotenv()

# 初始化配置
server_config = ServerConfig()
cors_config = CORSConfig()
logging_config = LoggingConfig()
db_config = DBConfig()
db_manager = DatabaseManager(db_config.db_path)

# 配置日志
logging_config.configure()
logger = logging_config.logger

# 创建 FastAPI 应用
app = FastAPI(
    title="金融文本处理服务",
    description="提供金融文本的实体识别、标准化、缩写扩展、文本纠正和内容生成功能",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 注册错误处理中间件
app.add_exception_handler(Exception, error_handler)

# 配置跨域资源共享
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.origins,
    allow_credentials=True,
    allow_methods=cors_config.methods,
    allow_headers=cors_config.headers,
)

# 初始化各个服务
logger.info("正在初始化服务...")
ner_service = FinancialNERService()  # 金融实体识别服务
standardization_service = FinancialStdService()  # 金融术语标准化服务
abbr_service = FinancialAbbrService()  # 金融缩写扩展服务
gen_service = FinancialGenService()  # 金融文本生成服务
corr_service = FinancialCorrService()  # 金融文本纠正服务
logger.info("所有服务初始化完成")

# 基础模型类
class BaseInputModel(BaseModel):
    """基础输入模型，包含所有模型共享的字段"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    llmOptions: Dict[str, str] = Field(
        default_factory=lambda: {
            "provider": "zhipu",
            "model": "glm-4-plus"
        },
        description="大语言模型配置选项"
    )

class ZhipuOptions(BaseModel):
    """智谱AI配置选项"""
    model_type: Literal["embedding", "llm"] = Field(
        default="embedding",
        description="模型类型"
    )
    model_name: str = Field(
        default="glm-4-plus",
        description="模型名称"
    )
    db_name: str = Field(
        default="financial_terms_zhipu",
        description="向量数据库名称"
    )
    collection_name: str = Field(
        default="financial_terms",
        description="集合名称"
    )

class TextInput(BaseInputModel):
    """文本输入模型，用于标准化和实体识别"""
    text: str = Field(..., description="输入文本")
    options: Dict[str, bool] = Field(
        default_factory=dict,
        description="处理选项"
    )
    term_types: Dict[str, bool] = Field(
        default_factory=dict,
        description="金融术语类型"
    )
    zhipu_options: ZhipuOptions = Field(
        default_factory=ZhipuOptions,
        description="智谱AI配置选项"
    )

class AbbrInput(BaseInputModel):
    """金融缩写扩展输入模型"""
    text: str = Field(..., description="输入文本")
    context: str = Field(
        default="",
        description="上下文信息"
    )
    method: Literal["simple_expansion", "context_aware_expansion"] = Field(
        default="simple_expansion",
        description="处理方法"
    )
    zhipu_options: Optional[ZhipuOptions] = Field(
        default_factory=ZhipuOptions,
        description="智谱AI配置选项"
    )

class ErrorOptions(BaseModel):
    """错误生成选项"""
    probability: float = Field(
        default=0.3,
        description="错误生成概率",
        ge=0.0,
        le=1.0
    )
    max_errors: int = Field(
        default=5,
        description="最大错误数量",
        ge=1
    )

class CorrInput(BaseInputModel):
    """金融文本纠正输入模型"""
    text: str = Field(..., description="输入文本")
    method: Literal["correct_spelling", "add_mistakes"] = Field(
        default="correct_spelling",
        description="处理方法"
    )
    error_options: ErrorOptions = Field(
        default_factory=ErrorOptions,
        description="错误生成选项"
    )

class CompanyInfo(BaseModel):
    """公司信息模型"""
    name: str = Field(..., description="公司名称")
    industry: str = Field(..., description="所属行业")
    market_cap: Optional[float] = Field(
        None,
        description="市值（亿元）"
    )
    financial_status: Optional[str] = Field(
        None,
        description="财务状况"
    )

class GenInput(BaseInputModel):
    """金融内容生成输入模型"""
    company_info: CompanyInfo = Field(..., description="公司信息")
    financial_metrics: List[Dict[str, Any]] = Field(..., description="财务指标")
    analysis_type: str = Field(
        default="",
        description="分析类型"
    )
    investment_strategy: str = Field(
        default="",
        description="投资策略"
    )
    method: Literal["generate_financial_report", "generate_financial_analysis", "generate_investment_plan"] = Field(
        default="generate_financial_report",
        description="生成方法"
    )

# 统一响应格式
def standard_response(
    data: Any,
    message: str = "success",
    status_code: int = 200
) -> Dict[str, Any]:
    """生成标准API响应
    
    Args:
        data: 响应数据
        message: 响应消息
        status_code: 状态码
        
    Returns:
        Dict[str, Any]: 标准响应格式
    """
    return {
        "status": "success",
        "code": status_code,
        "message": message,
        "data": data
    }

@app.get("/")
async def root():
    """API根路径
    
    Returns:
        Dict[str, Any]: API信息
    """
    return standard_response({
        "message": "欢迎使用金融文本处理服务",
        "version": "1.0.0",
        "services": [
            "术语识别",
            "术语标准化",
            "缩写扩展",
            "文本纠正",
            "内容生成"
        ]
    })

@app.post("/api/std", response_model=Dict[str, Any])
async def standardization(input: TextInput):
    """金融术语标准化
    
    将输入的金融术语标准化为标准形式。
    
    Args:
        input: 文本输入模型
        
    Returns:
        Dict[str, Any]: 标准化结果
        
    Raises:
        ValidationError: 当输入参数无效时
        ModelError: 当模型处理失败时
    """
    try:
        logger.info("开始处理术语标准化请求")
        result = await standardization_service.standardize(
            input.text,
            input.options,
            input.term_types,
            input.zhipu_options.dict()
        )
        logger.info("术语标准化请求处理完成")
        return standard_response(result)
    except Exception as e:
        logger.error(f"术语标准化请求处理失败: {str(e)}")
        raise

@app.post("/api/ner", response_model=Dict[str, Any])
async def ner(input: TextInput):
    """金融实体识别
    
    识别文本中的金融实体。
    
    Args:
        input: 文本输入模型
        
    Returns:
        Dict[str, Any]: 识别结果
        
    Raises:
        ValidationError: 当输入参数无效时
        ModelError: 当模型处理失败时
    """
    try:
        logger.info("开始处理实体识别请求")
        result = await ner_service.recognize(
            input.text,
            input.options,
            input.term_types,
            input.zhipu_options.dict()
        )
        logger.info("实体识别请求处理完成")
        return standard_response(result)
    except Exception as e:
        logger.error(f"实体识别请求处理失败: {str(e)}")
        raise

@app.post("/api/corr", response_model=Dict[str, Any])
async def correct_text(input: CorrInput):
    """金融文本纠正
    
    纠正文本中的错误或添加错误。
    
    Args:
        input: 文本纠正输入模型
        
    Returns:
        Dict[str, Any]: 纠正结果
        
    Raises:
        ValidationError: 当输入参数无效时
        ModelError: 当模型处理失败时
    """
    try:
        logger.info("开始处理文本纠正请求")
        result = await corr_service.correct(
            input.text,
            input.method,
            input.error_options.dict(),
            input.llmOptions
        )
        logger.info("文本纠正请求处理完成")
        return standard_response(result)
    except Exception as e:
        logger.error(f"文本纠正请求处理失败: {str(e)}")
        raise

@app.post("/api/abbr", response_model=Dict[str, Any])
async def expand_abbreviations(input: AbbrInput):
    """金融缩写扩展
    
    扩展文本中的金融缩写。
    
    Args:
        input: 缩写扩展输入模型
        
    Returns:
        Dict[str, Any]: 扩展结果
        
    Raises:
        ValidationError: 当输入参数无效时
        ModelError: 当模型处理失败时
    """
    try:
        logger.info("开始处理缩写扩展请求")
        result = await abbr_service.expand(
            input.text,
            input.context,
            input.method,
            input.zhipu_options.dict()
        )
        logger.info("缩写扩展请求处理完成")
        return standard_response(result)
    except Exception as e:
        logger.error(f"缩写扩展请求处理失败: {str(e)}")
        raise

@app.post("/api/gen", response_model=Dict[str, Any])
async def generate_financial_content(input: GenInput):
    """金融内容生成
    
    生成金融相关的内容。
    
    Args:
        input: 内容生成输入模型
        
    Returns:
        Dict[str, Any]: 生成结果
        
    Raises:
        ValidationError: 当输入参数无效时
        ModelError: 当模型处理失败时
    """
    try:
        logger.info("开始处理内容生成请求")
        result = await gen_service.generate(
            input.company_info.dict(),
            input.financial_metrics,
            input.analysis_type,
            input.investment_strategy,
            input.method,
            input.llmOptions
        )
        logger.info("内容生成请求处理完成")
        return standard_response(result)
    except Exception as e:
        logger.error(f"内容生成请求处理失败: {str(e)}")
        raise

@app.post("/api/database", response_model=Dict[str, Any])
async def handle_database(request: Dict[str, Any]):
    """数据库操作
    
    执行数据库操作。
    
    Args:
        request: 数据库请求
        
    Returns:
        Dict[str, Any]: 操作结果
        
    Raises:
        ValidationError: 当输入参数无效时
        DatabaseError: 当数据库操作失败时
    """
    try:
        logger.info(f"开始处理数据库操作: {request['operation']}")
        with db_manager.transaction() as conn:
            cursor = conn.cursor()
            
            if request['operation'] == "query":
                # 查询操作
                cursor.execute(request['data'].get("query", ""))
                result = cursor.fetchall()
            elif request['operation'] == "insert":
                # 插入操作
                cursor.execute(
                    request['data'].get("query", ""),
                    request['data'].get("params", [])
                )
                result = {"affected_rows": cursor.rowcount}
            elif request['operation'] == "update":
                # 更新操作
                cursor.execute(
                    request['data'].get("query", ""),
                    request['data'].get("params", [])
                )
                result = {"affected_rows": cursor.rowcount}
            elif request['operation'] == "delete":
                # 删除操作
                cursor.execute(
                    request['data'].get("query", ""),
                    request['data'].get("params", [])
                )
                result = {"affected_rows": cursor.rowcount}
            else:
                raise ValidationError(f"不支持的操作类型: {request['operation']}")
            
            logger.info(f"数据库操作完成: {request['operation']}")
            return standard_response(result)
    except Exception as e:
        logger.error(f"数据库操作失败: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

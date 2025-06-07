from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional
import traceback
from utils.logging_config import LoggingConfig

logger = LoggingConfig().logger

class APIError(Exception):
    """API错误基类"""
    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)

class ValidationError(APIError):
    """参数验证错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(400, message, details)

class ModelError(APIError):
    """模型处理错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(500, message, details)

class DatabaseError(APIError):
    """数据库操作错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(500, message, details)

def format_error_response(
    code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """格式化错误响应
    
    Args:
        code: 错误代码
        message: 错误消息
        details: 错误详情
        
    Returns:
        Dict[str, Any]: 格式化的错误响应
    """
    response = {
        "status": "error",
        "code": code,
        "message": message
    }
    if details:
        response["details"] = details
    return response

async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局错误处理中间件
    
    Args:
        request: 请求对象
        exc: 异常对象
        
    Returns:
        JSONResponse: 错误响应
    """
    # 记录错误日志
    logger.error(f"Error processing request: {request.url}")
    logger.error(f"Error details: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    if isinstance(exc, APIError):
        # 处理自定义API错误
        return JSONResponse(
            status_code=exc.code,
            content=format_error_response(
                exc.code,
                exc.message,
                exc.details
            )
        )
    elif isinstance(exc, HTTPException):
        # 处理FastAPI HTTP异常
        return JSONResponse(
            status_code=exc.status_code,
            content=format_error_response(
                exc.status_code,
                exc.detail
            )
        )
    else:
        # 处理其他未预期的错误
        return JSONResponse(
            status_code=500,
            content=format_error_response(
                500,
                "Internal server error",
                {"error": str(exc)}
            )
        ) 
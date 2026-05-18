"""Unified API response structure: all endpoints return HTTP 200 with {code, message, data}."""

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.error_codes import ERROR_MESSAGES, SUCCESS


class ApiResponse(BaseModel):
    code: int = SUCCESS
    message: str = "成功"
    data: Any = None

    @staticmethod
    def success(data: Any = None, message: str = "成功") -> dict:
        return {"code": SUCCESS, "message": message, "data": data}

    @staticmethod
    def error(code: int, message: str | None = None, data: Any = None) -> dict:
        msg = message or ERROR_MESSAGES.get(code, "未知错误")
        return {"code": code, "message": msg, "data": data}


def register_exception_handler(app: FastAPI) -> None:
    """Register a global exception handler that converts all errors to ApiResponse format."""
    from fastapi import HTTPException, Request
    import logging

    logger = logging.getLogger(__name__)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Convert HTTPException to ApiResponse with HTTP 200."""
        logger.warning(
            "HTTP exception on %s %s: status=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        # Map HTTP status codes to our error codes
        code_map = {
            400: None,  # will use the code from detail if it's a dict, else GENERAL_ERROR
            404: None,
            422: None,
            500: None,
        }
        if exc.status_code in code_map:
            # If detail is a dict with 'code', use it; otherwise map by status
            if isinstance(exc.detail, dict) and "code" in exc.detail:
                return JSONResponse(
                    status_code=200,
                    content=ApiResponse.error(
                        code=exc.detail["code"],
                        message=exc.detail.get("message"),
                        data=exc.detail.get("data"),
                    ),
                )
            # Default mapping
            from app.core.error_codes import GENERAL_ERROR, KB_NOT_FOUND, UNSUPPORTED_FILE_TYPE, INVALID_SEARCH_QUERY

            if exc.status_code == 404:
                return JSONResponse(
                    status_code=200,
                    content=ApiResponse.error(code=KB_NOT_FOUND, message=str(exc.detail)),
                )
            elif exc.status_code == 400:
                return JSONResponse(
                    status_code=200,
                    content=ApiResponse.error(code=GENERAL_ERROR, message=str(exc.detail)),
                )
            else:
                return JSONResponse(
                    status_code=200,
                    content=ApiResponse.error(code=GENERAL_ERROR, message=str(exc.detail)),
                )
        return JSONResponse(
            status_code=200,
            content=ApiResponse.error(code=GENERAL_ERROR, message=str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all for unhandled exceptions."""
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
        from app.core.error_codes import GENERAL_ERROR

        return JSONResponse(
            status_code=200,
            content=ApiResponse.error(code=GENERAL_ERROR, message=f"服务器内部错误: {str(exc)}"),
        )

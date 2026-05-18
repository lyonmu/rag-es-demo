from .chat import router as chat_router
from .knowledge_base import router as kb_router
from .search import router as search_router
from .upload import router as upload_router

__all__ = ["chat_router", "kb_router", "search_router", "upload_router"]

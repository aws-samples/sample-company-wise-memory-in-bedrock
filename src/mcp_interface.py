"""
MCP Interface Layer using fastmcp for agent orchestration.
"""
import os
import sys
from typing import List, Tuple

from fastmcp import FastMCP

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.memory_management import MemoryManagementError, MemoryManagementService  # noqa: E402
from src.utils.config import config  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

# Initialize FastMCP application
mcp = FastMCP('Graph Memory')
memory_service = MemoryManagementService()


@mcp.tool()
def search_graph_memories(user_id: str, query: str, top_k: int = 10) -> List[Tuple[str, str]]:
    """Search entity's graph memories.

    Args:
        user_id: User ID
        query: Natural language query
        top_k: Maximum number of results to return (default: 10)

    Returns:
        List of tuples (memory_id, statement)

    Raises:
        Exception: If search fails
    """

    try:
        if not user_id or not user_id.strip():
            raise ValueError('User ID is required')

        if not query or not query.strip():
            return []

        messages = [{'role': 'user', 'content': query}]
        memories = memory_service.search(user_id, messages, top_k)

        result = [(memory.id, memory.statement) for memory in memories]

        logger.debug(f'MCP search returned {len(result)} memories for user {user_id}')
        return result

    except MemoryManagementError as e:
        logger.error(f'Memory management error in MCP search: {e}')
        raise Exception(f'Memory search failed: {e}')
    except Exception as e:
        logger.error(f'Unexpected error in MCP search: {e}')
        raise Exception(f'Memory search failed: {e}')


if __name__ == '__main__':
    transport = config.mcp.transport
    host = config.mcp.host
    port = config.mcp.port
    mcp.run(transport=transport, host=host, port=port)

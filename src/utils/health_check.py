"""
Health check utilities for the application.
"""

from typing import Any, Dict

from .bedrock_embed import BedrockEmbed
from .bedrock_llm import BedrockLLM
from .bedrock_rerank import BedrockRerank
from .config import config
from .logging_config import get_logger
from .neptune_client import NeptuneClient
from .opensearch_client import OpenSearchClient

logger = get_logger(__name__)


def check_health() -> bool:
    """Check the health of all system components.

    Returns:
        True if all components are healthy, False otherwise
    """
    try:
        health_status = get_health_status()

        # Check if all components are healthy
        all_healthy = all(status.get('healthy', False) for status in health_status.values())

        if all_healthy:
            logger.info('All system components are healthy')
        else:
            logger.warning('Some system components are unhealthy')

        return all_healthy

    except Exception as e:
        logger.error(f'Health check failed: {e}')
        return False


def get_health_status() -> Dict[str, Any]:
    """Get detailed health status of all components.

    Returns:
        Dictionary with health status of each component
    """
    health_status = {}

    # Check Bedrock LLM
    try:
        llm = BedrockLLM(config.bedrock_llm)
        llm_healthy = llm.health_check()
        health_status['bedrock_llm'] = {
            'healthy': llm_healthy,
            'service': 'Amazon Bedrock LLM',
            'model': config.bedrock_llm.model_id
        }
    except Exception as e:
        health_status['bedrock_llm'] = {'healthy': False, 'service': 'Amazon Bedrock LLM', 'error': str(e)}

    # Check Bedrock Embed
    try:
        embed = BedrockEmbed(config.bedrock_embed)
        embed_healthy = embed.health_check()
        health_status['bedrock_embed'] = {
            'healthy': embed_healthy,
            'service': 'Amazon Bedrock Embed',
            'model': config.bedrock_embed.model_id
        }
    except Exception as e:
        health_status['bedrock_embed'] = {'healthy': False, 'service': 'Amazon Bedrock Embed', 'error': str(e)}

    # Check Bedrock Rerank
    try:
        rerank = BedrockRerank(config.bedrock_rerank)
        rerank_healthy = rerank.health_check()
        health_status['bedrock_rerank'] = {
            'healthy': rerank_healthy,
            'service': 'Amazon Bedrock Rerank',
            'model': config.bedrock_rerank.model_id
        }
    except Exception as e:
        health_status['bedrock_rerank'] = {'healthy': False, 'service': 'Amazon Bedrock Rerank', 'error': str(e)}

    # Check Neptune
    try:
        neptune = NeptuneClient(config.neptune)
        neptune_healthy = neptune.health_check()
        health_status['neptune'] = {
            'healthy': neptune_healthy,
            'service': 'Amazon Neptune',
            'endpoint': config.neptune.endpoint
        }
    except Exception as e:
        health_status['neptune'] = {'healthy': False, 'service': 'Amazon Neptune', 'error': str(e)}

    # Check OpenSearch
    try:
        opensearch = OpenSearchClient(config.opensearch)
        opensearch_healthy = opensearch.health_check()
        health_status['opensearch'] = {
            'healthy': opensearch_healthy,
            'service': 'Amazon OpenSearch',
            'endpoint': config.opensearch.endpoint
        }
    except Exception as e:
        health_status['opensearch'] = {'healthy': False, 'service': 'Amazon OpenSearch', 'error': str(e)}

    return health_status


def get_system_info() -> Dict[str, Any]:
    """Get system information and configuration.

    Returns:
        Dictionary with system information
    """
    return {
        'service_name': 'GraphMem',
        'version': '1.0.0',
        'configuration': {
            'bedrock_llm_model': config.bedrock_llm.model_id,
            'bedrock_embed_model': config.bedrock_embed.model_id,
            'memory_expiration_days': config.memory.default_expiration_days,
            'aws_region': config.bedrock_llm.region
        },
        'health_status': get_health_status()
    }

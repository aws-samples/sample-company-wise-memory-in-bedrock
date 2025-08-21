"""
Configuration management for AWS services and application settings.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class BedrockLLMConfig:
    """Configuration for Amazon Bedrock LLM service."""
    region: str
    model_id: str
    max_tokens: int
    temperature: float
    retry_attempts: int
    retry_delay: float


@dataclass
class BedrockEmbedConfig:
    """Configuration for Amazon Bedrock Embed service."""
    region: str
    model_id: str
    dimension: int
    retry_attempts: int
    retry_delay: float


@dataclass
class BedrockRerankConfig:
    """Configuration for Amazon Bedrock Embed service."""
    region: str
    model_id: str
    retry_attempts: int
    retry_delay: float


@dataclass
class NeptuneConfig:
    """Configuration for Amazon Neptune graph database."""
    endpoint: str
    port: int
    region: str


@dataclass
class OpenSearchConfig:
    """Configuration for OpenSearch."""
    endpoint: str
    port: int
    region: str
    index_name: str
    dimension: int


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    default_expiration_days: int
    cleanup_interval_hours: int


@dataclass
class MCPConfig:
    """Configuration for MCP interface."""
    transport: str
    host: str
    port: int


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: str
    log_level: str
    bedrock_llm: BedrockLLMConfig
    bedrock_embed: BedrockEmbedConfig
    bedrock_rerank: BedrockRerankConfig
    neptune: NeptuneConfig
    opensearch: OpenSearchConfig
    memory: MemoryConfig
    mcp: MCPConfig


def load_config() -> AppConfig:
    """Load configuration from environment variables with defaults."""
    environment = os.getenv('ENVIRONMENT', 'development')

    # Bedrock configuration
    bedrock_llm_config = BedrockLLMConfig(region=os.getenv('BEDROCK_LLM_AWS_REGION', 'us-east-1'),
                                          model_id=os.getenv('BEDROCK_LLM_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
                                          max_tokens=int(os.getenv('BEDROCK_LLM_MAX_TOKENS', '4096')),
                                          temperature=float(os.getenv('BEDROCK_LLM_TEMPERATURE', '0.0')),
                                          retry_attempts=int(os.getenv('BEDROCK_LLM_RETRY_ATTEMPTS', '3')),
                                          retry_delay=float(os.getenv('BEDROCK_LLM_RETRY_DELAY', '1.0')))

    # Bedrock Embed configuration
    bedrock_embed_config = BedrockEmbedConfig(region=os.getenv('BEDROCK_EMBED_AWS_REGION', 'us-east-1'),
                                              model_id=os.getenv('BEDROCK_EMBED_MODEL_ID', 'amazon.titan-embed-text-v2:0'),
                                              dimension=int(os.getenv('BEDROCK_EMBED_DIMENSION', '1024')),
                                              retry_attempts=int(os.getenv('BEDROCK_EMBED_RETRY_ATTEMPTS', '3')),
                                              retry_delay=float(os.getenv('BEDROCK_EMBED_RETRY_DELAY', '1.0')))

    # Bedrock Rerank configuration
    bedrock_rerank_config = BedrockRerankConfig(region=os.getenv('BEDROCK_RERANK_AWS_REGION', 'us-west-2'),
                                                model_id=os.getenv('BEDROCK_RERANK_MODEL_ID', 'amazon.rerank-v1:0'),
                                                retry_attempts=int(os.getenv('BEDROCK_RERANK_RETRY_ATTEMPTS', '3')),
                                                retry_delay=float(os.getenv('BEDROCK_RERANK_RETRY_DELAY', '1.0')))

    # Neptune configuration
    neptune_config = NeptuneConfig(endpoint=os.getenv('NEPTUNE_ENDPOINT', 'localhost'),
                                   port=int(os.getenv('NEPTUNE_PORT', '8182')),
                                   region=os.getenv('NEPTUNE_AWS_REGION', 'us-east-1'))

    # Vector search configuration
    opensearch_config = OpenSearchConfig(endpoint=os.getenv('OPENSEARCH_ENDPOINT', 'localhost'),
                                         port=os.getenv('OPENSEARCH_PORT', '443'),
                                         region=os.getenv('OPENSEARCH_AWS_REGION', 'us-east-1'),
                                         index_name=os.getenv('OPENSEARCH_INDEX', 'memory_embeddings'),
                                         dimension=int(os.getenv('OPENSEARCH_DIMENSION', '1024')))

    # Memory configuration
    memory_config = MemoryConfig(default_expiration_days=int(os.getenv('MEMORY_DEFAULT_EXPIRATION_DAYS', '90')),
                                 cleanup_interval_hours=int(os.getenv('MEMORY_CLEANUP_INTERVAL_HOURS', '24')))

    # MCP configuration
    mcp_config = MCPConfig(transport=os.getenv('MCP_TRANSPORT', 'sse'),
                           host=os.getenv('MCP_HOST', '127.0.0.1'),
                           port=int(os.getenv('MCP_PORT', '8000')))

    return AppConfig(environment=environment,
                     log_level=os.getenv('LOG_LEVEL', 'INFO'),
                     bedrock_llm=bedrock_llm_config,
                     bedrock_rerank=bedrock_rerank_config,
                     bedrock_embed=bedrock_embed_config,
                     neptune=neptune_config,
                     opensearch=opensearch_config,
                     memory=memory_config,
                     mcp=mcp_config)


# Global configuration instance
config = load_config()

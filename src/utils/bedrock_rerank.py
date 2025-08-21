"""
Amazon Bedrock Rerank client for reordering search results.
"""

import json
import random
import time
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import BedrockRerankConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class BedrockRerankError(Exception):
    """Custom exception for Bedrock Rerank errors."""
    pass


class BedrockRerank:
    """Amazon Bedrock Rerank client with error handling and retry logic."""

    def __init__(self, config: BedrockRerankConfig):
        """
        Initialize Bedrock Rerank client.

        Args:
            config: BedrockRerankConfig instance with connection parameters
        """
        self.config = config
        self.model_id = config.model_id
        assert config.region in ['us-west-2', 'ap-northeast-1', 'ca-central-1', 'eu-central-1']

        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=config.region,
        )
        logger.info(f'Initialized Bedrock Rerank client in region: {config.region}, with model: {config.model_id}')

    def rerank(self,
               query: str,
               documents: List[Union[str, Dict[str, Any]]],
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query string
            documents: List of documents (strings or dictionaries)
            top_k: Number of top results to return (default: all documents)

        Returns:
            List of reranked results with scores and indices

        Raises:
            BedrockRerankError: If reranking fails
        """
        if not query or not query.strip():
            logger.warning('Empty query provided for reranking')
            return []

        if not documents:
            logger.warning('No documents provided for reranking')
            return []

        # Default to returning all documents if top_k not specified
        if top_k is None:
            top_k = len(documents)
        # Convert documents to strings for reranking
        serialized_documents = [json.dumps(doc) if isinstance(doc, dict) else str(doc) for doc in documents]

        # Construct the request body
        data = {'query': query.strip(), 'documents': serialized_documents, 'top_n': min(top_k, len(documents))}
        if 'cohere' in self.model_id.lower():
            data['api_version'] = 2
        body = json.dumps(data)

        logger.debug(f'Reranking {len(documents)} documents for query: {query[:50]}...')

        for attempt in range(self.config.retry_attempts):
            try:
                # Invoke the Bedrock rerank model
                response = self.bedrock_runtime.invoke_model(modelId=self.model_id,
                                                             accept='application/json',
                                                             contentType='application/json',
                                                             body=body)

                # Process the response
                response_body = json.loads(response.get('body').read())

                if 'results' not in response_body:
                    logger.error('Invalid response format from Bedrock rerank')
                    raise BedrockRerankError('Invalid response format')

                results = response_body['results']
                logger.debug(f'Reranking returned {len(results)} results')
                results = [{'document': documents[res['index']], **res} for res in results]
                return results

            except (ClientError, BotoCoreError, json.JSONDecodeError) as e:
                logger.warning(f'Bedrock Rerank attempt {attempt + 1}/{self.config.retry_attempts} failed: {e}')

                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff with jitter
                    delay = self.config.retry_delay * (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise BedrockRerankError(f'Bedrock Rerank failed after {self.config.retry_attempts} attempts: {e}')
            except Exception as e:
                logger.error(f'Unexpected error during reranking: {e}')
                raise BedrockRerankError(f'Unexpected reranking error: {e}')
        raise BedrockRerankError(f'Bedrock Rerank failed after {self.config.retry_attempts} attempts')

    def health_check(self) -> bool:
        """
        Perform a health check on the Bedrock Rerank service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Simple test with minimal documents
            test_documents = ['Test document 1', 'Test document 2']
            results = self.rerank('test query', test_documents, top_k=1)
            return len(results) > 0

        except Exception as e:
            logger.error(f'Bedrock Rerank health check failed: {e}')
            return False

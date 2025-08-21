"""
Amazon Bedrock embedding client wrapper with retry logic and error handling.
"""

import json
import random
import time
from typing import List

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import BedrockEmbedConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class BedrockEmbedError(Exception):
    """Custom exception for Bedrock embedding errors."""
    pass


class BedrockEmbed:
    """Amazon Bedrock embedding client with retry logic and error handling."""

    def __init__(self, config: BedrockEmbedConfig):
        """
        Initialize Bedrock embedding client.

        Args:
            config: BedrockEmbedConfig instance with connection parameters
        """
        self.config = config
        self.model_id = config.model_id
        self.output_embedding_length = config.dimension

        # Create Bedrock runtime client
        self.bedrock = boto3.client(service_name='bedrock-runtime', region_name=config.region)

        logger.info(f'Initialized Bedrock Embed client with model: {self.model_id}')

    def _call_with_retry(self, data: dict) -> dict:
        """
        Make a Bedrock API call with retry logic.

        Args:
            data: Request data dictionary

        Returns:
            Response dictionary from Bedrock API

        Raises:
            BedrockEmbedError: If all retry attempts fail
        """
        body = json.dumps(data)

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f'Bedrock Embed request attempt {attempt + 1}/{self.config.retry_attempts}')

                accept = 'application/json'
                content_type = 'application/json'
                response = self.bedrock.invoke_model(body=body, modelId=self.model_id, accept=accept, contentType=content_type)

                result = json.loads(response.get('body').read())
                logger.debug('Bedrock Embed request successful')
                return result

            except (ClientError, BotoCoreError) as e:
                logger.warning(f'Bedrock Embed attempt {attempt + 1}/{self.config.retry_attempts} failed: {e}')

                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff with jitter
                    delay = self.config.retry_delay * (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise BedrockEmbedError(f'Bedrock Embed failed after {self.config.retry_attempts} attempts: {e}')

            except Exception as e:
                logger.error(f'Unexpected error in Bedrock Embed: {e}')
                raise BedrockEmbedError(f'Unexpected Bedrock Embed error: {e}')

        raise BedrockEmbedError(f'Bedrock Embed failed after {self.config.retry_attempts} attempts')

    def embed_document(self, text: str) -> List[float]:
        """
        Generate embeddings for document text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values

        Raises:
            BedrockEmbedError: If embedding generation fails
        """
        if not text or not text.strip():
            logger.warning('Empty text provided for document embedding')
            return [0.0] * self.output_embedding_length

        try:
            if 'titan' in self.model_id.lower():
                data = {'inputText': text, 'dimensions': self.output_embedding_length}
                response = self._call_with_retry(data)
                return response.get('embedding', [0.0] * self.output_embedding_length)

            elif 'cohere' in self.model_id.lower():
                if self.output_embedding_length != 1024:
                    raise BedrockEmbedError(f'Cohere models only support 1024 dimensions, got {self.output_embedding_length}')

                data = {'input_type': 'search_document', 'texts': [text]}
                response = self._call_with_retry(data)
                embeddings = response.get('embeddings', [[0.0] * self.output_embedding_length])
                return embeddings[0] if embeddings else [0.0] * self.output_embedding_length

            else:
                raise BedrockEmbedError(f'Unsupported model for document embedding: {self.model_id}')

        except BedrockEmbedError:
            raise
        except Exception as e:
            logger.error(f'Error generating document embedding: {e}')
            raise BedrockEmbedError(f'Document embedding failed: {e}')

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for query text.

        Args:
            text: Query text to embed

        Returns:
            List of embedding values

        Raises:
            BedrockEmbedError: If embedding generation fails
        """
        if not text or not text.strip():
            logger.warning('Empty text provided for query embedding')
            return [0.0] * self.output_embedding_length

        try:
            if 'titan' in self.model_id.lower():
                data = {'inputText': text, 'dimensions': self.output_embedding_length}
                response = self._call_with_retry(data)
                return response.get('embedding', [0.0] * self.output_embedding_length)

            elif 'cohere' in self.model_id.lower():
                if self.output_embedding_length != 1024:
                    raise BedrockEmbedError(f'Cohere models only support 1024 dimensions, got {self.output_embedding_length}')

                data = {'input_type': 'search_query', 'texts': [text]}
                response = self._call_with_retry(data)
                embeddings = response.get('embeddings', [[0.0] * self.output_embedding_length])
                return embeddings[0] if embeddings else [0.0] * self.output_embedding_length

            else:
                raise BedrockEmbedError(f'Unsupported model for query embedding: {self.model_id}')

        except BedrockEmbedError:
            raise
        except Exception as e:
            logger.error(f'Error generating query embedding: {e}')
            raise BedrockEmbedError(f'Query embedding failed: {e}')

    def health_check(self) -> bool:
        """
        Perform a health check on the Bedrock embedding service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            test_embedding = self.embed_document('test')
            return len(test_embedding) == self.output_embedding_length

        except Exception as e:
            logger.error(f'Bedrock Embed health check failed: {e}')
            return False

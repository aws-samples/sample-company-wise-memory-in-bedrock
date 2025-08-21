"""
Amazon Bedrock LLM client wrapper with retry logic and error handling.
"""

import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from .config import BedrockLLMConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class BedrockLLMError(Exception):
    """Custom exception for Bedrock LLM errors."""
    pass


class BedrockLLM:
    """Amazon Bedrock LLM client with retry logic and error handling."""

    def __init__(self, config: BedrockLLMConfig):
        """
        Initialize Bedrock LLM client.

        Args:
            config: BedrockLLMConfig instance with connection parameters
        """
        self.config = config
        self.model_id = config.model_id

        # Create Bedrock runtime client with timeout configuration
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=config.region,
            config=BotoConfig(
                connect_timeout=600,
                read_timeout=600,
                retries={'max_attempts': 0}  # We handle retries manually
            ))

        logger.info(f'Initialized Bedrock LLM client with model: {self.model_id}')

    def generate_response(self,
                          messages: List[Dict[str, Any]],
                          system_prompt: str,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          stop_sequences: Optional[List[str]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response using Bedrock LLM with retry logic.

        Args:
            messages: List of message dictionaries in Bedrock format
            system_prompt: System prompt for the conversation
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Temperature for generation (uses config default if None)
            stop_sequences: Stop sequences for generation

        Returns:
            Tuple of (response_text, invoke_metrics)

        Raises:
            BedrockLLMError: If all retry attempts fail
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        stop_sequences = stop_sequences or []

        system = [{'text': system_prompt}]
        inf_params = {
            'maxTokens': max_tokens,
            'temperature': temperature,
            'stopSequences': stop_sequences,
        }

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f'Bedrock LLM request attempt {attempt + 1}/{self.config.retry_attempts}')

                stream = self.bedrock_runtime.converse_stream(modelId=self.model_id,
                                                              messages=messages,
                                                              system=system,
                                                              inferenceConfig=inf_params).get('stream')

                msg = ''
                invoke_metrics = None

                if stream:
                    for event in stream:
                        if 'contentBlockDelta' in event:
                            msg += event['contentBlockDelta']['delta']['text']
                        if 'metadata' in event:
                            invoke_metrics = {**event['metadata']['usage'], **event['metadata']['metrics']}

                logger.debug(f'Bedrock LLM response generated successfully (length: {len(msg)})')
                return msg, invoke_metrics

            except (ClientError, BotoCoreError, json.JSONDecodeError) as e:
                logger.warning(f'Bedrock LLM attempt {attempt + 1}/{self.config.retry_attempts} failed: {e}')

                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff with jitter
                    delay = self.config.retry_delay * (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise BedrockLLMError(f'Bedrock LLM failed after {self.config.retry_attempts} attempts: {e}')

            except Exception as e:
                logger.error(f'Unexpected error in Bedrock LLM: {e}')
                raise BedrockLLMError(f'Unexpected Bedrock LLM error: {e}')

        raise BedrockLLMError(f'Bedrock LLM failed after {self.config.retry_attempts} attempts')

    def health_check(self) -> bool:
        """
        Perform a health check on the Bedrock LLM service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            test_messages = [{'role': 'user', 'content': [{'text': 'Hi'}]}]
            response, _ = self.generate_response(messages=test_messages,
                                                 system_prompt="You are a helpful assistant. Respond with just 'OK'.",
                                                 max_tokens=10,
                                                 temperature=0.0)
            return len(response.strip()) > 0

        except Exception as e:
            logger.error(f'Bedrock LLM health check failed: {e}')
            return False

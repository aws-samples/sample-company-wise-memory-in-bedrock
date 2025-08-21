"""
Response Memory Mapping Generation Service for mapping response sentences to source memories.
"""

import json
import re
from typing import Dict, List, Tuple

from ..utils.bedrock_llm import BedrockLLM
from ..utils.config import config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ResponseMemoryMappingService:
    """Generate response-memory mappings from AI response and retrieved memories."""

    def __init__(self):
        """Initialize the response-memory mapping generation service."""
        self.llm = BedrockLLM(config.bedrock_llm)
        logger.info('Initialized ResponseMemoryMappingService')

    def generate_response_memory_mapping(self, response: str, memories: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Map sentences in response to source memories using LLM.

        Args:
            response: AI response text to analyze
            memories: List of (memory_id, statement) tuples that were used to generate the response

        Returns:
            Dictionary mapping response sentences to list of source memory IDs
        """
        if not response or not response.strip():
            logger.warning('Empty response provided for mapping')
            return {}

        if not memories:
            logger.debug('No memories provided for mapping')
            return {}

        try:
            sentences = self._split_into_sentences(response)
            if not sentences:
                return {}

            mapping = self._generate_mapping_with_llm(sentences, memories)
            logger.debug(f'Generated mapping for {len(mapping)} sentences')
            return mapping

        except Exception as e:
            logger.error(f'Error generating response memory mapping: {e}')
            return {}

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences supporting multiple languages."""
        # Support English, Chinese, Japanese, and other language sentence endings
        # Split on sentence endings, with or without following whitespace
        sentences = re.split(r'[.!?。！？]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_mapping_with_llm(self, sentences: List[str], memories: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Use LLM to generate sentence-to-memory mapping."""
        sentences_text = '\n'.join([f'{i}: {s}' for i, s in enumerate(sentences)])
        memories_text = '\n'.join([f'{i}: {statement}' for i, (_, statement) in enumerate(memories)])

        system_prompt = """You are an expert at analyzing AI responses and tracing information back to source memories.

## Task
Analyze each sentence in the AI response and identify which memories (if any) directly support or provide evidence for the claims made in that sentence.

## Guidelines
1. **Direct Support**: Only map sentences to memories that directly support the factual claims
2. **Semantic Matching**: Consider semantic similarity, not just keyword matching
3. **Multiple Sources**: A sentence can be supported by multiple memories
4. **No Support**: If a sentence has no supporting memory, omit it from the output
5. **Partial Support**: Include memories that support part of a compound sentence

## Examples
- Sentence: "John works at Google as a software engineer"
- Memory: "John Smith is employed at Google" → MATCH (supports employment)
- Memory: "John's role is software engineer" → MATCH (supports job title)
- Memory: "Google is a tech company" → NO MATCH (doesn't support the specific claim)

## Output format
```json
{
    "sentence_index_A": [matched_memory_index_A, matched_memory_index_B]
}
```

Example:
```json
{
    "0": [1, 3],
    "1": [0],
    "5": [4]
}
```

Only include sentences that have supporting memories."""  # noqa: E501

        user_message = f"""## Available memories
{memories_text}

## Response sentences
{sentences_text}"""

        messages = [{
            'role': 'user',
            'content': [{
                'text': user_message
            }]
        }, {
            'role': 'assistant',
            'content': [{
                'text': '```json'
            }]
        }]

        try:
            response, _ = self.llm.generate_response(messages=messages, system_prompt=system_prompt, stop_sequences=['```'])
            return self._parse_mapping_response(response, sentences, memories)
        except Exception as e:
            logger.error(f'LLM mapping generation failed: {e}')
            return {}

    def _parse_mapping_response(self, response: str, sentences: List[str], memories: List[Tuple[str,
                                                                                                str]]) -> Dict[str, List[str]]:
        """Parse LLM response to extract mapping."""
        try:
            mapping_indices = json.loads(response)

            result = {}
            for sentence_idx, memory_indices in mapping_indices.items():
                try:
                    idx = int(sentence_idx)
                    if 0 <= idx < len(sentences):
                        memories_info = [memories[int(i)] for i in memory_indices if 0 <= int(i) < len(memories)]
                        if memories_info:
                            result[sentences[idx]] = memories_info
                except (ValueError, IndexError):
                    continue

            return result
        except json.JSONDecodeError:
            logger.error('Failed to parse LLM mapping response as JSON')
            return {}

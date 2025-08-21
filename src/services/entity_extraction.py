"""
Entity Extraction Service with two-step LLM process.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..models.core import Entity, Memory
from ..utils.bedrock_embed import BedrockEmbed, BedrockEmbedError
from ..utils.bedrock_llm import BedrockLLM, BedrockLLMError
from ..utils.config import config
from ..utils.json_utils import clean_json_response
from ..utils.logging_config import get_logger
from ..utils.timestamp_utils import to_datetime

logger = get_logger(__name__)


class EntityExtractionError(Exception):
    """Custom exception for entity extraction errors."""
    pass


class EntityExtractionService:
    """Extract entities and relationships from user messages using Bedrock LLMs."""

    def __init__(self):
        """Initialize the entity extraction service."""
        self.llm = BedrockLLM(config.bedrock_llm)
        self.embed = BedrockEmbed(config.bedrock_embed)

        logger.info('Initialized EntityExtractionService')

    def extract_entities(self, user_id: str, messages: List[Dict[str, str]], timestamp: Optional[int] = None) -> List[Entity]:
        """First LLM call: Extract entities from messages.

        Args:
            user_id: User ID for entity isolation
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            List of extracted Entity objects

        Raises:
            EntityExtractionError: If entity extraction fails
        """
        if not messages:
            logger.warning('Empty messages provided for entity extraction')
            return []

        try:
            content_list = []
            for msg in messages:
                if msg.get('role') in ['user', 'assistant'] and msg.get('content', '').strip():
                    content_list.append(f'{msg["role"].capitalize()}:\n{msg["content"]}')

            if not content_list:
                logger.debug('No content found for entity extraction')
                return []

            content = '\n\n'.join(content_list)

            system_prompt = f"""
You are an expert entity extraction system. Extract entities from the conversation.

Extract entities that are:
- People (names, roles, relationships)
- Places (locations, addresses, venues)
- Organizations (companies, institutions, groups)
- Concepts (ideas, topics, subjects)
- Events (meetings, activities, occasions)
- Objects (items, products, things)
- Severities (severity level, severity score, ...)

Special handling for pronouns:
- If USER content contains self reference such as 'I', 'me', 'my' etc. then use user's ID {user_id} as the source entity
- If ASSISTANT content contains self reference such as 'I', 'me', 'my' etc. IGNORE

Return a JSON array of entities with this exact format:
```json
[
  {{
    "name": "entity name",
    "type": "person|place|organization|concept|event|object|severity"
  }}
]
```

Only extract entities that are explicitly mentioned. Do not infer or assume entities.
Return empty array [] if no entities found."""

            llm_messages = [{
                'role': 'user',
                'content': [{
                    'text': f'Extract entities from the conversation:\n{content}'
                }]
            }, {
                'role': 'assistant',
                'content': [{
                    'text': '```json'
                }]
            }]

            response, _ = self.llm.generate_response(messages=llm_messages, system_prompt=system_prompt, stop_sequences=['```'])
            # Parse JSON response
            try:
                # Clean response - remove code block markers if present
                cleaned_response = clean_json_response(response)
                entities_data = json.loads(cleaned_response)
                if not isinstance(entities_data, list):
                    logger.warning(f'Expected list, got {type(entities_data)}')
                    return []
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse entity extraction JSON: {e}')
                return []

            entities = []
            current_time = to_datetime(timestamp)

            for entity_data in entities_data:
                if not isinstance(entity_data, dict):
                    continue

                name = entity_data.get('name', '').strip().lower()
                entity_type = entity_data.get('type', '').strip().lower()

                if not name or not entity_type:
                    continue

                # Generate embedding for entity name
                try:
                    embedding = self.embed.embed_document(name)
                except BedrockEmbedError as e:
                    logger.warning(f"Failed to generate embedding for entity '{name}': {e}")
                    embedding = [0.0] * config.bedrock_embed.dimension

                entity = Entity(id=str(uuid.uuid4()),
                                user_id=user_id,
                                name=name,
                                type=entity_type,
                                embedding=embedding,
                                created_at=current_time)

                entities.append(entity)

            logger.debug(f'Extracted {len(entities)} entities from message')
            return entities

        except BedrockLLMError as e:
            logger.error(f'LLM error during entity extraction: {e}')
            raise EntityExtractionError(f'Entity extraction failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during entity extraction: {e}')
            raise EntityExtractionError(f'Unexpected entity extraction error: {e}')

    def extract_entities_and_memories(self,
                                      user_id: str,
                                      messages: List[Dict[str, str]],
                                      timestamp: Optional[int] = None) -> tuple:
        """Extract both entities and memories from messages.

        Args:
            user_id: User ID for isolation
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Tuple of (entities, memories)

        Raises:
            EntityExtractionError: If extraction fails
        """
        if not messages:
            logger.warning('Empty messages provided for extraction')
            return [], []

        try:
            # Extract entities from user messages
            entities = self.extract_entities(user_id, messages, timestamp)
            if not entities:
                logger.debug('No entities found, no memories to extract')
                return [], []
            if len(entities) == 1:
                logger.debug('Only one entity found, no memories to extract')
                return entities, []

            content_list = []
            for msg in messages:
                if msg.get('role') in ['user', 'assistant'] and msg.get('content', '').strip():
                    content_list.append(f'{msg["role"].capitalize()}:\n{msg["content"]}')

            if not content_list:
                logger.debug('No content found for entity extraction')
                return []

            content = '\n\n'.join(content_list)

            if not content.strip():
                return entities, []

            entity_names = [entity.name for entity in entities]

            system_prompt = f"""
You are an expert relationship extraction system. Extract relationships between entities from the conversation.

Entities found in the message: {', '.join(entity_names)}

Extract relationships that connect these entities. Create natural language statements in subject-predicate-object format.

Special handling for pronouns in relationships:
- If USER content contains self reference such as 'I', 'me', 'my' etc. then use user's ID {user_id} as the source entity
- If ASSISTANT content contains self reference such as 'I', 'me', 'my' etc. IGNORE


Return a JSON array of memories with this exact format:
```json
[
  {{
    "statement": "natural language statement describing the relationship",
    "subject_name": "subject entity name",
    "predicate_name": "relationship/action name",
    "object_name": "object entity name",
    "confidence": 0.95
  }}
]
```

Only extract relationships that are explicitly stated or strongly implied in the message.
Confidence should be between 0.0 and 1.0.
Return empty array [] if no relationships found."""

            llm_messages = [{
                'role': 'user',
                'content': [{
                    'text': f'Extract relationships from the conversation:\n{content}'
                }]
            }, {
                'role': 'assistant',
                'content': [{
                    'text': '```json'
                }]
            }]

            response, _ = self.llm.generate_response(messages=llm_messages, system_prompt=system_prompt, stop_sequences=['```'])
            # Parse JSON response
            try:
                # Clean response - remove code block markers if present
                cleaned_response = clean_json_response(response)
                memories_data = json.loads(cleaned_response)
                if not isinstance(memories_data, list):
                    logger.warning(f'Expected list, got {type(memories_data)}')
                    return entities, []
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse memory extraction JSON: {e}')
                return entities, []

            memories = []
            current_time = to_datetime(timestamp)
            expires_at = datetime.fromtimestamp(current_time.timestamp() + (config.memory.default_expiration_days * 24 * 3600))

            for memory_data in memories_data:
                if not isinstance(memory_data, dict):
                    continue

                statement = memory_data.get('statement', '').strip()
                subject_name = memory_data.get('subject_name', '').strip().lower()
                predicate_name = memory_data.get('predicate_name', '').strip().lower()
                object_name = memory_data.get('object_name', '').strip().lower()
                confidence = memory_data.get('confidence', 0.0)

                if not all([statement, subject_name, predicate_name, object_name]):
                    continue

                # Validate confidence
                try:
                    confidence = float(confidence)
                    if not 0.0 <= confidence <= 1.0:
                        confidence = 0.0
                except (ValueError, TypeError):
                    confidence = 0.0

                # entity name to id
                entities_name_to_id = {e.name.strip().lower(): e.id for e in entities}

                # Check if subject and object entities exist
                if subject_name not in entities_name_to_id or object_name not in entities_name_to_id:
                    logger.debug('Skip memory due to missing subject or object!')
                    continue

                memory = Memory(id=str(uuid.uuid4()),
                                user_id=user_id,
                                statement=statement,
                                subject_id=entities_name_to_id[subject_name],
                                object_id=entities_name_to_id[object_name],
                                subject_name=subject_name,
                                predicate_name=predicate_name,
                                object_name=object_name,
                                confidence=confidence,
                                source_message=content,
                                created_at=current_time,
                                expires_at=expires_at)

                memories.append(memory)

            logger.debug(f'Extracted {len(memories)} memories from message')
            return entities, memories

        except EntityExtractionError as e:
            # Re-raise entity extraction errors as memory extraction errors
            logger.error(f'Entity extraction error during memory extraction: {e}')
            raise EntityExtractionError(f'Memory extraction failed: {e}')
        except BedrockLLMError as e:
            logger.error(f'LLM error during memory extraction: {e}')
            raise EntityExtractionError(f'Memory extraction failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during memory extraction: {e}')
            raise EntityExtractionError(f'Memory extraction failed: {e}')

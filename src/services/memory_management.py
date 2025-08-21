"""
Memory Management Service for unified memory operations.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from ..models.core import Entity, Memory
from ..utils.bedrock_embed import BedrockEmbed, BedrockEmbedError
from ..utils.bedrock_llm import BedrockLLM, BedrockLLMError
from ..utils.bedrock_rerank import BedrockRerank, BedrockRerankError
from ..utils.config import config
from ..utils.json_utils import clean_json_response
from ..utils.logging_config import get_logger
from ..utils.neptune_client import NeptuneClient, NeptuneError
from ..utils.opensearch_client import OpenSearchClient, OpenSearchError
from ..utils.timestamp_utils import to_seconds_str
from .entity_extraction import EntityExtractionError, EntityExtractionService

logger = get_logger(__name__)


class MemoryManagementError(Exception):
    """Custom exception for memory management errors."""
    pass


class MemoryManagementService:
    """Unified service for memory operations including processing, retrieval, updates, and expiration."""

    def __init__(self):
        """Initialize the memory management service."""
        self.neptune = NeptuneClient(config.neptune)
        self.opensearch = OpenSearchClient(config.opensearch)
        self.embed = BedrockEmbed(config.bedrock_embed)
        self.llm = BedrockLLM(config.bedrock_llm)
        self.rerank = BedrockRerank(config.bedrock_rerank)
        self.entity_extraction = EntityExtractionService()

        # Initialize memory log file
        # self.memory_log_path = 'memory_operations.txt'

        # Initialize OpenSearch indexes
        try:
            self.opensearch.create_index_if_not_exists(index_type='entity')
            self.opensearch.create_index_if_not_exists(index_type='statement')
        except OpenSearchError as e:
            logger.warning(f'Failed to create OpenSearch indexes: {e}')

        logger.info('Initialized MemoryManagementService')

    def add(self, user_id: str, messages: List[Dict[str, str]], timestamp: Optional[int] = None) -> None:
        """Process messages to extract and store memories.

        Args:
            user_id: User ID for isolation
            messages: List of message dicts with 'role' and 'content' keys

        Raises:
            MemoryManagementError: If memory processing fails
        """
        if not messages:
            logger.warning('Empty messages provided for memory processing')
            return

        try:
            # Extract entities and relationships
            entities, memories = self.entity_extraction.extract_entities_and_memories(user_id, messages, timestamp)
            if not entities and not memories:
                logger.debug('No entities or memories extracted from message')
                return
            # Build entity mapping (original_id -> final_id, original_id --> final_name)
            entity_id_mapping = {}
            entity_name_mapping = {}
            for entity in entities:
                final_entity_id, final_entity_name = self._build_entity_or_reuse(entity)
                entity_id_mapping[entity.id] = final_entity_id or entity.id
                entity_name_mapping[entity.id] = final_entity_name or entity.name

            # Update memory entities with final IDs and Names
            for memory in memories:
                if memory.subject_id in entity_id_mapping and memory.subject_id in entity_name_mapping:
                    memory.subject_name = entity_name_mapping[memory.subject_id]
                    memory.subject_id = entity_id_mapping[memory.subject_id]
                if memory.object_id in entity_id_mapping and memory.object_id in entity_name_mapping:
                    memory.object_name = entity_name_mapping[memory.object_id]
                    memory.object_id = entity_id_mapping[memory.object_id]

                self._build_memory(memory)

            logger.debug(f'Processed {len(entities)} entities and {len(memories)} memories')

        except EntityExtractionError as e:
            logger.error(f'Entity extraction error during memory processing: {e}')
            raise MemoryManagementError(f'Memory add failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during memory processing: {e}')
            raise MemoryManagementError(f'Memory add failed: {e}')

    def search_graph(self, user_id: str, messages: List[Dict[str, str]], hops: int = 2) -> List[Memory]:
        """Search memories using graph traversal.

        Args:
            user_id: User ID for isolation
            messages: List of message dicts with 'role' and 'content' keys
            hops: Number of hops to traverse (default: 2)

        Returns:
            List of Memory objects from graph search
        """
        if not messages:
            return []

        # Get the last user message as query
        query = None
        for msg in reversed(messages):
            if msg.get('role') == 'user' and msg.get('content', '').strip():
                query = msg['content']
                break

        if not query:
            return []

        # Extract entities from query
        query_messages = [{'role': 'user', 'content': query}]
        query_entities = self.entity_extraction.extract_entities(user_id, query_messages)

        all_entity_names = set()
        for entity in query_entities:
            entity_semantic_results = self.opensearch.hybrid_search(query_text=entity.name,
                                                                    query_vector=entity.embedding,
                                                                    user_id=user_id,
                                                                    top_k=3,
                                                                    index_type='entity')
            for result in entity_semantic_results:
                entity_name = result['document'].get('name', '')
                if entity_name:
                    all_entity_names.add(entity_name)

        # Get 2-hop graph connections
        graph_memories = []
        for entity_name in all_entity_names:
            related_memories = self.neptune.get_related_memories(user_id=user_id, entity_name=entity_name, hops=hops)
            graph_memories.extend(related_memories)

        # Deduplicate by memory ID
        seen_ids = set()
        unique_memories = []
        for memory in graph_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                unique_memories.append(memory)

        return unique_memories

    def search_vector(self, user_id: str, messages: List[Dict[str, str]], top_k: int = 10) -> List[Memory]:
        """Search memories using vector similarity.

        Args:
            user_id: User ID for isolation
            messages: List of message dicts with 'role' and 'content' keys
            top_k: Maximum number of results to return

        Returns:
            List of Memory objects from vector search
        """
        if not messages:
            return []

        # Get the last user message as query
        query = None
        for msg in reversed(messages):
            if msg.get('role') == 'user' and msg.get('content', '').strip():
                query = msg['content']
                break

        if not query:
            return []

        # Vector search on statements
        query_embedding = self.embed.embed_query(query)
        statement_results = self.opensearch.hybrid_search(query_text=query,
                                                          query_vector=query_embedding,
                                                          user_id=user_id,
                                                          top_k=top_k,
                                                          index_type='statement')

        memories = []
        for result in statement_results:
            doc = result['document']
            if doc.get('statement'):
                memories.append(
                    Memory(
                        id=doc.get('id', ''),
                        user_id=doc.get('user_id', ''),
                        statement=doc.get('statement', ''),
                        subject_id=doc.get('subject_id', ''),
                        subject_name=doc.get('subject_name', ''),
                        predicate_name=doc.get('predicate_name', ''),
                        object_id=doc.get('object_id', ''),
                        object_name=doc.get('object_name', ''),
                        confidence=float(doc.get('confidence', 1.0)),
                        source_message=doc.get('source_message', ''),
                        created_at=datetime.fromisoformat(doc.get('created_at', '1970-01-01T00:00:00')) if
                        isinstance(doc.get('created_at'), str) and 'T' in doc.get('created_at', '') else datetime.fromtimestamp(
                            int(doc.get('created_at', 0))) if doc.get('created_at') else datetime.fromtimestamp(0),
                        expires_at=datetime.fromisoformat(doc.get('expires_at', '2099-12-31T23:59:59'))
                        if isinstance(doc.get('expires_at'), str) and 'T' in doc.get('expires_at', '') else
                        datetime.fromtimestamp(int(doc.get('expires_at', 4102444800)))
                        if doc.get('expires_at') else datetime.fromtimestamp(4102444800)))

        return memories

    def search(self, user_id: str, messages: List[Dict[str, str]], top_k: int = 10, hops: int = 2) -> List[Memory]:
        """Search memories using both graph and vector search, then rerank results.

        Args:
            user_id: User ID for isolation
            messages: List of message dicts with 'role' and 'content' keys
            top_k: Maximum number of results to return
            hops: Number of hops to traverse (default: 2)

        Returns:
            List of relevant Memory objects

        Raises:
            MemoryManagementError: If search fails
        """
        if not messages:
            logger.warning('Empty messages provided for memory search')
            return []

        try:
            # Get the last user message as query
            query = None
            for msg in reversed(messages):
                if msg.get('role') == 'user' and msg.get('content', '').strip():
                    query = msg['content']
                    break

            if not query:
                logger.warning('No user message found for memory search')
                return []

            # Get results from both search methods
            graph_memories = self.search_graph(user_id, messages, hops=hops)
            vector_memories = self.search_vector(user_id, messages, top_k)

            # Combine and deduplicate by memory ID
            all_memories = []
            seen_ids = set()
            for memory in graph_memories + vector_memories:
                if memory.id not in seen_ids:
                    seen_ids.add(memory.id)
                    all_memories.append(memory)

            if not all_memories:
                logger.debug('No memories found for query')
                return []

            logger.debug(f'Found {len(all_memories)} memories for query: {query}')

            # Rerank combined results using statements
            memory_statements = [memory.statement for memory in all_memories]
            rerank_results = self.rerank.rerank(query, memory_statements, top_k=top_k)

            # Map reranked statements back to Memory objects
            statement_to_memory = {memory.statement: memory for memory in all_memories}
            reranked_memories = []
            for result in rerank_results:
                statement = result['document']
                if statement in statement_to_memory:
                    reranked_memories.append(statement_to_memory[statement])

            logger.debug(f'Return {len(reranked_memories)} memories')
            return reranked_memories

        except (OpenSearchError, NeptuneError, BedrockEmbedError, EntityExtractionError, BedrockRerankError) as e:
            logger.error(f'Service error during memory search: {e}')
            raise MemoryManagementError(f'Memory search failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during memory search: {e}')
            raise MemoryManagementError(f'Memory search failed: {e}')

    def delete(self, user_id: str, memory_id: str) -> None:
        """Delete memory from Neptune and mark as deleted in OpenSearch.

        Args:
            user_id: User ID for Neptune deletion
            memory_id: Memory ID to delete from Neptune
        """
        if not memory_id or not memory_id.strip():
            logger.warning('Empty memory ID provided for deletion')
            return False

        # Use get_document to find OpenSearch ID
        doc_result = self.opensearch.get_document(user_id=user_id, info_id=memory_id, index_type='statement')
        if doc_result and doc_result.get('id', None).strip() not in [None, '']:
            opensearch_memory_id = doc_result['id']
        else:
            logger.warning(f'No OpenSearch document found for memory ID: {memory_id}')
            return False

        try:
            self.neptune.delete_memory(memory_id, user_id)
            self.opensearch.delete_document(doc_id=opensearch_memory_id, index_type='statement')
            logger.debug(f'Deleted memory: {memory_id}')
            return True
        except NeptuneError as e:
            logger.error(f'Neptune error during memory deletion: {e}')
            raise MemoryManagementError(f'Memory deletion failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during memory deletion: {e}')
            raise MemoryManagementError(f'Memory deletion failed: {e}')

    def cleanup_expired_memories(self, user_id: Optional[str] = None) -> int:
        """Remove expired memories, optionally for specific user.

        Args:
            user_id: Optional user ID to limit cleanup to specific user

        Returns:
            Number of memories deleted

        Raises:
            MemoryManagementError: If cleanup fails
        """
        try:
            deleted_count = self.neptune.cleanup_expired_memories(user_id)

            if deleted_count > 0:
                logger.info(f'Cleaned up {deleted_count} expired memories')
            else:
                logger.debug('No expired memories found for cleanup')

            return deleted_count

        except NeptuneError as e:
            logger.error(f'Neptune error during memory cleanup: {e}')
            raise MemoryManagementError(f'Memory cleanup failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during memory cleanup: {e}')
            raise MemoryManagementError(f'Memory cleanup failed: {e}')

    def _build_entity_or_reuse(self, entity: Entity) -> Optional[str]:
        """Build entity vector database with similarity detection.

        Args:
            entity: Entity to process

        Returns:
            Final entity ID (reused or new)
            Final entity Name (reused or new)
        """
        try:
            # Check for similar entities using vector search
            similar_entities = self.opensearch.hybrid_search(query_text=entity.name,
                                                             query_vector=entity.embedding,
                                                             user_id=entity.user_id,
                                                             top_k=5,
                                                             index_type='entity')
            if similar_entities:
                for entity_info in similar_entities:
                    if entity_info['document'].get('name') == entity.name:
                        logger.debug(f"Similar entity found: {entity_info['document'].get('id')}")
                        return entity_info['document'].get('id'), entity_info['document'].get('name')
                # Use LLM to decide whether to create new entity or reuse existing one
                existing_entity_id, existing_entity_name = self._decide_entity_reuse(entity, similar_entities)
                if existing_entity_id and existing_entity_name:
                    # Update existing entity
                    logger.debug(f'Reusing existing entity: {existing_entity_name}({existing_entity_id})')
                    return existing_entity_id, existing_entity_name
            # Create new entity in Neptune
            success = self.neptune.create_entity_vertex(entity_id=entity.id,
                                                        user_id=entity.user_id,
                                                        name=entity.name,
                                                        entity_type=entity.type,
                                                        created_at=to_seconds_str(int(entity.created_at.timestamp())))

            if success:
                # Index entity in OpenSearch for similarity search
                document = {
                    'id': entity.id,
                    'user_id': entity.user_id,
                    'name': entity.name,
                    'type': entity.type,
                    'embedding': entity.embedding,
                    'created_at': entity.created_at.isoformat()
                }

                self.opensearch.index_document(document, index_type='entity')
                logger.debug(f'Created and indexed new entity: {entity.id}')
                return entity.id, entity.name
            else:
                logger.warning(f'Failed to create entity in Neptune: {entity.id}')
                return None, None

        except (NeptuneError, OpenSearchError) as e:
            logger.error(f'Error processing entity {entity.id}: {e}')
            raise MemoryManagementError(f'Entity processing failed: {e}')

    def _build_memory(self, memory: Memory) -> None:
        """Build memory with similarity check for existing memories.

        Args:
            memory: Memory to process
        """
        try:
            related_memories_subject = self.neptune.get_related_memories(user_id=memory.user_id,
                                                                         entity_name=memory.subject_name,
                                                                         hops=1)
            related_memories_object = self.neptune.get_related_memories(user_id=memory.user_id,
                                                                        entity_name=memory.object_name,
                                                                        hops=1)
            # Combine and deduplicate Memory objects
            seen_ids = set()
            graph_memories = []
            for mem in related_memories_subject + related_memories_object:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    graph_memories.append(mem)

            if len(graph_memories):
                # Extract statements for reranking
                memory_statements = [mem.statement for mem in graph_memories]
                if len(memory_statements) > 20:
                    rerank_results = self.rerank.rerank(memory.statement, memory_statements, top_k=20)
                    # Map back to Memory objects
                    statement_to_memory = {mem.statement: mem for mem in graph_memories}
                    graph_memories = [
                        statement_to_memory[res['document']] for res in rerank_results if res['document'] in statement_to_memory
                    ]
                    memory_statements = [mem.statement for mem in graph_memories]

                # Use LLM to decide memory operations
                create_or_not, delete_statements = self._decide_memory_operations(memory, memory_statements)
                # Delete specified statements
                for statement in delete_statements:
                    # Find the memory with matching statement
                    for mem in graph_memories:
                        if mem.statement.lower().strip() == statement.lower().strip():
                            # Log deletion
                            # self._log_memory_operation('delete', memory.user_id, memory.statement, mem.statement,
                            #                            memory.subject_name, memory.object_name)
                            self.delete(memory.user_id, mem.id)
                            break

                # Create new memory if instructed
                if create_or_not:
                    self._create_new_memory(memory)
                    # self._log_memory_operation('create', memory.user_id, memory.statement, None, memory.subject_name,
                    #                            memory.object_name)
                else:
                    pass
                    # # Log when memory is not created
                    # self._log_memory_operation('not_created',
                    #                         memory.user_id,
                    #                         memory.statement,
                    #                         None,
                    #                         memory.subject_name,
                    #                         memory.object_name,
                    #                         similar_statements=memory_statements)
            else:
                # Create new memory if no similar ones found
                self._create_new_memory(memory)
                # self._log_memory_operation('create', memory.user_id, memory.statement, None, memory.subject_name,
                #                            memory.object_name)

        except (BedrockEmbedError, OpenSearchError, NeptuneError) as e:
            logger.error(f'Error building memory {memory.id}: {e}')
            raise MemoryManagementError(f'Memory building failed: {e}')

    def _create_new_memory(self, memory: Memory) -> None:
        """Create new memory in Neptune and index statement.

        Args:
            memory: Memory to create
        """
        # Create memory edge in Neptune
        success = self.neptune.create_memory_edge(memory_id=memory.id,
                                                  user_id=memory.user_id,
                                                  subject_id=memory.subject_id,
                                                  object_id=memory.object_id,
                                                  statement=memory.statement,
                                                  subject_name=memory.subject_name,
                                                  predicate_name=memory.predicate_name,
                                                  object_name=memory.object_name,
                                                  confidence=memory.confidence,
                                                  source_message=memory.source_message,
                                                  created_at=to_seconds_str(int(memory.created_at.timestamp())),
                                                  expires_at=to_seconds_str(int(memory.expires_at.timestamp())))

        if success:
            # Index statement in OpenSearch
            document = {
                'id': memory.id,
                'user_id': memory.user_id,
                'statement': memory.statement,
                'subject_id': memory.subject_id,
                'subject_name': memory.subject_name,
                'predicate_name': memory.predicate_name,
                'object_id': memory.object_id,
                'object_name': memory.object_name,
                'confidence': memory.confidence,
                'embedding': self.embed.embed_document(memory.statement),
                'created_at': memory.created_at.isoformat(),
                'expires_at': memory.expires_at.isoformat()
            }

            self.opensearch.index_document(document, index_type='statement')
            logger.debug(f'Created and indexed new memory: {memory.id}')
        else:
            logger.warning(f'Failed to create memory edge: {memory.id}')

    def _decide_entity_reuse(self, entity_or_name, similar_entities: List[Dict]) -> Optional[str]:
        """Use LLM to decide whether to reuse existing entity or create new one.

        Args:
            entity_or_name: New entity to process
            similar_entities: List of similar entities from vector search

        Returns:
            Entity ID to reuse, or None to create new entity
            Entity Name to reuse, or None to create new entity
        """
        try:
            similar_entities_id_to_name = {}
            candidates_info = ''
            for result in similar_entities:
                doc = result['document']
                candidates_info += f"ID: {doc.get('id')}:\nName: {doc.get('name')}\nType: {doc.get('type')}\n\n"
                similar_entities_id_to_name[doc.get('id')] = doc.get('name')
            entity_name = entity_or_name.name if isinstance(entity_or_name, Entity) else entity_or_name
            system_prompt = """You are an entity deduplication expert. Decide whether a new entity should reuse an existing similar entity or be created as a new entity.

Consider:
- Semantic similarity of names
- Entity types compatibility
- Avoid over-merging distinct entities

Respond with JSON:
```json
{
    "reuse": true/false,
    "entity_id": "id_if_reuse" or null,
    "reason": "explanation"}
}
```
"""  # noqa: E501

            user_message = f"""New entity: {entity_name}

Similar existing entities:
{candidates_info}"""

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

            response, _ = self.llm.generate_response(messages=messages, system_prompt=system_prompt, stop_sequences=['```'])

            try:
                cleaned_response = clean_json_response(response)
                decision = json.loads(cleaned_response)
                if decision.get('reuse', False):
                    entity_id = decision.get('entity_id')
                    reason = decision.get('reason', 'No reason provided')
                    logger.debug(f'LLM decided to reuse entity {entity_id}: {reason}')
                    return entity_id, similar_entities_id_to_name.get(entity_id)
                else:
                    reason = decision.get('reason', 'No reason provided')
                    logger.debug(f'LLM decided to create new entity: {reason}')
                    return None, None
            except json.JSONDecodeError:
                logger.warning(f'Failed to parse LLM response: {response}')
                return None, None

        except BedrockLLMError as e:
            logger.error(f'LLM error in entity reuse decision: {e}')
            return None, None
        except Exception as e:
            logger.error(f'Unexpected error in entity reuse decision: {e}')
            return None, None

    def _decide_memory_operations(self, new_memory: Memory, similar_statements: List[str]) -> tuple[bool, List[str]]:
        """Use LLM to decide memory operations (delete, update, create).

        Args:
            new_memory: New memory to process
            similar_statements: List of similar statement strings

        Returns:
            Tuple of (create_new, delete_statements)
        """
        try:
            # Prepare context for LLM
            candidates_info = ''
            for i, statement in enumerate(similar_statements):
                if statement:
                    candidates_info += f'Statement {i}: {statement}\n'

            system_prompt = """You are an AI memory management system. Your task is to analyze a new memory against existing memories and decide how to update the memory database.

## Input
- A new memory (i.e., statement)
- A list of existing related memories

## Task
1. DELETE: IF the new memory contradicts existing ones, THEN mark contradictory existing memories for deletion.
2. CREATE: After considering DELETION, decide if the new memory should be created:
   - Check if the new memory is redundant with the REMAINING memories (excluding the ones you marked for deletion)
   - Only create the new memory if it provides unique information not covered by the remaining memories

## Rule
ONLY delete if ALL these conditions are met:
- The statements are about the EXACT SAME specific entity/person/object
- They make contradictory claims about the SAME attribute
- Both statements cannot be true simultaneously

Valid deletion example:
- "John Smith is 25 years old" vs "John Smith is 30 years old" (same person, contradictory ages)

NEVER delete for examples:
- Different entities: "University of Chicago" vs "Harvard University" (different schools)
- Different topics: "algorithms" vs "pharmacies" vs "ships" (unrelated subjects)
- Different aspects: "Robert proposed merger" vs "Students are funded" (different facts about universities)
- Compatible facts that can coexist

When ANY doubt exists, keep both memories.

## Output format
```json
{
    "delete_indices": [],
    "create_new": true/false
}
```
"""  # noqa: E501
            user_message = f"""## Existing memories
{candidates_info}

## New memory
{new_memory.statement}
"""

            messages = [
                {
                    'role': 'user',
                    'content': [{
                        'text': user_message
                    }]
                },
                {
                    'role': 'assistant',
                    'content': [{
                        'text': '```json'
                    }]
                },
            ]

            response, _ = self.llm.generate_response(messages=messages, system_prompt=system_prompt, stop_sequences=['```'])
            # Parse LLM response
            try:
                cleaned_response = clean_json_response(response)
                decision = json.loads(cleaned_response)

                # Add operations from LLM decision
                delete_indices = decision.get('delete_indices', [])
                create_or_not = decision.get('create_new', True)
                # Convert indices to actual statements
                delete_statements = []
                for idx in delete_indices:
                    if 0 <= idx < len(similar_statements):
                        delete_statements.append(similar_statements[idx])

                return create_or_not, delete_statements
            except json.JSONDecodeError:
                logger.warning(f'Failed to parse LLM response: {response}')
                return True, []

        except BedrockLLMError as e:
            logger.error(f'LLM error in memory operations decision: {e}')
            return True, []
        except Exception as e:
            logger.error(f'Unexpected error in memory operations decision: {e}')
            return True, []

    # def _log_memory_operation(self,
    #                           action: str,
    #                           user_id: str,
    #                           statement: str,
    #                           target_statement: Optional[str] = None,
    #                           subject_name: Optional[str] = None,
    #                           object_name: Optional[str] = None,
    #                           similar_statements: Optional[List[str]] = None) -> None:
    #     """Log memory operations (create, delete, not_created) to file."""
    #     try:
    #         log_entry = {
    #             'timestamp': datetime.now().isoformat(),
    #             'user_id': user_id,
    #             'action': action,
    #             'statement': statement,
    #             'subject_name': subject_name,
    #             'object_name': object_name
    #         }

    #         if action == 'delete':
    #             log_entry['deleted_statement'] = target_statement
    #         elif action == 'not_created':
    #             log_entry['similar_statements'] = similar_statements

    #         with open(self.memory_log_path, 'a', encoding='utf-8') as f:
    #             f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    #     except Exception as e:
    #         logger.error(f'Failed to log memory operation: {e}')

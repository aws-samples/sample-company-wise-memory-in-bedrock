"""
OpenSearch client wrapper for vector similarity search.
"""

import time
from typing import Any, Dict, List, Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import OpenSearchException
from requests_aws4auth import AWS4Auth

from .config import OpenSearchConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class OpenSearchError(Exception):
    """Custom exception for OpenSearch errors."""
    pass


class OpenSearchClient:
    """OpenSearch client with AWS authentication and error handling."""

    def __init__(self, config: OpenSearchConfig):
        """
        Initialize OpenSearch client.

        Args:
            config: OpenSearchConfig instance with connection parameters
        """
        self.config = config

        # Get AWS credentials and create auth
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(region=config.region, service='aoss', refreshable_credentials=credentials)
        # Parse endpoint to get host and port
        endpoint = config.endpoint
        port = config.port
        if '://' in endpoint:
            # Remove protocol if present
            endpoint = endpoint.split('://', 1)[1]

        # Create OpenSearch client
        self.client = OpenSearch(hosts=[{
            'host': endpoint,
            'port': port
        }],
                                 http_auth=auth,
                                 use_ssl=True,
                                 verify_certs=True,
                                 connection_class=RequestsHttpConnection)

        logger.info(f'Initialized OpenSearch client for endpoint: {config.endpoint}')

    def create_index_if_not_exists(self, index_name: Optional[str] = None, index_type: str = 'entity') -> bool:
        """
        Create index if it doesn't exist.

        Args:
            index_name: Name of the index (uses config default if None)
            index_type: Type of index (entity or statement)

        Returns:
            True if index was created or already exists, False otherwise
        """
        if index_name is None:
            index_name = f'{self.config.index_name}_{index_type}'

        try:
            if self.client.indices.exists(index=index_name):
                logger.debug(f'Index {index_name} already exists')
                return 'exists'

            # Create index with vector field mapping
            if index_type == 'entity':
                index_body = {
                    'mappings': {
                        'properties': {
                            'id': {
                                'type': 'keyword'
                            },
                            'user_id': {
                                'type': 'keyword'
                            },
                            'name': {
                                'type': 'text'
                            },
                            'type': {
                                'type': 'keyword'
                            },
                            'embedding': {
                                'type': 'knn_vector',
                                'dimension': self.config.dimension,
                                'method': {
                                    'name': 'hnsw',
                                    'space_type': 'cosinesimil',
                                    'engine': 'nmslib'
                                }
                            },
                            'created_at': {
                                'type': 'date'
                            }
                        }
                    },
                    'settings': {
                        'index': {
                            'knn': True,
                            'knn.algo_param.ef_search': 100
                        }
                    }
                }
            else:  # statement index
                index_body = {
                    'mappings': {
                        'properties': {
                            'id': {
                                'type': 'keyword'
                            },
                            'user_id': {
                                'type': 'keyword'
                            },
                            'statement': {
                                'type': 'text'
                            },
                            'subject_id': {
                                'type': 'text'
                            },
                            'subject_name': {
                                'type': 'text'
                            },
                            'predicate_name': {
                                'type': 'text'
                            },
                            'object_id': {
                                'type': 'text'
                            },
                            'object_name': {
                                'type': 'text'
                            },
                            'confidence': {
                                'type': 'float'
                            },
                            'embedding': {
                                'type': 'knn_vector',
                                'dimension': self.config.dimension,
                                'method': {
                                    'name': 'hnsw',
                                    'space_type': 'cosinesimil',
                                    'engine': 'nmslib'
                                }
                            },
                            'created_at': {
                                'type': 'date'
                            },
                            'expires_at': {
                                'type': 'date'
                            }
                        }
                    },
                    'settings': {
                        'index': {
                            'knn': True,
                            'knn.algo_param.ef_search': 100
                        }
                    }
                }

            response = self.client.indices.create(index=index_name, body=index_body)
            logger.info(f'Created index {index_name}')
            if response.get('acknowledged', False):
                logger.info(f'Waiting 15s for index {index_name} sync-up...')
                time.sleep(15)
                return 'created'
            else:
                return 'failed'
        except OpenSearchException as e:
            logger.error(f'Error creating index {index_name}: {e}')
            raise OpenSearchError(f'Failed to create index: {e}')
        except Exception as e:
            logger.error(f'Unexpected error creating index {index_name}: {e}')
            raise OpenSearchError(f'Unexpected error creating index: {e}')

    def index_document(self, document: Dict[str, Any], index_name: Optional[str] = None, index_type: str = 'entity') -> bool:
        """
        Index a document in OpenSearch.

        Args:
            document: Document to index
            index_name: Name of the index (uses config default if None)
            index_type: Type of index (entity or statement)

        Returns:
            True if indexing was successful, False otherwise
        """
        if index_name is None:
            index_name = f'{self.config.index_name}_{index_type}'

        try:
            response = self.client.index(index=index_name, body=document)

            success = response.get('result') in ['created', 'updated']
            if success:
                logger.debug(f'Indexed document in {index_name}')
            else:
                logger.warning(f'Unexpected result indexing document: {response}')

            return success

        except OpenSearchException as e:
            logger.error(f'Error indexing document: {e}')
            raise OpenSearchError(f'Failed to index document: {e}')
        except Exception as e:
            logger.error(f'Unexpected error indexing document: {e}')
            raise OpenSearchError(f'Unexpected error indexing document: {e}')

    def vector_search(self,
                      query_vector: List[float],
                      user_id: str,
                      top_k: Optional[int] = 20,
                      index_name: Optional[str] = None,
                      index_type: str = 'entity') -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query vector for similarity search
            user_id: User ID to filter results
            top_k: Number of results to return (default 20)
            index_name: Name of the index (uses config default if None)
            index_type: Type of index (entity or statement)

        Returns:
            List of search results with scores and documents
        """
        if index_name is None:
            index_name = f'{self.config.index_name}_{index_type}'

        try:
            search_body = {
                'size': top_k,
                'query': {
                    'bool': {
                        'must': [{
                            'knn': {
                                'embedding': {
                                    'vector': query_vector,
                                    'k': top_k
                                }
                            }
                        }],
                        'filter': [{
                            'term': {
                                'user_id': user_id
                            }
                        }]
                    }
                },
                '_source': {
                    'excludes': ['embedding']  # Don't return embedding in results
                }
            }

            response = self.client.search(index=index_name, body=search_body)

            results = []
            for hit in response['hits']['hits']:
                result = {'id': hit['_id'], 'score': hit['_score'], 'document': hit['_source']}
                results.append(result)

            logger.debug(f'Vector search returned {len(results)} results for user {user_id}')
            return results

        except OpenSearchException as e:
            logger.error(f'Error performing vector search: {e}')
            raise OpenSearchError(f'Vector search failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error in vector search: {e}')
            raise OpenSearchError(f'Unexpected error in vector search: {e}')

    def keyword_search(self,
                       query_text: str,
                       user_id: str,
                       top_k: Optional[int] = 20,
                       index_name: Optional[str] = None,
                       index_type: str = 'statement') -> List[Dict[str, Any]]:
        """Perform keyword-based text search.

        Args:
            query_text: Text query for keyword search
            user_id: User ID to filter results
            top_k: Number of results to return
            index_name: Name of the index
            index_type: Type of index

        Returns:
            List of search results with scores and documents
        """
        if index_name is None:
            index_name = f'{self.config.index_name}_{index_type}'

        # Choose field based on index type
        field = 'name' if index_type == 'entity' else 'statement'

        try:
            search_body = {
                'size': top_k,
                'query': {
                    'bool': {
                        'must': [{
                            'match': {
                                field: query_text
                            }
                        }],
                        'filter': [{
                            'term': {
                                'user_id': user_id
                            }
                        }]
                    }
                },
                '_source': {
                    'excludes': ['embedding']
                }
            }

            response = self.client.search(index=index_name, body=search_body)

            results = []
            for hit in response['hits']['hits']:
                result = {'id': hit['_id'], 'score': hit['_score'], 'document': hit['_source']}
                results.append(result)

            logger.debug(f'Keyword search returned {len(results)} results for user {user_id}')
            return results

        except OpenSearchException as e:
            logger.error(f'Error performing keyword search: {e}')
            raise OpenSearchError(f'Keyword search failed: {e}')
        except Exception as e:
            logger.error(f'Unexpected error in keyword search: {e}')
            raise OpenSearchError(f'Unexpected error in keyword search: {e}')

    def hybrid_search(self,
                      query_text: str,
                      query_vector: List[float],
                      user_id: str,
                      top_k: Optional[int] = 20,
                      index_name: Optional[str] = None,
                      index_type: str = 'statement',
                      vector_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query_text: Text query for keyword search
            query_vector: Vector for similarity search
            user_id: User ID to filter results
            top_k: Number of results to return
            index_name: Name of the index
            index_type: Type of index
            vector_weight: Weight for vector search (0-1)

        Returns:
            List of search results with combined scores
        """
        # Get results from both searches
        vector_results = self.vector_search(query_vector, user_id, top_k * 2, index_name, index_type=index_type)
        keyword_results = self.keyword_search(query_text, user_id, top_k * 2, index_name, index_type=index_type)

        # Normalize scores to 0-1 range
        def normalize_scores(results):
            if not results:
                return results
            scores = [r['score'] for r in results]
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return results
            for result in results:
                result['score'] = (result['score'] - min_score) / (max_score - min_score)
            return results

        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)

        # Combine normalized scores
        combined_results = {}
        keyword_weight = 1.0 - vector_weight

        # Add vector results
        for result in vector_results:
            doc_id = result['id']
            combined_results[doc_id] = {'document': result['document'], 'vector_score': result['score'], 'keyword_score': 0.0}

        # Add keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['score']
            else:
                combined_results[doc_id] = {
                    'document': result['document'],
                    'vector_score': 0.0,
                    'keyword_score': result['score']
                }

        # Calculate combined scores and sort
        final_results = []
        for doc_id, data in combined_results.items():
            combined_score = (data['vector_score'] * vector_weight + data['keyword_score'] * keyword_weight)
            final_results.append({'id': doc_id, 'score': combined_score, 'document': data['document']})

        # Sort by combined score and return top_k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]

    def get_document(self,
                     user_id: str,
                     info_id: str,
                     index_name: Optional[str] = None,
                     index_type: str = 'statement') -> Optional[Dict[str, Any]]:
        """
        Get a specific document by user_id and document id.

        Args:
            user_id: User ID to filter results
            info_id: Information ID to retrieve
            index_name: Name of the index (uses config default if None)
            index_type: Type of index (entity or statement)

        Returns:
            Document if found, None otherwise
        """
        if index_name is None:
            index_name = f'{self.config.index_name}_{index_type}'

        try:
            search_body = {
                'size': 1,
                'query': {
                    'bool': {
                        'must': [{
                            'term': {
                                'user_id': user_id
                            }
                        }, {
                            'term': {
                                'id': info_id
                            }
                        }]
                    }
                },
                '_source': {
                    'excludes': ['embedding']
                }
            }

            response = self.client.search(index=index_name, body=search_body)

            if response['hits']['total']['value'] > 0:
                hit = response['hits']['hits'][0]
                return {'id': hit['_id'], 'score': hit['_score'], 'document': hit['_source']}

            return None

        except OpenSearchException as e:
            logger.error(f'Error getting document {info_id} for user {user_id}: {e}')
            raise OpenSearchError(f'Failed to get document: {e}')
        except Exception as e:
            logger.error(f'Unexpected error getting document {info_id}: {e}')
            raise OpenSearchError(f'Unexpected error getting document: {e}')

    def delete_document(self, doc_id: str, index_name: Optional[str] = None, index_type: str = 'statement') -> bool:
        """
        Delete a document from the index.

        Args:
            doc_id: Document ID to delete
            index_name: Name of the index (uses config default if None)

        Returns:
            True if deletion was successful, False otherwise
        """
        if index_name is None:
            index_name = f'{self.config.index_name}_{index_type}'

        try:
            response = self.client.delete(index=index_name, id=doc_id)

            success = response.get('result') == 'deleted'
            if success:
                logger.debug(f'Deleted document {doc_id} from {index_name}')
            else:
                logger.warning(f'Document {doc_id} not found for deletion')

            return success

        except OpenSearchException as e:
            # OpenSearchException args: (status_code, error_type, error_info)
            if len(e.args) >= 2 and (e.args[0] == 404 or e.args[1] == 'not_found'):
                logger.warning(f'Document {doc_id} not found for deletion')
                return False
            logger.error(f'Error deleting document {doc_id}: {e}')
            raise OpenSearchError(f'Failed to delete document: {e}')
        except Exception as e:
            logger.error(f'Unexpected error deleting document {doc_id}: {e}')
            raise OpenSearchError(f'Unexpected error deleting document: {e}')

    def cleanup(self) -> bool:
        """
        Clean up all indices from OpenSearch.

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Delete entity index
            entity_index = f'{self.config.index_name}_entity'
            if self.client.indices.exists(index=entity_index):
                self.client.indices.delete(index=entity_index)
                logger.info(f'✓ Deleted entity index: {entity_index}')
            else:
                logger.info(f'Entity index {entity_index} does not exist')

            # Delete statement index
            statement_index = f'{self.config.index_name}_statement'
            if self.client.indices.exists(index=statement_index):
                self.client.indices.delete(index=statement_index)
                logger.info(f'✓ Deleted statement index: {statement_index}')
            else:
                logger.info(f'Statement index {statement_index} does not exist')

            return True

        except OpenSearchException as e:
            logger.error(f'Error during OpenSearch cleanup: {e}')
            raise OpenSearchError(f'Failed to cleanup OpenSearch: {e}')
        except Exception as e:
            logger.error(f'Unexpected error during OpenSearch cleanup: {e}')
            raise OpenSearchError(f'Unexpected error during OpenSearch cleanup: {e}')

    def health_check(self) -> bool:
        """
        Perform a health check on the OpenSearch service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.client.indices.exists(index='test_index')

            return response in [True, False]

        except Exception as e:
            logger.error(f'OpenSearch health check failed: {e}')
            return False

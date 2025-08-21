"""
Amazon Neptune graph database client with Gremlin Python driver and AWS SigV4 authentication.
"""

from datetime import datetime
from functools import wraps
from typing import List, Optional

from boto3 import Session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from gremlin_python.driver.aiohttp.transport import AiohttpTransport
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import P

from ..models.core import Memory
from .config import NeptuneConfig
from .logging_config import get_logger
from .timestamp_utils import to_seconds_str

logger = get_logger(__name__)


class NeptuneError(Exception):
    """Custom exception for Neptune errors."""
    pass


def retry_on_connection_error(func):
    """Decorator to retry Neptune operations on connection errors."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if 'cannot write to closing transport' in str(e).lower():
                logger.warning(f'Connection error detected: {e}. Reconnecting...')
                self.close()
                self._connect()
                try:
                    return func(self, *args, **kwargs)
                except Exception as retry_e:
                    logger.error(f'Error in {func.__name__}: {retry_e}')
                    raise NeptuneError(f'Failed to {func.__name__}: {retry_e}')
            else:
                logger.error(f'Error in {func.__name__}: {e}')
                raise NeptuneError(f'Failed to {func.__name__}: {e}')

    return wrapper


class NeptuneClient:
    """Amazon Neptune client using Gremlin Python driver with AWS authentication."""

    def __init__(self, config: NeptuneConfig):
        """
        Initialize Neptune client with Gremlin driver.

        Args:
            config: NeptuneConfig instance with connection parameters
        """
        self.config = config
        self.connection = None
        self.g = None
        self._connect()

        logger.info(f'Connected to Neptune at {config.endpoint}')

    def _connect(self):
        """Establish connection to Neptune."""
        # Build WebSocket connection string
        conn_string = f'wss://{self.config.endpoint}:8182/gremlin'

        # Get AWS credentials
        credentials = Session().get_credentials()
        if credentials is None:
            raise NeptuneError('No AWS credentials found')
        creds = credentials.get_frozen_credentials()

        # Get region
        region = Session().region_name or self.config.region or 'us-east-1'

        # Create signed request for WebSocket connection
        request = AWSRequest(method='GET', url=conn_string, data=None)
        SigV4Auth(creds, 'neptune-db', region).add_auth(request)

        # Initialize Gremlin connection
        self.connection = DriverRemoteConnection(conn_string,
                                                 'g',
                                                 headers=request.headers.items(),
                                                 transport_factory=lambda: AiohttpTransport(call_from_event_loop=True))

        try:
            self.g = traversal().with_remote(self.connection)
        except Exception:
            self.g = traversal().withRemote(self.connection)

    def close(self):
        """Close the Neptune connection."""
        if hasattr(self, 'connection'):
            self.connection.close()

    @retry_on_connection_error
    def create_entity_vertex(self,
                             entity_id: str,
                             user_id: str,
                             name: str,
                             entity_type: str,
                             created_at: Optional[str] = None) -> bool:
        """
        Create an entity vertex in the graph.

        Args:
            entity_id: Unique entity identifier
            user_id: User ID for isolation
            name: Entity name
            entity_type: Type of entity
            created_at: Creation timestamp

        Returns:
            True if creation was successful, False otherwise
        """
        created_at = created_at or to_seconds_str()

        # Check if vertex already exists
        existing = self.g.V().has('id', entity_id).to_list()
        if existing:
            logger.debug(f'Entity vertex already exists: {entity_id}')
            return True

        # Create new vertex
        traversal = self.g.addV('Entity').property('id', entity_id)\
            .property('user_id', user_id)\
            .property('name', name)\
            .property('type', entity_type)\
            .property('created_at', created_at)

        traversal.next()
        logger.debug(f'Created entity vertex: {entity_id}')
        return True

    @retry_on_connection_error
    def create_memory_edge(self,
                           memory_id: str,
                           user_id: str,
                           subject_id: str,
                           object_id: str,
                           statement: str,
                           subject_name: str,
                           predicate_name: str,
                           object_name: str,
                           confidence: float = 1.0,
                           source_message: Optional[str] = None,
                           created_at: Optional[str] = None,
                           expires_at: Optional[str] = None) -> bool:
        """
        Create a memory edge between two entity vertices.

        Args:
            memory_id: Unique memory identifier
            user_id: User ID for isolation
            subject_id: Subject entity ID
            object_id: Object entity ID
            statement: Natural language statement
            subject_name: Subject entity name
            predicate_name: Predicate/relationship name
            object_name: Object entity name
            confidence: Confidence score (0.0 to 1.0)
            source_message: Original message that generated this memory
            created_at: Creation timestamp
            expires_at: Expiration timestamp

        Returns:
            True if creation was successful, False otherwise
        """
        created_at = created_at or to_seconds_str()

        # Find subject and object vertices
        subject = self.g.V().has('id', subject_id).next()
        object_vertex = self.g.V().has('id', object_id).next()

        # Create edge
        traversal = self.g.V(subject).addE('Memory').to(object_vertex)\
            .property('id', memory_id)\
            .property('user_id', user_id)\
            .property('statement', statement)\
            .property('subject_id', subject_id)\
            .property('subject_name', subject_name)\
            .property('predicate_name', predicate_name)\
            .property('object_id', object_id)\
            .property('object_name', object_name)\
            .property('confidence', confidence)\
            .property('created_at', created_at)

        if source_message:
            traversal = traversal.property('source_message', source_message)

        if expires_at:
            traversal = traversal.property('expires_at', expires_at)

        traversal.next()
        logger.debug(f'Created memory edge: {memory_id}')
        return True

    @retry_on_connection_error
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete a specific memory edge.

        Args:
            memory_id: Memory ID to delete
            user_id: User ID for security check

        Returns:
            True if deletion was successful, False otherwise
        """
        self.g.E().has('id', memory_id).has('user_id', user_id).drop().iterate()
        logger.debug(f'Deleted memory: {memory_id}')
        return True

    @retry_on_connection_error
    def delete_entity(self, entity_id: str, user_id: str) -> bool:
        """
        Delete a specific entity vertex and all its connected memories.

        Args:
            entity_id: Entity ID to delete
            user_id: User ID for security check

        Returns:
            True if deletion was successful, False otherwise
        """
        # Delete all connected memory edges first
        self.g.V().has('id', entity_id).has('user_id', user_id)\
            .both_e().has('user_id', user_id).drop().iterate()

        # Then delete the vertex itself
        self.g.V().has('id', entity_id).has('user_id', user_id).drop().iterate()

        logger.debug(f'Deleted entity and connected memories: {entity_id}')
        return True

    @retry_on_connection_error
    def cleanup_expired_memories(self, user_id: Optional[str] = None) -> int:
        """
        Remove expired memories from the graph.

        Args:
            user_id: Optional user ID to limit cleanup to specific user

        Returns:
            Number of memories deleted
        """
        current_time = to_seconds_str()

        # Build traversal to find expired memories
        traversal = self.g.E()
        if user_id:
            traversal = traversal.has('user_id', user_id)

        # Get count first
        count = traversal.has('expires_at').where(__.values('expires_at').is_(P.lt(current_time))).count().next()

        if count > 0:
            # Delete expired memories
            traversal = self.g.E()
            if user_id:
                traversal = traversal.has('user_id', user_id)

            traversal.has('expires_at').where(__.values('expires_at').is_(P.lt(current_time))).drop().iterate()
            logger.info(f'Cleaned up {count} expired memories')

        return int(count)

    @retry_on_connection_error
    def get_related_memories(self, user_id: str, entity_name: str, hops: int = 1) -> List[Memory]:
        """
        Get memory objects within specified hops from entities with given name.

        Args:
            user_id: User ID to filter by
            entity_name: Entity name to search for
            hops: Number of hops to traverse (default: 1)

        Returns:
            List of Memory objects
        """
        memory_data = self.g.V().has('name', entity_name).has('user_id', user_id)\
            .repeat(__.both_e().has('user_id', user_id).other_v().has('user_id', user_id))\
            .times(hops)\
            .path()\
            .unfold()\
            .has_label('Memory')\
            .has('user_id', user_id)\
            .dedup()\
            .value_map(True).to_list()

        memories = []
        seen_ids = set()
        for data in memory_data:
            memory_id = data.get('id', [''])[0] if isinstance(data.get('id'), list) else data.get('id', '')
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                memories.append(
                    Memory(
                        id=memory_id,
                        user_id=data.get('user_id', [''])[0] if isinstance(data.get('user_id'), list) else data.get(
                            'user_id', ''),
                        statement=data.get('statement', [''])[0] if isinstance(data.get('statement'), list) else data.get(
                            'statement', ''),
                        subject_id=data.get('subject_id', [''])[0] if isinstance(data.get('subject_id'), list) else data.get(
                            'subject_id', ''),
                        subject_name=data.get('subject_name', [''])[0] if isinstance(data.get('subject_name'),
                                                                                     list) else data.get('subject_name', ''),
                        predicate_name=data.get('predicate_name', [''])[0] if isinstance(
                            data.get('predicate_name'), list) else data.get('predicate_name', ''),
                        object_id=data.get('object_id', [''])[0] if isinstance(data.get('object_id'), list) else data.get(
                            'object_id', ''),
                        object_name=data.get('object_name', [''])[0] if isinstance(data.get('object_name'), list) else data.get(
                            'object_name', ''),
                        confidence=float(
                            data.get('confidence', [1.0])[0] if isinstance(data.get('confidence'), list) else data.
                            get('confidence', 1.0)),
                        source_message=data.get('source_message', [''])[0] if isinstance(
                            data.get('source_message'), list) else data.get('source_message', ''),
                        created_at=datetime.fromtimestamp(
                            int(
                                data.get('created_at', [0])[0] if isinstance(data.get('created_at'), list) else data.
                                get('created_at', 0))) if data.get('created_at') else datetime.fromtimestamp(0),
                        expires_at=datetime.fromtimestamp(
                            int(
                                data.get('expires_at', [4102444800])[0] if isinstance(data.get('expires_at'), list) else data.
                                get('expires_at', 4102444800)))
                        if data.get('expires_at') else datetime.fromtimestamp(4102444800)))

        logger.debug(f"Found {len(memories)} memory objects within {hops} hops of entity '{entity_name}'")
        return memories

    @retry_on_connection_error
    def cleanup(self) -> bool:
        """
        Clean up all data from Neptune (vertices and edges).

        Returns:
            True if cleanup was successful, False otherwise
        """
        # Delete all edges first
        logger.info('Deleting all edges from Neptune...')
        self.g.E().drop().iterate()
        logger.info('✓ All edges deleted')

        # Delete all vertices
        logger.info('Deleting all vertices from Neptune...')
        self.g.V().drop().iterate()
        logger.info('✓ All vertices deleted')

        return True

    @retry_on_connection_error
    def health_check(self) -> bool:
        """
        Perform a health check on the Neptune service.

        Returns:
            True if service is healthy, False otherwise
        """
        # Simple query to test connectivity
        self.g.V().limit(1).count().next()
        return True

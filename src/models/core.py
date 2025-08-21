"""
Core data models for the long-term memory system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Entity:
    """Represents an entity extracted from user conversations."""
    id: str  # Unique identifier within user's graph
    user_id: str  # Each entity belongs to a specific user's graph
    name: str
    type: str  # Entity type (person, place, concept, etc.)
    embedding: List[float]  # For external vector search
    created_at: datetime


@dataclass
class Memory:
    """Represents a subject-predicate-object triplet within a user's knowledge graph.

    Each triplet exists within a specific user's graph, ensuring complete isolation
    between users while maintaining the semantic structure.
    """
    id: str
    user_id: str  # Triplet belongs to specific user's graph
    statement: str  # Natural language statement (subject-predicate-object)
    subject_id: str  # Reference to Entity ID within same user's graph
    subject_name: str  # Reference to Entity ID within same user's graph
    predicate_name: str  # Reference to Relationship ID within same user's graph
    object_id: str  # Reference to Entity ID within same user's graph
    object_name: str  # Reference to Entity ID within same user's graph
    confidence: float
    source_message: str
    created_at: datetime
    expires_at: datetime

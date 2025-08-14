"""UUID generation utilities."""

import hashlib
import uuid
from typing import Optional


def to_uuid_from_string(value: str, namespace: Optional[uuid.UUID] = None) -> str:
    """Generate deterministic UUID from string.
    
    Args:
        value: Input string
        namespace: Optional UUID namespace
        
    Returns:
        UUID string
    """
    if namespace:
        # Use UUID5 with namespace
        return str(uuid.uuid5(namespace, value))
    else:
        # Use MD5 hash-based UUID for backwards compatibility
        hexdigest = hashlib.md5(value.encode()).hexdigest()
        return str(uuid.UUID(hexdigest))


def generate_uuid() -> str:
    """Generate a random UUID.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if string is a valid UUID.
    
    Args:
        uuid_string: String to check
        
    Returns:
        True if valid UUID
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, AttributeError):
        return False


def short_uuid(full_uuid: str, length: int = 8) -> str:
    """Get shortened version of UUID.
    
    Args:
        full_uuid: Full UUID string
        length: Desired length
        
    Returns:
        Shortened UUID
    """
    # Remove hyphens and take first N characters
    clean_uuid = full_uuid.replace("-", "")
    return clean_uuid[:length]
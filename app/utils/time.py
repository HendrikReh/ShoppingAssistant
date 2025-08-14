"""Time and date utilities."""

from datetime import datetime


def format_seconds(seconds: float) -> str:
    """Format seconds to human-readable string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 1:
        return f"{seconds:.2f}s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs >= 1:
        parts.append(f"{int(secs)}s")
    elif not parts:  # Less than 1 second and no other parts
        parts.append(f"{secs:.2f}s")
    
    return " ".join(parts)


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp as string.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def parse_timestamp(timestamp_str: str, format_str: str = "%Y%m%d_%H%M%S") -> datetime:
    """Parse timestamp string to datetime.
    
    Args:
        timestamp_str: Timestamp string
        format_str: Expected format
        
    Returns:
        datetime object
    """
    return datetime.strptime(timestamp_str, format_str)


def time_since(start_time: datetime) -> str:
    """Get human-readable time since start_time.
    
    Args:
        start_time: Start time
        
    Returns:
        Human-readable duration
    """
    delta = datetime.now() - start_time
    return format_seconds(delta.total_seconds())


def format_duration(start: float, end: float) -> str:
    """Format duration between two timestamps.
    
    Args:
        start: Start timestamp (from time.time())
        end: End timestamp (from time.time())
        
    Returns:
        Formatted duration
    """
    duration = end - start
    return format_seconds(duration)
"""I/O and file utilities."""

from pathlib import Path
from typing import Iterable, List, Tuple


def ensure_dirs() -> Tuple[Path, Path]:
    """Ensure evaluation directories exist.
    
    Returns:
        Tuple of (datasets_dir, results_dir)
    """
    datasets_dir = Path("eval/datasets")
    results_dir = Path("eval/results")
    
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return datasets_dir, results_dir


def ensure_path(path: Path) -> Path:
    """Ensure path's parent directories exist.
    
    Args:
        path: Path to ensure
        
    Returns:
        The path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def chunked(items: Iterable, n: int) -> Iterable[List]:
    """Yield successive n-sized chunks from items.
    
    Args:
        items: Iterable to chunk
        n: Chunk size
        
    Yields:
        Chunks of size n
    """
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def safe_path(base_path: Path, filename: str) -> Path:
    """Create safe path by sanitizing filename.
    
    Args:
        base_path: Base directory
        filename: Filename to sanitize
        
    Returns:
        Safe path
    """
    # Remove problematic characters
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    safe_name = safe_name.strip()
    
    if not safe_name:
        safe_name = "unnamed"
    
    return base_path / safe_name


def file_size_str(size_bytes: int) -> str:
    """Format file size as human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
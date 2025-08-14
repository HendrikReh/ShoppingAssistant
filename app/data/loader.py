"""Data loading utilities."""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> List[Dict]:
    """Read JSONL file and return list of dictionaries.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries
        
    Raises:
        ValueError: If invalid JSON is encountered
    """
    rows: List[Dict] = []
    with path.open("r") as fp:
        for idx, line in enumerate(fp, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {e}") from e
    return rows


def load_jsonl(path: Path, max_samples: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Load JSONL file with optional sampling.
    
    Args:
        path: Path to JSONL file
        max_samples: Maximum number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of dictionaries
    """
    data = read_jsonl(path)
    
    if max_samples and len(data) > max_samples:
        random.seed(seed)
        data = random.sample(data, max_samples)
    
    return data


def save_json(path: Path, data: Any) -> None:
    """Save data to JSON file.
    
    Args:
        path: Path to save file
        data: Data to save
    """
    with path.open("w") as fp:
        json.dump(data, fp, indent=2)


def write_jsonl(path: Path, data: List[Dict]) -> None:
    """Write list of dictionaries to JSONL file.
    
    Args:
        path: Path to save file
        data: List of dictionaries to write
    """
    with path.open("w") as fp:
        for item in data:
            fp.write(json.dumps(item) + "\n")


def load_json(path: Path) -> Any:
    """Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with path.open("r") as fp:
        return json.load(fp)


def ensure_path(path: Path) -> Path:
    """Ensure path exists, creating parent directories if needed.
    
    Args:
        path: Path to ensure
        
    Returns:
        The path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
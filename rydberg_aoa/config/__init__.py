"""
Configuration management module.

Provides centralized loading and validation of simulation parameters.
Supports YAML for dynamic, external configuration files.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

def load_yaml_config(config_path: str | Path, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, merging with defaults if provided.
    
    Args:
        config_path: Path to YAML config file.
        defaults: Optional default config dict.
    
    Returns:
        Loaded configuration dict.
    
    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        if defaults:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return defaults
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if yaml is None:
        raise ImportError("YAML support requires pyyaml: pip install pyyaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        
        if defaults:
            # Simple deep merge defaults with config overrides
            def deep_merge(d1: Dict, d2: Dict) -> Dict:
                for k, v in d2.items():
                    if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
                        deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                return d1
            deep_merge(defaults, config)
            config = defaults
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise

# Example usage: load_yaml_config('config.yaml')

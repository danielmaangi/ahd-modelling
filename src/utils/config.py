"""
HIV Disease Prediction - Configuration Utilities

This module provides utilities for loading and managing configuration
files for the HIV disease prediction system.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config or {}
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return {}


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'api.host')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if config:
            merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_environment_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Configuration dictionary from environment variables
    """
    env_config = {}
    
    # API configuration
    if os.getenv('API_HOST'):
        env_config.setdefault('api', {})['host'] = os.getenv('API_HOST')
    
    if os.getenv('API_PORT'):
        env_config.setdefault('api', {})['port'] = int(os.getenv('API_PORT'))
    
    if os.getenv('API_WORKERS'):
        env_config.setdefault('api', {})['workers'] = int(os.getenv('API_WORKERS'))
    
    # Model configuration
    if os.getenv('MODEL_PATH'):
        env_config.setdefault('model', {})['path'] = os.getenv('MODEL_PATH')
    
    # Logging configuration
    if os.getenv('LOG_LEVEL'):
        env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
    
    # Database configuration
    if os.getenv('DATABASE_URL'):
        env_config.setdefault('database', {})['url'] = os.getenv('DATABASE_URL')
    
    return env_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_sections = ['api', 'model']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate API configuration
    api_config = config.get('api', {})
    if 'host' not in api_config:
        logger.error("Missing API host configuration")
        return False
    
    if 'port' not in api_config:
        logger.error("Missing API port configuration")
        return False
    
    # Validate model configuration
    model_config = config.get('model', {})
    if 'path' not in model_config:
        logger.error("Missing model path configuration")
        return False
    
    return True

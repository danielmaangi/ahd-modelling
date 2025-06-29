"""
HIV Disease Prediction - Logging Configuration Utilities

This module provides utilities for setting up and configuring logging
for the HIV disease prediction system.
"""

import logging
import logging.config
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def setup_logging(config_path: Optional[str] = None, 
                 log_level: Optional[str] = None,
                 log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to logging configuration file
        log_level: Override log level
        log_file: Override log file path
    """
    # Default configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': 'logs/app.log',
                'maxBytes': 52428800,  # 50MB
                'backupCount': 10,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'fastapi': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }
    
    # Load configuration from file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Use loaded config
            logging_config = config
            
        except Exception as e:
            print(f"Failed to load logging config from {config_path}: {e}")
            logging_config = default_config
    else:
        logging_config = default_config
    
    # Override log level if provided
    if log_level:
        log_level = log_level.upper()
        if 'loggers' in logging_config:
            for logger_config in logging_config['loggers'].values():
                logger_config['level'] = log_level
        if 'handlers' in logging_config:
            for handler_config in logging_config['handlers'].values():
                handler_config['level'] = log_level
    
    # Override log file if provided
    if log_file:
        if 'handlers' in logging_config and 'file' in logging_config['handlers']:
            logging_config['handlers']['file']['filename'] = log_file
    
    # Create log directory if it doesn't exist
    if 'handlers' in logging_config:
        for handler_config in logging_config['handlers'].values():
            if 'filename' in handler_config:
                log_path = Path(handler_config['filename'])
                log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply logging configuration
    try:
        logging.config.dictConfig(logging_config)
        print("Logging configuration applied successfully")
    except Exception as e:
        print(f"Failed to apply logging configuration: {e}")
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/app.log')
            ]
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_uvicorn_logging() -> Dict[str, Any]:
    """
    Configure logging for Uvicorn server.
    
    Returns:
        Uvicorn logging configuration
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }


def setup_structured_logging(service_name: str = "hiv-prediction-api",
                           environment: str = "development") -> None:
    """
    Setup structured logging with additional context.
    
    Args:
        service_name: Name of the service
        environment: Environment (development, staging, production)
    """
    import logging
    
    class StructuredFormatter(logging.Formatter):
        """Custom formatter for structured logging."""
        
        def format(self, record):
            log_entry = {
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'service': service_name,
                'environment': environment,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated', 
                              'thread', 'threadName', 'processName', 'process',
                              'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    log_entry[key] = value
            
            return json.dumps(log_entry)
    
    # Configure structured logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    
    # Get root logger and configure
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def add_request_id_filter():
    """Add request ID filter for tracing requests."""
    import uuid
    from contextvars import ContextVar
    
    request_id_var: ContextVar[str] = ContextVar('request_id', default='')
    
    class RequestIDFilter(logging.Filter):
        """Filter to add request ID to log records."""
        
        def filter(self, record):
            record.request_id = request_id_var.get('')
            return True
    
    # Add filter to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIDFilter())
    
    return request_id_var


def configure_performance_logging():
    """Configure performance logging for monitoring."""
    performance_logger = logging.getLogger('performance')
    
    # Create performance log handler
    handler = logging.FileHandler('logs/performance.log')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    performance_logger.addHandler(handler)
    performance_logger.setLevel(logging.INFO)
    
    return performance_logger


def log_function_performance(func):
    """Decorator to log function performance."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger('performance')
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed successfully in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def setup_security_logging():
    """Setup security event logging."""
    security_logger = logging.getLogger('security')
    
    # Create security log handler
    handler = logging.FileHandler('logs/security.log')
    formatter = logging.Formatter(
        '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    security_logger.addHandler(handler)
    security_logger.setLevel(logging.WARNING)
    
    return security_logger


def log_security_event(event_type: str, details: Dict[str, Any], 
                      severity: str = 'WARNING'):
    """
    Log security events.
    
    Args:
        event_type: Type of security event
        details: Event details
        severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
    """
    security_logger = logging.getLogger('security')
    
    log_message = f"{event_type}: {json.dumps(details)}"
    
    if severity.upper() == 'INFO':
        security_logger.info(log_message)
    elif severity.upper() == 'WARNING':
        security_logger.warning(log_message)
    elif severity.upper() == 'ERROR':
        security_logger.error(log_message)
    elif severity.upper() == 'CRITICAL':
        security_logger.critical(log_message)


def configure_audit_logging():
    """Configure audit logging for compliance."""
    audit_logger = logging.getLogger('audit')
    
    # Create audit log handler with strict formatting
    handler = logging.FileHandler('logs/audit.log')
    formatter = logging.Formatter(
        '%(asctime)s - AUDIT - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    
    return audit_logger


def log_audit_event(action: str, user_id: str, resource: str, 
                   outcome: str, details: Optional[Dict[str, Any]] = None):
    """
    Log audit events for compliance.
    
    Args:
        action: Action performed
        user_id: User identifier
        resource: Resource accessed
        outcome: Action outcome (SUCCESS, FAILURE)
        details: Additional details
    """
    audit_logger = logging.getLogger('audit')
    
    audit_entry = {
        'action': action,
        'user_id': user_id,
        'resource': resource,
        'outcome': outcome,
        'timestamp': logging.Formatter().formatTime(logging.LogRecord(
            '', 0, '', 0, '', (), None
        ))
    }
    
    if details:
        audit_entry['details'] = details
    
    audit_logger.info(json.dumps(audit_entry))


# Environment-specific logging configurations
def get_production_logging_config():
    """Get production logging configuration."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'json',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': '/var/log/hiv-prediction/app.log',
                'maxBytes': 104857600,  # 100MB
                'backupCount': 20,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }


def get_development_logging_config():
    """Get development logging configuration."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'root': {
                'level': 'DEBUG',
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

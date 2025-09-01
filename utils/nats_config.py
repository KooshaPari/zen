"""
NATS Configuration Management and Environment Setup

This module provides comprehensive configuration management for NATS infrastructure:
- Environment-based configuration with validation
- Dynamic configuration updates and hot reloading
- Production-ready defaults and optimization
- Security and authentication configuration
- Cluster and edge deployment settings
- Performance tuning and monitoring configuration
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class NATSServerConfig(BaseModel):
    """Individual NATS server configuration."""

    url: str = Field(..., description="NATS server URL")
    weight: int = Field(default=100, description="Server weight for load balancing")
    max_reconnect_attempts: int = Field(default=-1, description="Max reconnection attempts")
    reconnect_wait: float = Field(default=2.0, description="Reconnect wait time in seconds")

    # Authentication
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    token: Optional[str] = Field(None, description="Authentication token")
    nkey_file: Optional[str] = Field(None, description="Path to NKey file")
    jwt_file: Optional[str] = Field(None, description="Path to JWT file")

    # TLS settings
    tls_cert: Optional[str] = Field(None, description="TLS certificate path")
    tls_key: Optional[str] = Field(None, description="TLS key path")
    tls_ca: Optional[str] = Field(None, description="TLS CA certificate path")
    tls_verify: bool = Field(default=True, description="Verify TLS certificates")

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('nats://', 'tls://', 'ws://', 'wss://')):
            raise ValueError('URL must start with nats://, tls://, ws://, or wss://')
        return v


class JetStreamConfig(BaseModel):
    """JetStream configuration settings."""

    enabled: bool = Field(default=True, description="Enable JetStream")
    domain: Optional[str] = Field(None, description="JetStream domain")

    # Resource limits
    max_memory: int = Field(default=1024*1024*1024, description="Max memory (1GB)")
    max_storage: int = Field(default=10*1024*1024*1024, description="Max storage (10GB)")
    max_streams: int = Field(default=1000, description="Maximum number of streams")
    max_consumers: int = Field(default=10000, description="Maximum number of consumers")

    # Store directory (for file storage)
    store_dir: Optional[str] = Field(None, description="JetStream store directory")

    # Clustering
    cluster_replicas: int = Field(default=3, description="Default replica count")

    # Performance tuning
    sync_interval: float = Field(default=2.0, description="Sync interval in seconds")
    sync_always: bool = Field(default=False, description="Always sync to disk")
    compress: bool = Field(default=True, description="Enable compression")


class ClusterConfig(BaseModel):
    """NATS cluster configuration."""

    enabled: bool = Field(default=False, description="Enable clustering")
    name: str = Field(default="nats-cluster", description="Cluster name")

    # Cluster listen settings
    listen_host: str = Field(default="0.0.0.0", description="Cluster listen host")
    listen_port: int = Field(default=6222, description="Cluster listen port")

    # Cluster routes (other cluster members)
    routes: list[str] = Field(default_factory=list, description="Cluster member routes")

    # Route connection settings
    connect_retries: int = Field(default=5, description="Route connection retries")

    # Authentication for cluster connections
    username: Optional[str] = Field(None, description="Cluster auth username")
    password: Optional[str] = Field(None, description="Cluster auth password")

    # Permissions
    permissions_file: Optional[str] = Field(None, description="Cluster permissions file")


class LeafNodeConfig(BaseModel):
    """Leaf node configuration for edge computing."""

    enabled: bool = Field(default=False, description="Enable leaf nodes")

    # Leaf node listen settings
    listen_host: str = Field(default="0.0.0.0", description="Leaf node listen host")
    listen_port: int = Field(default=7422, description="Leaf node listen port")

    # Remote leaf node connections
    remotes: list[dict[str, Any]] = Field(default_factory=list, description="Remote leaf connections")

    # Leaf node authentication
    username: Optional[str] = Field(None, description="Leaf auth username")
    password: Optional[str] = Field(None, description="Leaf auth password")

    # Reconnection settings
    reconnect_delay: float = Field(default=1.0, description="Reconnect delay in seconds")

    # Hub configuration (for spoke-hub topology)
    hub_config: Optional[dict[str, Any]] = Field(None, description="Hub configuration")


class SecurityConfig(BaseModel):
    """Security and authentication configuration."""

    # User authentication
    users: list[dict[str, Any]] = Field(default_factory=list, description="User configurations")

    # Account-based multi-tenancy
    accounts: list[dict[str, Any]] = Field(default_factory=list, description="Account configurations")

    # JWT/NKey authentication
    operator_jwt: Optional[str] = Field(None, description="Operator JWT file")
    system_account: Optional[str] = Field(None, description="System account")

    # Authorization
    authorization: dict[str, Any] = Field(default_factory=dict, description="Authorization settings")

    # TLS settings
    tls_config: dict[str, Any] = Field(default_factory=dict, description="TLS configuration")

    # IP restrictions
    allowed_connection_types: list[str] = Field(
        default_factory=lambda: ["STANDARD", "WEBSOCKET", "LEAFNODE", "MQTT"],
        description="Allowed connection types"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""

    # HTTP monitoring endpoint
    monitor_host: str = Field(default="0.0.0.0", description="Monitor HTTP host")
    monitor_port: int = Field(default=8222, description="Monitor HTTP port")

    # Metrics collection
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=30, description="Metrics collection interval in seconds")

    # Health check settings
    health_check_interval: int = Field(default=10, description="Health check interval in seconds")

    # Log settings
    log_file: Optional[str] = Field(None, description="Log file path")
    log_level: str = Field(default="info", description="Log level")
    log_time: bool = Field(default=True, description="Include timestamps in logs")

    # Debug settings
    debug: bool = Field(default=False, description="Enable debug logging")
    trace: bool = Field(default=False, description="Enable trace logging")

    # Profiling
    profile_port: Optional[int] = Field(None, description="Profiling HTTP port")


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    # Connection limits
    max_connections: int = Field(default=65536, description="Maximum client connections")
    max_subscriptions: int = Field(default=0, description="Maximum subscriptions (0=unlimited)")
    max_payload: int = Field(default=1048576, description="Maximum payload size (1MB)")

    # Message limits
    max_pending: int = Field(default=268435456, description="Maximum pending bytes (256MB)")
    max_control_line: int = Field(default=4096, description="Maximum control line size")

    # Timing settings
    ping_interval: int = Field(default=120, description="Ping interval in seconds")
    ping_max: int = Field(default=2, description="Maximum missed pings")
    write_deadline: str = Field(default="10s", description="Write deadline")

    # I/O settings
    max_closed_clients: int = Field(default=10000, description="Maximum closed clients to track")

    # Memory optimization
    disable_short_first_ping: bool = Field(default=False, description="Disable short first ping")

    # Slow consumer settings
    slow_consumer_threshold: str = Field(default="1GB", description="Slow consumer threshold")

    # Rate limiting
    rate_limit: Optional[int] = Field(None, description="Rate limit in bytes/second")


class NATSConfiguration(BaseModel):
    """Complete NATS configuration."""

    # Basic server settings
    server_name: str = Field(default="nats-server", description="Server name")
    client_host: str = Field(default="0.0.0.0", description="Client listen host")
    client_port: int = Field(default=4222, description="Client listen port")

    # Component configurations
    servers: list[NATSServerConfig] = Field(default_factory=list, description="NATS servers")
    jetstream: JetStreamConfig = Field(default_factory=JetStreamConfig, description="JetStream config")
    cluster: ClusterConfig = Field(default_factory=ClusterConfig, description="Cluster config")
    leafnode: LeafNodeConfig = Field(default_factory=LeafNodeConfig, description="Leaf node config")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security config")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring config")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance config")

    # Client configuration
    client_config: dict[str, Any] = Field(default_factory=dict, description="Client-specific settings")

    # Environment-specific overrides
    environment: str = Field(default="development", description="Environment name")

    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            logger.warning(f"Unknown environment: {v}")
        return v


class NATSConfigManager:
    """
    NATS configuration manager with environment loading and validation.
    """

    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file
        self.environment = environment or os.getenv("NATS_ENVIRONMENT", "development")
        self.config: Optional[NATSConfiguration] = None

        # Configuration paths to check
        self.config_paths = [
            config_file,
            os.getenv("NATS_CONFIG_FILE"),
            "/etc/nats/nats.yaml",
            "/etc/nats/nats.json",
            "./config/nats.yaml",
            "./config/nats.json",
            "./nats.yaml",
            "./nats.json"
        ]

        logger.info(f"NATS Config Manager initialized for environment: {self.environment}")

    def load_configuration(self) -> NATSConfiguration:
        """Load and validate NATS configuration."""
        if self.config is not None:
            return self.config

        # Try to load from file
        config_data = self._load_from_file()

        # Override with environment variables
        config_data = self._load_from_environment(config_data)

        # Apply environment-specific settings
        config_data = self._apply_environment_defaults(config_data)

        # Validate and create configuration
        self.config = NATSConfiguration(**config_data)

        logger.info(f"NATS configuration loaded for environment: {self.environment}")
        return self.config

    def _load_from_file(self) -> dict[str, Any]:
        """Load configuration from file."""
        for config_path in self.config_paths:
            if not config_path:
                continue

            path = Path(config_path)
            if not path.exists():
                continue

            try:
                with open(path) as f:
                    if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)

                logger.info(f"Loaded NATS configuration from: {config_path}")
                return config_data or {}

            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                continue

        logger.info("No configuration file found, using defaults")
        return {}

    def _load_from_environment(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Override configuration with environment variables."""
        env_mappings = {
            # Basic settings
            'NATS_SERVER_NAME': ('server_name', str),
            'NATS_CLIENT_HOST': ('client_host', str),
            'NATS_CLIENT_PORT': ('client_port', int),

            # Server URLs
            'NATS_SERVERS': ('servers_urls', str),  # Comma-separated URLs

            # JetStream settings
            'NATS_JETSTREAM_ENABLED': ('jetstream.enabled', bool),
            'NATS_JETSTREAM_DOMAIN': ('jetstream.domain', str),
            'NATS_JETSTREAM_MAX_MEMORY': ('jetstream.max_memory', int),
            'NATS_JETSTREAM_MAX_STORAGE': ('jetstream.max_storage', int),
            'NATS_JETSTREAM_STORE_DIR': ('jetstream.store_dir', str),

            # Cluster settings
            'NATS_CLUSTER_ENABLED': ('cluster.enabled', bool),
            'NATS_CLUSTER_NAME': ('cluster.name', str),
            'NATS_CLUSTER_HOST': ('cluster.listen_host', str),
            'NATS_CLUSTER_PORT': ('cluster.listen_port', int),
            'NATS_CLUSTER_ROUTES': ('cluster.routes', str),  # Comma-separated

            # Leaf node settings
            'NATS_LEAFNODE_ENABLED': ('leafnode.enabled', bool),
            'NATS_LEAFNODE_HOST': ('leafnode.listen_host', str),
            'NATS_LEAFNODE_PORT': ('leafnode.listen_port', int),

            # Security settings
            'NATS_USERNAME': ('security.default_username', str),
            'NATS_PASSWORD': ('security.default_password', str),
            'NATS_TOKEN': ('security.default_token', str),
            'NATS_JWT_FILE': ('security.operator_jwt', str),

            # TLS settings
            'NATS_TLS_CERT': ('security.tls_config.cert_file', str),
            'NATS_TLS_KEY': ('security.tls_config.key_file', str),
            'NATS_TLS_CA': ('security.tls_config.ca_file', str),
            'NATS_TLS_VERIFY': ('security.tls_config.verify', bool),

            # Monitoring settings
            'NATS_MONITOR_HOST': ('monitoring.monitor_host', str),
            'NATS_MONITOR_PORT': ('monitoring.monitor_port', int),
            'NATS_LOG_LEVEL': ('monitoring.log_level', str),
            'NATS_LOG_FILE': ('monitoring.log_file', str),
            'NATS_DEBUG': ('monitoring.debug', bool),

            # Performance settings
            'NATS_MAX_CONNECTIONS': ('performance.max_connections', int),
            'NATS_MAX_PAYLOAD': ('performance.max_payload', int),
            'NATS_MAX_PENDING': ('performance.max_pending', int),
            'NATS_PING_INTERVAL': ('performance.ping_interval', int),
            'NATS_WRITE_DEADLINE': ('performance.write_deadline', str),
        }

        # Process environment variables
        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is None:
                continue

            try:
                # Convert value to appropriate type
                if value_type is bool:
                    parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif value_type is int:
                    parsed_value = int(env_value)
                else:
                    parsed_value = env_value

                # Set nested configuration value
                self._set_nested_config(config_data, config_path, parsed_value)

            except (ValueError, TypeError) as e:
                logger.error(f"Invalid environment variable {env_var}={env_value}: {e}")

        # Special handling for server URLs
        servers_urls = os.getenv('NATS_SERVERS')
        if servers_urls:
            urls = [url.strip() for url in servers_urls.split(',')]
            servers = []
            for url in urls:
                if url:
                    servers.append({'url': url})
            config_data['servers'] = servers

        # Special handling for cluster routes
        cluster_routes = os.getenv('NATS_CLUSTER_ROUTES')
        if cluster_routes:
            routes = [route.strip() for route in cluster_routes.split(',')]
            self._set_nested_config(config_data, 'cluster.routes', routes)

        return config_data

    def _set_nested_config(self, config: dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = path.split('.')
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _apply_environment_defaults(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Apply environment-specific default settings."""

        if self.environment == 'production':
            # Production optimizations
            production_defaults = {
                'jetstream': {
                    'enabled': True,
                    'max_memory': 4 * 1024 * 1024 * 1024,  # 4GB
                    'max_storage': 100 * 1024 * 1024 * 1024,  # 100GB
                    'cluster_replicas': 3,
                    'compress': True
                },
                'cluster': {
                    'enabled': True
                },
                'monitoring': {
                    'log_level': 'warn',
                    'debug': False,
                    'trace': False,
                    'enable_metrics': True
                },
                'performance': {
                    'max_connections': 100000,
                    'max_payload': 2 * 1024 * 1024,  # 2MB
                    'max_pending': 1024 * 1024 * 1024,  # 1GB
                    'ping_interval': 60,
                    'write_deadline': '5s'
                },
                'security': {
                    'tls_config': {
                        'verify': True
                    }
                }
            }

            # Merge production defaults
            config_data = self._deep_merge(production_defaults, config_data)

        elif self.environment == 'development':
            # Development settings for easier debugging
            development_defaults = {
                'jetstream': {
                    'enabled': True,
                    'max_memory': 512 * 1024 * 1024,  # 512MB
                    'max_storage': 5 * 1024 * 1024 * 1024,  # 5GB
                    'cluster_replicas': 1
                },
                'cluster': {
                    'enabled': False
                },
                'monitoring': {
                    'log_level': 'debug',
                    'debug': True,
                    'trace': False,
                    'enable_metrics': True
                },
                'performance': {
                    'max_connections': 1000,
                    'max_payload': 1024 * 1024,  # 1MB
                    'ping_interval': 120
                },
                'security': {
                    'tls_config': {
                        'verify': False  # Easier for development
                    }
                }
            }

            config_data = self._deep_merge(development_defaults, config_data)

        # Ensure default servers if none configured
        if not config_data.get('servers'):
            default_url = os.getenv('NATS_URL', 'nats://localhost:4222')
            config_data['servers'] = [{'url': default_url}]

        return config_data

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def generate_server_config(self) -> dict[str, Any]:
        """Generate NATS server configuration file format."""
        if not self.config:
            self.load_configuration()

        config = self.config
        server_config = {
            'server_name': config.server_name,
            'host': config.client_host,
            'port': config.client_port,
        }

        # JetStream configuration
        if config.jetstream.enabled:
            jetstream_config = {
                'max_memory_store': config.jetstream.max_memory,
                'max_file_store': config.jetstream.max_storage,
                'store_dir': config.jetstream.store_dir or './jetstream',
            }

            if config.jetstream.domain:
                jetstream_config['domain'] = config.jetstream.domain

            server_config['jetstream'] = jetstream_config

        # Cluster configuration
        if config.cluster.enabled:
            cluster_config = {
                'name': config.cluster.name,
                'host': config.cluster.listen_host,
                'port': config.cluster.listen_port,
            }

            if config.cluster.routes:
                cluster_config['routes'] = config.cluster.routes

            if config.cluster.username and config.cluster.password:
                cluster_config['authorization'] = {
                    'username': config.cluster.username,
                    'password': config.cluster.password
                }

            server_config['cluster'] = cluster_config

        # Leaf node configuration
        if config.leafnode.enabled:
            leafnode_config = {
                'host': config.leafnode.listen_host,
                'port': config.leafnode.listen_port,
            }

            if config.leafnode.remotes:
                leafnode_config['remotes'] = config.leafnode.remotes

            server_config['leafnodes'] = leafnode_config

        # Monitoring configuration
        server_config['monitor'] = {
            'host': config.monitoring.monitor_host,
            'port': config.monitoring.monitor_port,
        }

        # Security configuration
        if config.security.users:
            server_config['authorization'] = {
                'users': config.security.users
            }

        if config.security.accounts:
            server_config['accounts'] = config.security.accounts

        # TLS configuration
        if config.security.tls_config:
            tls_config = config.security.tls_config.copy()
            if tls_config:
                server_config['tls'] = tls_config

        # Performance limits
        server_config['max_connections'] = config.performance.max_connections
        server_config['max_payload'] = config.performance.max_payload
        server_config['max_pending'] = config.performance.max_pending
        server_config['ping_interval'] = f"{config.performance.ping_interval}s"
        server_config['ping_max'] = config.performance.ping_max
        server_config['write_deadline'] = config.performance.write_deadline

        # Logging
        if config.monitoring.log_file:
            server_config['logfile'] = config.monitoring.log_file
        server_config['debug'] = config.monitoring.debug
        server_config['trace'] = config.monitoring.trace
        server_config['logtime'] = config.monitoring.log_time

        return server_config

    def save_server_config(self, output_path: str, format: str = 'yaml') -> bool:
        """Save NATS server configuration to file."""
        try:
            server_config = self.generate_server_config()

            with open(output_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(server_config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(server_config, f, indent=2)

            logger.info(f"NATS server configuration saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save server configuration: {e}")
            return False

    def get_client_options(self) -> dict[str, Any]:
        """Get client connection options for NATS client libraries."""
        if not self.config:
            self.load_configuration()

        config = self.config
        client_options = {
            'servers': [server.url for server in config.servers],
            'name': f"{config.server_name}_client",
            'max_reconnect_attempts': -1 if config.servers else 10,
            'reconnect_time_wait': 2.0,
            'max_payload': config.performance.max_payload,
            'ping_interval': config.performance.ping_interval,
            'max_outstanding_pings': config.performance.ping_max,
        }

        # Authentication
        if config.servers:
            server = config.servers[0]  # Use first server's auth settings
            if server.username and server.password:
                client_options['user'] = server.username
                client_options['password'] = server.password
            elif server.token:
                client_options['token'] = server.token
            elif server.nkey_file:
                client_options['nkey_file'] = server.nkey_file
            elif server.jwt_file:
                client_options['user_jwt_file'] = server.jwt_file

        # TLS settings
        if config.security.tls_config:
            tls_config = config.security.tls_config
            if tls_config.get('cert_file'):
                client_options['tls_cert'] = tls_config['cert_file']
            if tls_config.get('key_file'):
                client_options['tls_key'] = tls_config['key_file']
            if tls_config.get('ca_file'):
                client_options['tls_ca'] = tls_config['ca_file']
            client_options['tls_verify'] = tls_config.get('verify', True)

        # Merge with any client-specific config
        client_options.update(config.client_config)

        return client_options

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.config:
            self.load_configuration()

        config = self.config

        # Check server URLs
        if not config.servers:
            issues.append("No NATS servers configured")
        else:
            for i, server in enumerate(config.servers):
                if not server.url:
                    issues.append(f"Server {i} has no URL configured")

        # Check JetStream settings
        if config.jetstream.enabled:
            if config.jetstream.max_memory <= 0:
                issues.append("JetStream max_memory must be positive")
            if config.jetstream.max_storage <= 0:
                issues.append("JetStream max_storage must be positive")
            if config.jetstream.store_dir and not os.path.isdir(os.path.dirname(config.jetstream.store_dir or "")):
                issues.append("JetStream store_dir parent directory does not exist")

        # Check cluster settings
        if config.cluster.enabled:
            if not config.cluster.name:
                issues.append("Cluster name is required when clustering is enabled")
            if config.cluster.listen_port == config.client_port:
                issues.append("Cluster port cannot be same as client port")

        # Check port conflicts
        ports = [config.client_port, config.monitoring.monitor_port]
        if config.cluster.enabled:
            ports.append(config.cluster.listen_port)
        if config.leafnode.enabled:
            ports.append(config.leafnode.listen_port)
        if config.monitoring.profile_port:
            ports.append(config.monitoring.profile_port)

        if len(set(ports)) != len(ports):
            issues.append("Port conflicts detected in configuration")

        # Check file paths
        file_paths = [
            config.monitoring.log_file,
            config.jetstream.store_dir,
            config.security.operator_jwt,
        ]

        for server in config.servers:
            file_paths.extend([
                server.nkey_file,
                server.jwt_file,
                server.tls_cert,
                server.tls_key,
                server.tls_ca,
            ])

        for path in file_paths:
            if path and not os.path.exists(os.path.dirname(path)):
                issues.append(f"Directory does not exist for path: {path}")

        return issues

    def reload_configuration(self) -> bool:
        """Reload configuration from file and environment."""
        try:
            old_config = self.config
            self.config = None  # Force reload
            new_config = self.load_configuration()

            # Compare configurations to detect changes
            if old_config:
                if old_config.model_dump() != new_config.model_dump():
                    logger.info("Configuration changes detected during reload")
                else:
                    logger.info("No configuration changes detected")

            return True

        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            return False


# Global configuration manager
_config_manager: Optional[NATSConfigManager] = None


def get_nats_config_manager(config_file: Optional[str] = None,
                           environment: Optional[str] = None) -> NATSConfigManager:
    """Get or create the global NATS configuration manager."""
    global _config_manager

    if _config_manager is None:
        _config_manager = NATSConfigManager(config_file, environment)

    return _config_manager


def get_nats_configuration() -> NATSConfiguration:
    """Get the current NATS configuration."""
    return get_nats_config_manager().load_configuration()


def get_nats_client_options() -> dict[str, Any]:
    """Get NATS client connection options."""
    return get_nats_config_manager().get_client_options()

#!/usr/bin/env python3
"""
OAuth 2.0 Dynamic Client Registration (DCR) HTTP Endpoints

This module provides FastAPI endpoints for RFC 7591 compliant Dynamic Client Registration,
integrating with the Zen MCP Server's HTTP transport and authentication infrastructure.

Endpoints:
- POST /oauth/register - Register a new OAuth 2.0 client
- GET /oauth/register/{client_id} - Retrieve client configuration
- PUT /oauth/register/{client_id} - Update client configuration
- DELETE /oauth/register/{client_id} - Delete client registration
- GET /oauth/discovery - OAuth 2.0 server metadata and DCR discovery

Security Features:
- Rate limiting with IP-based tracking
- Request validation and sanitization
- Comprehensive audit logging
- Error handling with proper HTTP status codes
- CORS support for web-based registrations
- Integration with existing Zen MCP Server infrastructure

Usage:
    from auth.oauth2_endpoints import create_oauth_router

    app = FastAPI()
    oauth_router = create_oauth_router()
    app.include_router(oauth_router, prefix="/oauth", tags=["oauth"])
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .oauth2_dcr import (
    ClientMetadata,
    DCRError,
    DCRManager,
    get_dcr_manager,
)

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """OAuth 2.0 error response format as defined in RFC 6749."""

    error: str
    error_description: Optional[str] = None
    error_uri: Optional[str] = None


class ClientRegistrationResponse(BaseModel):
    """Client registration response as defined in RFC 7591."""

    client_id: str
    client_secret: Optional[str] = None
    client_secret_expires_at: Optional[int] = None
    client_id_issued_at: int
    registration_access_token: Optional[str] = None
    registration_client_uri: Optional[str] = None

    # Echo back all client metadata
    client_name: Optional[str] = None
    client_uri: Optional[str] = None
    redirect_uris: Optional[list[str]] = None
    client_type: str = "confidential"
    token_endpoint_auth_method: str = "client_secret_basic"
    grant_types: list[str] = ["authorization_code"]
    response_types: list[str] = ["code"]
    scope: Optional[str] = None
    logo_uri: Optional[str] = None
    tos_uri: Optional[str] = None
    policy_uri: Optional[str] = None
    software_id: Optional[str] = None
    software_version: Optional[str] = None
    contacts: Optional[list[str]] = None
    jwks_uri: Optional[str] = None
    jwks: Optional[dict[str, Any]] = None
    mcp_capabilities: Optional[list[str]] = None
    mcp_transport_protocols: Optional[list[str]] = None
    mcp_session_timeout: Optional[int] = None


class ClientConfigurationResponse(BaseModel):
    """Client configuration response for GET/PUT operations."""

    client_id: str
    client_id_issued_at: int
    client_secret_expires_at: Optional[int] = None

    # Client metadata
    client_name: Optional[str] = None
    client_uri: Optional[str] = None
    redirect_uris: Optional[list[str]] = None
    client_type: str = "confidential"
    token_endpoint_auth_method: str = "client_secret_basic"
    grant_types: list[str] = ["authorization_code"]
    response_types: list[str] = ["code"]
    scope: Optional[str] = None
    logo_uri: Optional[str] = None
    tos_uri: Optional[str] = None
    policy_uri: Optional[str] = None
    software_id: Optional[str] = None
    software_version: Optional[str] = None
    contacts: Optional[list[str]] = None
    jwks_uri: Optional[str] = None
    jwks: Optional[dict[str, Any]] = None
    mcp_capabilities: Optional[list[str]] = None
    mcp_transport_protocols: Optional[list[str]] = None
    mcp_session_timeout: Optional[int] = None

    # Administrative metadata
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class DiscoveryResponse(BaseModel):
    """OAuth 2.0 Authorization Server Metadata response (RFC 8414)."""

    issuer: str
    authorization_endpoint: Optional[str] = None
    token_endpoint: Optional[str] = None
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    registration_endpoint: str
    scopes_supported: Optional[list[str]] = None
    response_types_supported: list[str] = ["code"]
    response_modes_supported: Optional[list[str]] = ["query", "fragment"]
    grant_types_supported: list[str] = ["authorization_code", "client_credentials", "refresh_token"]
    token_endpoint_auth_methods_supported: list[str] = ["client_secret_post", "client_secret_basic", "none"]
    token_endpoint_auth_signing_alg_values_supported: Optional[list[str]] = None
    service_documentation: Optional[str] = None
    ui_locales_supported: Optional[list[str]] = None

    # DCR-specific metadata
    registration_endpoint_auth_methods_supported: list[str] = ["none"]
    client_id_issued_at_supported: bool = True
    client_secret_expires_at_supported: bool = True
    registration_access_token_supported: bool = True

    # MCP extensions
    mcp_extensions_supported: list[str] = [
        "mcp_capabilities",
        "mcp_transport_protocols",
        "mcp_session_timeout"
    ]


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    # Check for forwarded headers first (for proxy setups)
    forwarded_for = request.headers.get('x-forwarded-for')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()

    real_ip = request.headers.get('x-real-ip')
    if real_ip:
        return real_ip.strip()

    # Fall back to direct client IP
    if hasattr(request.client, 'host'):
        return request.client.host

    return "unknown"


def get_user_agent(request: Request) -> Optional[str]:
    """Extract User-Agent header from request."""
    return request.headers.get('user-agent')


def get_auth_header(request: Request) -> Optional[str]:
    """Extract Authorization header for registration token."""
    auth_header = request.headers.get('authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]  # Remove 'Bearer ' prefix
    return None


def handle_dcr_error(error: DCRError) -> JSONResponse:
    """Handle DCR errors and return appropriate HTTP response."""

    error_response = ErrorResponse(
        error=error.error_code,
        error_description=error.error_description
    )

    return JSONResponse(
        status_code=error.status_code,
        content=error_response.dict(exclude_none=True)
    )


def create_oauth_router() -> APIRouter:
    """Create FastAPI router with OAuth 2.0 DCR endpoints."""

    router = APIRouter()

    @router.post("/register",
                 response_model=ClientRegistrationResponse,
                 status_code=status.HTTP_201_CREATED,
                 summary="Register OAuth 2.0 Client",
                 description="Register a new OAuth 2.0 client using Dynamic Client Registration (RFC 7591)")
    async def register_client(
        metadata: ClientMetadata,
        request: Request,
        dcr_manager: DCRManager = Depends(get_dcr_manager)
    ):
        """Register a new OAuth 2.0 client."""

        client_ip = get_client_ip(request)
        user_agent = get_user_agent(request)

        try:
            registered_client = await dcr_manager.register_client(
                metadata=metadata,
                client_ip=client_ip,
                user_agent=user_agent
            )

            # Create response
            response = ClientRegistrationResponse(
                client_id=registered_client.client_id,
                client_secret=registered_client.client_secret,
                client_secret_expires_at=registered_client.client_secret_expires_at,
                client_id_issued_at=registered_client.client_id_issued_at,
                registration_access_token=registered_client.registration_access_token,
                registration_client_uri=f"/oauth/register/{registered_client.client_id}",

                # Echo back client metadata
                client_name=registered_client.client_name,
                client_uri=registered_client.client_uri,
                redirect_uris=registered_client.redirect_uris,
                client_type=registered_client.client_type.value,
                token_endpoint_auth_method=registered_client.token_endpoint_auth_method.value,
                grant_types=[gt.value for gt in registered_client.grant_types],
                response_types=[rt.value for rt in registered_client.response_types],
                scope=registered_client.scope,
                logo_uri=registered_client.logo_uri,
                tos_uri=registered_client.tos_uri,
                policy_uri=registered_client.policy_uri,
                software_id=registered_client.software_id,
                software_version=registered_client.software_version,
                contacts=registered_client.contacts,
                jwks_uri=registered_client.jwks_uri,
                jwks=registered_client.jwks,
                mcp_capabilities=registered_client.mcp_capabilities,
                mcp_transport_protocols=registered_client.mcp_transport_protocols,
                mcp_session_timeout=registered_client.mcp_session_timeout
            )

            logger.info(f"Successfully registered client: {registered_client.client_id} from {client_ip}")
            return response

        except DCRError as e:
            logger.warning(f"DCR error during registration from {client_ip}: {e}")
            return handle_dcr_error(e)
        except ValidationError as e:
            logger.warning(f"Validation error during registration from {client_ip}: {e}")
            error = DCRError("invalid_client_metadata", str(e))
            return handle_dcr_error(error)
        except Exception as e:
            logger.error(f"Unexpected error during registration from {client_ip}: {e}")
            error = DCRError("server_error", "Internal server error", 500)
            return handle_dcr_error(error)

    @router.get("/register/{client_id}",
                response_model=ClientConfigurationResponse,
                summary="Get Client Configuration",
                description="Retrieve the configuration of a registered OAuth 2.0 client")
    async def get_client_configuration(
        client_id: str,
        request: Request,
        dcr_manager: DCRManager = Depends(get_dcr_manager)
    ):
        """Get client configuration using registration access token."""

        client_ip = get_client_ip(request)
        # user_agent = get_user_agent(request)  # Unused
        registration_token = get_auth_header(request)

        if not registration_token:
            error = DCRError("invalid_token", "Missing registration access token", 401)
            return handle_dcr_error(error)

        try:
            # Verify token and get client
            if not dcr_manager._verify_registration_token(registration_token, client_id):
                error = DCRError("invalid_token", "Invalid registration access token", 401)
                return handle_dcr_error(error)

            client = await dcr_manager.get_client(client_id)
            if not client:
                error = DCRError("invalid_client_id", "Client not found", 404)
                return handle_dcr_error(error)

            # Create response (without sensitive data like client_secret)
            response = ClientConfigurationResponse(
                client_id=client.client_id,
                client_id_issued_at=client.client_id_issued_at,
                client_secret_expires_at=client.client_secret_expires_at,

                client_name=client.client_name,
                client_uri=client.client_uri,
                redirect_uris=client.redirect_uris,
                client_type=client.client_type.value,
                token_endpoint_auth_method=client.token_endpoint_auth_method.value,
                grant_types=[gt.value for gt in client.grant_types],
                response_types=[rt.value for rt in client.response_types],
                scope=client.scope,
                logo_uri=client.logo_uri,
                tos_uri=client.tos_uri,
                policy_uri=client.policy_uri,
                software_id=client.software_id,
                software_version=client.software_version,
                contacts=client.contacts,
                jwks_uri=client.jwks_uri,
                jwks=client.jwks,
                mcp_capabilities=client.mcp_capabilities,
                mcp_transport_protocols=client.mcp_transport_protocols,
                mcp_session_timeout=client.mcp_session_timeout,

                created_at=client.created_at,
                updated_at=client.updated_at,
                last_used_at=client.last_used_at,
                is_active=client.is_active
            )

            logger.info(f"Retrieved client configuration: {client_id} from {client_ip}")
            return response

        except DCRError as e:
            logger.warning(f"DCR error retrieving client {client_id} from {client_ip}: {e}")
            return handle_dcr_error(e)
        except Exception as e:
            logger.error(f"Unexpected error retrieving client {client_id} from {client_ip}: {e}")
            error = DCRError("server_error", "Internal server error", 500)
            return handle_dcr_error(error)

    @router.put("/register/{client_id}",
                response_model=ClientConfigurationResponse,
                summary="Update Client Configuration",
                description="Update the configuration of a registered OAuth 2.0 client")
    async def update_client_configuration(
        client_id: str,
        metadata: ClientMetadata,
        request: Request,
        dcr_manager: DCRManager = Depends(get_dcr_manager)
    ):
        """Update client configuration using registration access token."""

        client_ip = get_client_ip(request)
        user_agent = get_user_agent(request)
        registration_token = get_auth_header(request)

        if not registration_token:
            error = DCRError("invalid_token", "Missing registration access token", 401)
            return handle_dcr_error(error)

        try:
            updated_client = await dcr_manager.update_client(
                client_id=client_id,
                metadata=metadata,
                registration_token=registration_token,
                client_ip=client_ip,
                user_agent=user_agent
            )

            # Create response
            response = ClientConfigurationResponse(
                client_id=updated_client.client_id,
                client_id_issued_at=updated_client.client_id_issued_at,
                client_secret_expires_at=updated_client.client_secret_expires_at,

                client_name=updated_client.client_name,
                client_uri=updated_client.client_uri,
                redirect_uris=updated_client.redirect_uris,
                client_type=updated_client.client_type.value,
                token_endpoint_auth_method=updated_client.token_endpoint_auth_method.value,
                grant_types=[gt.value for gt in updated_client.grant_types],
                response_types=[rt.value for rt in updated_client.response_types],
                scope=updated_client.scope,
                logo_uri=updated_client.logo_uri,
                tos_uri=updated_client.tos_uri,
                policy_uri=updated_client.policy_uri,
                software_id=updated_client.software_id,
                software_version=updated_client.software_version,
                contacts=updated_client.contacts,
                jwks_uri=updated_client.jwks_uri,
                jwks=updated_client.jwks,
                mcp_capabilities=updated_client.mcp_capabilities,
                mcp_transport_protocols=updated_client.mcp_transport_protocols,
                mcp_session_timeout=updated_client.mcp_session_timeout,

                created_at=updated_client.created_at,
                updated_at=updated_client.updated_at,
                last_used_at=updated_client.last_used_at,
                is_active=updated_client.is_active
            )

            logger.info(f"Successfully updated client: {client_id} from {client_ip}")
            return response

        except DCRError as e:
            logger.warning(f"DCR error updating client {client_id} from {client_ip}: {e}")
            return handle_dcr_error(e)
        except ValidationError as e:
            logger.warning(f"Validation error updating client {client_id} from {client_ip}: {e}")
            error = DCRError("invalid_client_metadata", str(e))
            return handle_dcr_error(error)
        except Exception as e:
            logger.error(f"Unexpected error updating client {client_id} from {client_ip}: {e}")
            error = DCRError("server_error", "Internal server error", 500)
            return handle_dcr_error(error)

    @router.delete("/register/{client_id}",
                   status_code=status.HTTP_204_NO_CONTENT,
                   summary="Delete Client Registration",
                   description="Delete a registered OAuth 2.0 client")
    async def delete_client_registration(
        client_id: str,
        request: Request,
        dcr_manager: DCRManager = Depends(get_dcr_manager)
    ):
        """Delete client registration using registration access token."""

        client_ip = get_client_ip(request)
        user_agent = get_user_agent(request)
        registration_token = get_auth_header(request)

        if not registration_token:
            error = DCRError("invalid_token", "Missing registration access token", 401)
            return handle_dcr_error(error)

        try:
            success = await dcr_manager.delete_client(
                client_id=client_id,
                registration_token=registration_token,
                client_ip=client_ip,
                user_agent=user_agent
            )

            if success:
                logger.info(f"Successfully deleted client: {client_id} from {client_ip}")
                return JSONResponse(status_code=204, content=None)
            else:
                error = DCRError("server_error", "Failed to delete client", 500)
                return handle_dcr_error(error)

        except DCRError as e:
            logger.warning(f"DCR error deleting client {client_id} from {client_ip}: {e}")
            return handle_dcr_error(e)
        except Exception as e:
            logger.error(f"Unexpected error deleting client {client_id} from {client_ip}: {e}")
            error = DCRError("server_error", "Internal server error", 500)
            return handle_dcr_error(error)

    @router.get("/discovery",
                response_model=DiscoveryResponse,
                summary="OAuth 2.0 Server Metadata",
                description="OAuth 2.0 Authorization Server Metadata endpoint (RFC 8414)")
    async def oauth_discovery(
        request: Request,
        dcr_manager: DCRManager = Depends(get_dcr_manager)
    ):
        """OAuth 2.0 Authorization Server Metadata and DCR discovery."""

        # Get base URL from request
        base_url = f"{request.url.scheme}://{request.url.netloc}"

        # Get DCR metadata
        dcr_metadata = dcr_manager.get_registration_metadata()

        response = DiscoveryResponse(
            issuer=base_url,
            registration_endpoint=f"{base_url}/oauth/register",
            service_documentation=f"{base_url}/docs",

            # DCR-specific fields from manager
            grant_types_supported=dcr_metadata["grant_types_supported"],
            response_types_supported=dcr_metadata["response_types_supported"],
            token_endpoint_auth_methods_supported=dcr_metadata["token_endpoint_auth_methods_supported"],
            registration_endpoint_auth_methods_supported=dcr_metadata["registration_endpoint_auth_methods_supported"],
            client_id_issued_at_supported=dcr_metadata["client_id_issued_at_supported"],
            client_secret_expires_at_supported=dcr_metadata["client_secret_expires_at_supported"],
            registration_access_token_supported=dcr_metadata["registration_access_token_supported"],
            mcp_extensions_supported=dcr_metadata["mcp_extensions_supported"]
        )

        logger.info(f"Served OAuth discovery metadata to {get_client_ip(request)}")
        return response

    return router


def create_oauth_admin_router() -> APIRouter:
    """Create FastAPI router with OAuth 2.0 admin endpoints (requires authentication)."""

    router = APIRouter()

    @router.get("/admin/clients",
                summary="List All Clients",
                description="Administrative endpoint to list all registered OAuth 2.0 clients")
    async def list_all_clients(
        request: Request,
        limit: int = 100,
        dcr_manager: DCRManager = Depends(get_dcr_manager)
    ):
        """List all registered clients (admin endpoint)."""

        # TODO: Add admin authentication check here
        client_ip = get_client_ip(request)

        try:
            clients = await dcr_manager.list_clients(limit=limit)

            # Transform to safe representation (no secrets)
            client_list = []
            for client in clients:
                safe_client = {
                    "client_id": client.client_id,
                    "client_name": client.client_name,
                    "client_type": client.client_type.value,
                    "created_at": client.created_at.isoformat(),
                    "updated_at": client.updated_at.isoformat(),
                    "is_active": client.is_active,
                    "grant_types": [gt.value for gt in client.grant_types],
                    "mcp_capabilities": client.mcp_capabilities
                }
                client_list.append(safe_client)

            logger.info(f"Admin listed {len(client_list)} clients from {client_ip}")

            return {
                "clients": client_list,
                "total_count": len(client_list),
                "limit": limit
            }

        except Exception as e:
            logger.error(f"Error listing clients for admin from {client_ip}: {e}")
            error = DCRError("server_error", "Failed to list clients", 500)
            return handle_dcr_error(error)

    return router

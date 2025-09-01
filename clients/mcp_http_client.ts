#!/usr/bin/env node
/**
 * MCP Streamable HTTP TypeScript Client
 * 
 * This module provides a TypeScript/Node.js client for connecting to MCP servers
 * using the Streamable HTTP transport protocol (MCP spec 2025-03-26).
 * 
 * Features:
 * - Streamable HTTP transport client
 * - Cross-language compatibility (works with Python servers)
 * - JSON-RPC 2.0 protocol support
 * - Tool execution and resource access
 * - Session management with secure UUIDs
 * - TypeScript type definitions
 */

import fetch, { Response } from 'node-fetch';

// MCP Types and Interfaces
interface MCPServerInfo {
    name: string;
    version: string;
    protocolVersion: string;
    capabilities: Record<string, any>;
    url: string;
    sessionId?: string;
}

interface MCPTool {
    name: string;
    description: string;
    inputSchema: Record<string, any>;
}

interface MCPResource {
    uri: string;
    name: string;
    description: string;
    mimeType?: string;
}

interface MCPPrompt {
    name: string;
    description: string;
    arguments: Array<Record<string, any>>;
}

interface JSONRPCRequest {
    jsonrpc: string;
    method: string;
    params?: Record<string, any>;
    id?: number;
}

interface JSONRPCResponse {
    jsonrpc: string;
    result?: any;
    error?: {
        code: number;
        message: string;
    };
    id?: number;
}

/**
 * MCP Streamable HTTP Client for TypeScript/Node.js
 */
export class MCPStreamableHTTPClient {
    private serverUrl: string;
    private timeout: number;
    private sessionId?: string;
    private serverInfo?: MCPServerInfo;
    private requestIdCounter: number = 0;

    constructor(serverUrl: string, timeout: number = 30000) {
        this.serverUrl = serverUrl.replace(/\/$/, ''); // Remove trailing slash
        this.timeout = timeout;
    }

    /**
     * Connect to the MCP server and initialize session
     */
    async connect(): Promise<void> {
        try {
            await this.initialize();
            console.log(`✅ Connected to MCP server: ${this.serverUrl}`);
        } catch (error) {
            console.error(`❌ Failed to connect to MCP server: ${error}`);
            throw error;
        }
    }

    /**
     * Disconnect from the MCP server
     */
    async disconnect(): Promise<void> {
        this.sessionId = undefined;
        this.serverInfo = undefined;
        console.log('👋 Disconnected from MCP server');
    }

    /**
     * Get next request ID
     */
    private getRequestId(): number {
        return ++this.requestIdCounter;
    }

    /**
     * Send JSON-RPC request to MCP server
     */
    private async sendRequest(
        method: string, 
        params?: Record<string, any>,
        notification: boolean = false
    ): Promise<any> {
        // Build JSON-RPC request
        const request: JSONRPCRequest = {
            jsonrpc: '2.0',
            method: method
        };

        if (params) {
            request.params = params;
        }

        if (!notification) {
            request.id = this.getRequestId();
        }

        // Prepare headers
        const headers: Record<string, string> = {
            'Content-Type': 'application/json'
        };

        if (this.sessionId) {
            headers['Mcp-Session-Id'] = this.sessionId;
        }

        try {
            // Send request with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            const response: Response = await fetch(this.serverUrl, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(request),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            // Handle session ID from response headers
            const sessionIdHeader = response.headers.get('mcp-session-id');
            if (sessionIdHeader) {
                this.sessionId = sessionIdHeader;
            }

            // Parse response
            const data: JSONRPCResponse = await response.json();

            // Check for JSON-RPC error
            if (data.error) {
                throw new Error(`MCP error ${data.error.code}: ${data.error.message}`);
            }

            return notification ? undefined : data.result;

        } catch (error) {
            console.error(`❌ MCP request failed for ${method}:`, error);
            throw error;
        }
    }

    /**
     * Initialize MCP connection
     */
    private async initialize(): Promise<void> {
        const initParams = {
            protocolVersion: '2025-03-26',
            capabilities: {
                experimental: {
                    streaming: true
                }
            },
            clientInfo: {
                name: 'ZenMCP-TSClient',
                version: '1.0.0'
            }
        };

        const result = await this.sendRequest('initialize', initParams);

        // Store server information
        this.serverInfo = {
            name: result?.serverInfo?.name || 'Unknown',
            version: result?.serverInfo?.version || 'Unknown',
            protocolVersion: result?.protocolVersion || 'Unknown',
            capabilities: result?.capabilities || {},
            url: this.serverUrl,
            sessionId: this.sessionId
        };

        // Send initialized notification
        await this.sendRequest('notifications/initialized', undefined, true);
    }

    /**
     * Get server information
     */
    async getServerInfo(): Promise<MCPServerInfo> {
        if (!this.serverInfo) {
            throw new Error('Not connected to server');
        }
        return this.serverInfo;
    }

    /**
     * List available tools
     */
    async listTools(): Promise<MCPTool[]> {
        const result = await this.sendRequest('tools/list');
        
        return (result?.tools || []).map((tool: any) => ({
            name: tool.name,
            description: tool.description || '',
            inputSchema: tool.inputSchema || {}
        }));
    }

    /**
     * Call a tool on the MCP server
     */
    async callTool(toolName: string, args?: Record<string, any>): Promise<any> {
        const params: any = { name: toolName };
        if (args) {
            params.arguments = args;
        }

        const result = await this.sendRequest('tools/call', params);

        // Extract text content from result
        const content = result?.content || [];
        if (Array.isArray(content)) {
            const textContent = content
                .filter((item: any) => item.type === 'text')
                .map((item: any) => item.text || '');

            if (textContent.length === 1) {
                return textContent[0];
            } else if (textContent.length > 1) {
                return textContent.join('\n');
            }
        }

        return result;
    }

    /**
     * List available resources
     */
    async listResources(): Promise<MCPResource[]> {
        const result = await this.sendRequest('resources/list');
        
        return (result?.resources || []).map((resource: any) => ({
            uri: resource.uri,
            name: resource.name || '',
            description: resource.description || '',
            mimeType: resource.mimeType
        }));
    }

    /**
     * Read a resource from the MCP server
     */
    async readResource(uri: string): Promise<string> {
        const result = await this.sendRequest('resources/read', { uri });

        const contents = result?.contents || [];
        if (Array.isArray(contents)) {
            // Return first text content
            for (const content of contents) {
                if (!content.mimeType || 
                    ['text/plain', 'application/json'].includes(content.mimeType)) {
                    return content.text || '';
                }
            }
        }

        return String(result);
    }

    /**
     * List available prompts
     */
    async listPrompts(): Promise<MCPPrompt[]> {
        const result = await this.sendRequest('prompts/list');
        
        return (result?.prompts || []).map((prompt: any) => ({
            name: prompt.name,
            description: prompt.description || '',
            arguments: prompt.arguments || []
        }));
    }

    /**
     * Get a prompt from the MCP server
     */
    async getPrompt(promptName: string, args?: Record<string, any>): Promise<string> {
        const params: any = { name: promptName };
        if (args) {
            params.arguments = args;
        }

        const result = await this.sendRequest('prompts/get', params);

        // Extract text from messages
        const messages = result?.messages || [];
        if (Array.isArray(messages)) {
            const textParts = messages.map((message: any) => {
                const content = message.content;
                if (typeof content === 'object' && content?.type === 'text') {
                    return content.text || '';
                } else if (typeof content === 'string') {
                    return content;
                }
                return '';
            }).filter(Boolean);

            return textParts.join('\n');
        }

        return result?.description || String(result);
    }

    /**
     * Set server logging level
     */
    async setLoggingLevel(level: string): Promise<void> {
        await this.sendRequest('logging/setLevel', { level });
    }

    /**
     * Test connection to server
     */
    async testConnection(): Promise<boolean> {
        try {
            const response = await fetch(this.serverUrl, { method: 'GET' });
            return response.ok;
        } catch {
            return false;
        }
    }
}

/**
 * Demo function to show MCP client functionality
 */
async function demoMCPClient(): Promise<void> {
    const serverUrl = process.env.MCP_SERVER_URL || 'http://localhost:8080/mcp';
    
    console.log('🧘 Zen MCP TypeScript Client Demo');
    console.log('='.repeat(40));
    console.log(`🔌 Connecting to MCP server: ${serverUrl}`);

    const client = new MCPStreamableHTTPClient(serverUrl);

    try {
        await client.connect();

        // Get server info
        const serverInfo = await client.getServerInfo();
        console.log(`✅ Connected to: ${serverInfo.name} v${serverInfo.version}`);
        console.log(`📡 Protocol version: ${serverInfo.protocolVersion}`);
        console.log(`🔧 Capabilities: ${Object.keys(serverInfo.capabilities).join(', ')}`);

        // List and test tools
        console.log('\n🛠️ Available Tools:');
        const tools = await client.listTools();
        tools.slice(0, 5).forEach(tool => {
            console.log(`  • ${tool.name}: ${tool.description}`);
        });

        console.log('\n🔄 Testing Tools:');
        
        // Test echo tool
        try {
            const result = await client.callTool('echo', { text: 'Hello from TypeScript client!' });
            console.log(`  📢 echo: ${result}`);
        } catch (error) {
            console.log(`  ❌ echo failed: ${error}`);
        }

        // Test get_time tool
        try {
            const result = await client.callTool('get_time');
            console.log(`  🕐 get_time: ${result}`);
        } catch (error) {
            console.log(`  ❌ get_time failed: ${error}`);
        }

        // Test multiply tool
        try {
            const result = await client.callTool('multiply', { a: 12, b: 7 });
            console.log(`  🔢 multiply(12, 7): ${result}`);
        } catch (error) {
            console.log(`  ❌ multiply failed: ${error}`);
        }

        // List and read resources
        console.log('\n📚 Available Resources:');
        const resources = await client.listResources();
        resources.forEach(resource => {
            console.log(`  • ${resource.name}: ${resource.uri}`);
        });

        if (resources.length > 0) {
            console.log('\n📖 Reading Resource:');
            try {
                const content = await client.readResource(resources[0].uri);
                console.log(`  📄 ${resources[0].name}:\n${content.substring(0, 200)}...`);
            } catch (error) {
                console.log(`  ❌ Failed to read resource: ${error}`);
            }
        }

        // List and get prompts
        console.log('\n💬 Available Prompts:');
        const prompts = await client.listPrompts();
        prompts.forEach(prompt => {
            console.log(`  • ${prompt.name}: ${prompt.description}`);
        });

        if (prompts.length > 0) {
            console.log('\n🎯 Getting Prompt:');
            try {
                const content = await client.getPrompt(prompts[0].name, { topic: 'tools' });
                console.log(`  📝 ${prompts[0].name}:\n${content.substring(0, 200)}...`);
            } catch (error) {
                console.log(`  ❌ Failed to get prompt: ${error}`);
            }
        }

        console.log('\n✅ Demo completed successfully!');

    } catch (error) {
        console.log(`❌ Connection failed: ${error}`);
        console.log('💡 Make sure the MCP server is running: python server_mcp_http.py');
    } finally {
        await client.disconnect();
    }
}

/**
 * Interactive MCP client shell
 */
async function interactiveMCPClient(): Promise<void> {
    const readline = require('readline');
    
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const question = (prompt: string): Promise<string> => {
        return new Promise((resolve) => {
            rl.question(prompt, resolve);
        });
    };

    console.log('🧘 Zen MCP Interactive TypeScript Client');
    console.log('='.repeat(40));
    
    const serverUrl = await question('MCP Server URL (default: http://localhost:8080/mcp): ') ||
                     'http://localhost:8080/mcp';

    const client = new MCPStreamableHTTPClient(serverUrl);

    try {
        await client.connect();
        
        const serverInfo = await client.getServerInfo();
        console.log(`✅ Connected to: ${serverInfo.name} v${serverInfo.version}`);

        // Load available capabilities
        const tools = await client.listTools();
        const resources = await client.listResources();
        const prompts = await client.listPrompts();

        console.log('\n📊 Server Capabilities:');
        console.log(`  • Tools: ${tools.length}`);
        console.log(`  • Resources: ${resources.length}`);
        console.log(`  • Prompts: ${prompts.length}`);

        console.log('\n💡 Commands:');
        console.log("  • 'tools' - List available tools");
        console.log("  • 'call <tool> [args]' - Call a tool (JSON args)");
        console.log("  • 'resources' - List available resources");
        console.log("  • 'read <uri>' - Read a resource");
        console.log("  • 'prompts' - List available prompts");
        console.log("  • 'prompt <name> [args]' - Get a prompt (JSON args)");
        console.log("  • 'exit' - Exit client");

        while (true) {
            const command = await question('\nmcp> ');
            
            if (command.trim() === 'exit') {
                break;
            }

            const parts = command.trim().split(' ');
            const cmd = parts[0];

            try {
                if (cmd === 'tools') {
                    tools.forEach(tool => {
                        console.log(`  • ${tool.name}: ${tool.description}`);
                    });
                } else if (cmd === 'call' && parts.length >= 2) {
                    const toolName = parts[1];
                    const argsStr = parts.slice(2).join(' ');
                    let args: any = undefined;
                    
                    if (argsStr) {
                        try {
                            args = JSON.parse(argsStr);
                        } catch {
                            console.log('❌ Invalid JSON arguments');
                            continue;
                        }
                    }
                    
                    const result = await client.callTool(toolName, args);
                    console.log(`Result: ${result}`);
                } else if (cmd === 'resources') {
                    resources.forEach(resource => {
                        console.log(`  • ${resource.name}: ${resource.uri}`);
                    });
                } else if (cmd === 'read' && parts.length >= 2) {
                    const uri = parts.slice(1).join(' ');
                    const content = await client.readResource(uri);
                    console.log(`Content:\n${content}`);
                } else if (cmd === 'prompts') {
                    prompts.forEach(prompt => {
                        console.log(`  • ${prompt.name}: ${prompt.description}`);
                    });
                } else if (cmd === 'prompt' && parts.length >= 2) {
                    const promptName = parts[1];
                    const argsStr = parts.slice(2).join(' ');
                    let args: any = undefined;
                    
                    if (argsStr) {
                        try {
                            args = JSON.parse(argsStr);
                        } catch {
                            console.log('❌ Invalid JSON arguments');
                            continue;
                        }
                    }
                    
                    const content = await client.getPrompt(promptName, args);
                    console.log(`Prompt:\n${content}`);
                } else if (command.trim()) {
                    console.log(`❌ Unknown command: ${cmd}`);
                }
            } catch (error) {
                console.log(`❌ Error: ${error}`);
            }
        }

        console.log('\n👋 Goodbye!');

    } catch (error) {
        console.log(`❌ Connection failed: ${error}`);
    } finally {
        await client.disconnect();
        rl.close();
    }
}

// Main execution
if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.includes('--demo')) {
        demoMCPClient().catch(console.error);
    } else if (args.includes('--interactive') || args.includes('-i')) {
        interactiveMCPClient().catch(console.error);
    } else {
        console.log('Zen MCP TypeScript Client');
        console.log('Usage: node mcp_http_client.js [--demo | --interactive]');
        console.log('  --demo        Run demonstration');
        console.log('  --interactive Interactive shell');
    }
}
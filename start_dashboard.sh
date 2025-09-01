#!/bin/bash

# Zen MCP Adaptive Learning Dashboard Launcher
# This script starts the performance monitoring dashboard

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings
HOST="0.0.0.0"
PORT=8080

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--host HOST] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --host HOST    Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT    Port to listen on (default: 8080)"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d ".zen_venv" ] && [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Please run ./run-server.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
if [ -d ".zen_venv" ]; then
    source .zen_venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if required packages are installed
echo -e "${BLUE}Checking dependencies...${NC}"
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
    pip install fastapi uvicorn[standard] python-multipart
fi

# Clear the screen
clear

# Display startup banner
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     🚀 Zen MCP Adaptive Learning Dashboard 🚀             ║"
echo "║                                                            ║"
echo "║     Performance Monitoring & Optimization System          ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Starting dashboard server...${NC}"
echo -e "Host: ${GREEN}$HOST${NC}"
echo -e "Port: ${GREEN}$PORT${NC}"
echo ""
echo -e "${YELLOW}Dashboard will be available at:${NC}"
echo -e "  Local:    ${GREEN}http://localhost:$PORT${NC}"

# Get local IP address
if command -v ifconfig &> /dev/null; then
    LOCAL_IP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
    if [ ! -z "$LOCAL_IP" ]; then
        echo -e "  Network:  ${GREEN}http://$LOCAL_IP:$PORT${NC}"
    fi
fi

echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Start the dashboard
python dashboard/performance_dashboard.py --host "$HOST" --port "$PORT"
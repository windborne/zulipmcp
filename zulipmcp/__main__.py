"""Allow `python -m zulipmcp` to start the MCP server."""
from zulipmcp.mcp import run_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zulip MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8235)
    args = parser.parse_args()
    run_server(transport=args.transport, host=args.host, port=args.port)

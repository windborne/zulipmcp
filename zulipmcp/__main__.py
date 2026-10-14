"""Allow `python -m zulipmcp` and the `zulipmcp` script to start the MCP server."""
import argparse

from zulipmcp.mcp import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Zulip MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8235)
    args = parser.parse_args()
    run_server(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

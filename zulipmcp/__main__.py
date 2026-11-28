"""Allow `python -m zulipmcp` and the `zulipmcp` script to start the MCP server."""
import argparse

from zulipmcp.mcp import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Zulip MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8235)
    parser.add_argument("--stateless", action="store_true",
                        help="HTTP transport without MCP session IDs")
    args = parser.parse_args()
    if args.stateless and args.transport != "http":
        parser.error("--stateless requires --transport http")
    run_server(transport=args.transport, host=args.host, port=args.port,
               stateless=args.stateless)


if __name__ == "__main__":
    main()

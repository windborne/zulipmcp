from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from zulipmcp.agent_backends import bootstrap_prompt, build_agent_cmd, build_agent_env


@dataclass
class Config:
    zuliprc: Path
    backend: str = "claude"
    agent_command: str = "claude"
    mcp_config: Path = Path(".mcp.json")
    system_prompt: Path = Path("prompt.md")
    working_dir: Path = Path(".")
    backend_flags: list[str] = field(default_factory=list)
    codex_permission_mode: str = "parity"


class AgentBackendTests(unittest.TestCase):
    def test_bootstrap_prompt_contains_session_context(self) -> None:
        prompt = bootstrap_prompt("engineering", "launch")

        self.assertIn("set_context('engineering', 'launch')", prompt)
        self.assertIn("listen", prompt)

    def test_claude_command_preserves_existing_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mcp_config = root / ".mcp.json"
            prompt = root / "system.md"
            zuliprc = root / ".zuliprc"
            mcp_config.write_text("{}")
            prompt.write_text("System prompt")
            zuliprc.write_text("[api]\n")
            cfg = Config(
                zuliprc=zuliprc,
                mcp_config=mcp_config,
                system_prompt=prompt,
                working_dir=root,
                backend_flags=["--model", "opus"],
            )

            cmd = build_agent_cmd(cfg, "engineering", "launch")

        self.assertEqual(cmd[0:4], ["claude", "--dangerously-skip-permissions", "--output-format", "stream-json"])
        self.assertIn("--verbose", cmd)
        self.assertIn("--mcp-config", cmd)
        self.assertIn("--append-system-prompt", cmd)
        self.assertIn("-p", cmd)
        self.assertEqual(cmd[-2:], ["--model", "opus"])

    def test_codex_command_translates_zulip_mcp_config_without_env_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mcp_config = root / ".mcp.json"
            prompt = root / "system.md"
            zuliprc = root / ".zuliprc"
            prompt.write_text("System prompt")
            zuliprc.write_text("[api]\n")
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "zulip": {
                        "command": "uv",
                        "args": ["run", "python", "-m", "zulipmcp.mcp"],
                        "env": {"EXAMPLE_TOKEN": "secret-value"},
                    }
                }
            }))
            cfg = Config(
                backend="codex",
                agent_command="codex",
                zuliprc=zuliprc,
                mcp_config=mcp_config,
                system_prompt=prompt,
                working_dir=root,
            )
            env = build_agent_env(cfg, {"id": 42, "sender_email": "user@example.com"})
            env["CALLER_CUSTOM_ENV"] = "caller-secret"

            cmd = build_agent_cmd(cfg, "engineering", "launch", env)

        joined = "\n".join(cmd)
        self.assertEqual(cmd[0], "codex")
        self.assertLess(cmd.index("--search"), cmd.index("exec"))
        self.assertIn("--json", cmd)
        self.assertNotIn("--ephemeral", cmd)
        self.assertIn("--yolo", cmd)
        self.assertIn("mcp_servers.zulip.command=\"uv\"", joined)
        self.assertIn("mcp_servers.zulip.tool_timeout_sec=10800", joined)
        self.assertIn("mcp_servers.zulip.required=true", joined)
        self.assertIn("EXAMPLE_TOKEN", joined)
        self.assertIn("SESSION_STREAM", joined)
        self.assertIn("SESSION_TOPIC", joined)
        self.assertIn("CALLER_CUSTOM_ENV", joined)
        self.assertNotIn("secret-value", joined)
        self.assertNotIn("caller-secret", joined)
        self.assertEqual(env["EXAMPLE_TOKEN"], "secret-value")
        self.assertEqual(env["TRIGGER_MESSAGE_ID"], "42")
        self.assertEqual(env["SESSION_USER_EMAIL"], "user@example.com")

    def test_codex_http_headers_are_forwarded_via_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mcp_config = root / ".mcp.json"
            zuliprc = root / ".zuliprc"
            zuliprc.write_text("[api]\n")
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "docs": {
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer top-secret"},
                    }
                }
            }))
            cfg = Config(
                backend="codex",
                agent_command="codex",
                zuliprc=zuliprc,
                mcp_config=mcp_config,
                system_prompt=root / "missing.md",
                working_dir=root,
            )
            env = build_agent_env(cfg, {"id": 1, "sender_email": ""})

            cmd = build_agent_cmd(cfg, "engineering", "launch", env)

        joined = "\n".join(cmd)
        self.assertIn("mcp_servers.docs.env_http_headers", joined)
        self.assertIn("ZULIPMCP_MCP_DOCS_AUTHORIZATION", joined)
        self.assertNotIn("top-secret", joined)
        self.assertEqual(env["ZULIPMCP_MCP_DOCS_AUTHORIZATION"], "Bearer top-secret")

    def test_codex_workspace_permission_mode(self) -> None:
        cfg = Config(
            backend="codex",
            agent_command="codex",
            zuliprc=Path(".zuliprc"),
            mcp_config=Path("missing.json"),
            system_prompt=Path("missing.md"),
            codex_permission_mode="workspace-write",
        )
        cmd = build_agent_cmd(cfg, "engineering", "launch", {})

        self.assertIn("--sandbox", cmd)
        self.assertIn("workspace-write", cmd)
        self.assertNotIn("--ask-for-approval", cmd)
        self.assertIn("approval_policy=\"never\"", "\n".join(cmd))
        self.assertIn("sandbox_workspace_write.network_access=true", "\n".join(cmd))
        self.assertNotIn("--yolo", cmd)

    def test_codex_top_level_only_flags_are_placed_before_exec(self) -> None:
        cfg = Config(
            backend="codex",
            agent_command="codex",
            zuliprc=Path(".zuliprc"),
            mcp_config=Path("missing.json"),
            system_prompt=Path("missing.md"),
            backend_flags=["--search", "--model", "gpt-5.5", "--output-schema", "schema.json"],
        )
        cmd = build_agent_cmd(cfg, "engineering", "launch", {})

        exec_index = cmd.index("exec")
        self.assertLess(cmd.index("--search"), exec_index)
        self.assertEqual(cmd.count("--search"), 1)
        self.assertGreater(cmd.index("--model"), exec_index)
        self.assertGreater(cmd.index("--output-schema"), exec_index)

    def test_codex_rejects_env_refs_in_argv_bound_mcp_fields(self) -> None:
        cases = [
            ("args", {"command": "tool", "args": ["--token", "${API_TOKEN}"]}),
            ("url", {"url": "https://example.com/mcp?token=${API_TOKEN}"}),
        ]
        for label, server in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                mcp_config = root / ".mcp.json"
                zuliprc = root / ".zuliprc"
                zuliprc.write_text("[api]\n")
                mcp_config.write_text(json.dumps({"mcpServers": {"leaky": server}}))
                cfg = Config(
                    backend="codex",
                    agent_command="codex",
                    zuliprc=zuliprc,
                    mcp_config=mcp_config,
                    system_prompt=root / "missing.md",
                    working_dir=root,
                )
                env = build_agent_env(cfg, {"id": 1, "sender_email": ""})
                env["API_TOKEN"] = "top-secret"

                with self.assertRaisesRegex(ValueError, "process argv"):
                    build_agent_cmd(cfg, "engineering", "launch", env)


if __name__ == "__main__":
    unittest.main()

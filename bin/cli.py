#!/usr/bin/env python3
"""
clip.py — Command-Line Interface for running agentic tasks.

Runs FinAI agent workflows directly from the terminal, supporting all
agent profiles, LLM providers, and optional RAG / publishing features.

Usage examples::

    # Basic agent task with Ollama (default)
    python bin/clip.py --agent Data_Analyst "Analyse NVDA revenue trends"

    # Use GitHub Models
    python bin/clip.py --agent Research_Publisher \\
        --provider github --model openai/gpt-4o \\
        --github-token "$GITHUB_TOKEN" \\
        "Write a research report on AAPL"

    # Use DeepSeek
    python bin/clip.py --agent Data_Analyst \\
        --provider deepseek --model deepseek-chat \\
        --deepseek-token "$DEEPSEEK_TOKEN" \\
        "Compare MSFT and GOOGL financials"

    # Publish as PDF and email
    python bin/clip.py --agent Research_Publisher \\
        --format pdf --email analyst@firm.com \\
        "Analyse the energy sector outlook"

    # List available agents
    python bin/clip.py --list-agents

    # List available tools
    python bin/clip.py --list-tools
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path setup — locate the fin-ai-research-studio package
# ---------------------------------------------------------------------------

def _find_fin_ai_root() -> Path | None:
    """Locate the ``fin-ai-research-studio`` project root."""
    # Search upward from this script's location
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir.parent, *script_dir.parents):
        src_dir = candidate / "src"
        if (src_dir / "fin_ai" / "__init__.py").exists():
            return candidate
    # Check the other workspace location
    alt = Path.home() / "Dev" / "Projects" / "ai" / "fin-ai-research-studio"
    if (alt / "src" / "fin_ai" / "__init__.py").exists():
        return alt
    return None


def _ensure_importable() -> None:
    """Add the fin-ai-research-studio paths to ``sys.path``."""
    root = _find_fin_ai_root()
    if root is None:
        print(
            "ERROR: Cannot locate fin-ai-research-studio project. "
            "Ensure it is cloned alongside quant-finance.",
            file=sys.stderr,
        )
        sys.exit(1)
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Also add the project root so the ``dashboard`` package is importable
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FinAI agentic tasks from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Agent selection
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Task prompt to send to the agent. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--agent",
        "-a",
        default="Data_Analyst",
        help="Agent profile name (default: Data_Analyst). "
             "Use --list-agents to see available profiles.",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all available agent profiles and exit.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available financial data tools and exit.",
    )

    # Provider / model
    parser.add_argument(
        "--provider",
        "-p",
        default="ollama",
        choices=["ollama", "github", "deepseek", "proxied_github", "proxied_deepseek"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        help="Model identifier (e.g. 'llama3.1', 'openai/gpt-4o', 'deepseek-chat'). "
             "Defaults to the provider's default model.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        help="Ollama API base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--github-endpoint",
        default=os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference"),
        help="GitHub Models API endpoint.",
    )
    parser.add_argument(
        "--github-token",
        default=os.getenv("GITHUB_TOKEN", ""),
        help="GitHub token for GitHub Models authentication.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--deepseek-token",
        default=os.getenv("DEEPSEEK_TOKEN", ""),
        help="DeepSeek API token.",
    )

    # Embedding config (for RAG agents)
    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("DEFAULT_EMBEDDINGS_PROVIDER", "ollama"),
        choices=["ollama", "github"],
        help="Embedding provider (default: ollama).",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text:latest"),
        help="Embedding model identifier.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        help="Embedding API base URL.",
    )

    # Publishing options
    parser.add_argument(
        "--publisher",
        action="store_true",
        help="Run as publisher agent (generates HTML/PDF report). "
             "Equivalent to --agent Research_Publisher.",
    )
    parser.add_argument(
        "--format",
        choices=["html", "pdf"],
        default="html",
        help="Report format for publisher agent (default: html).",
    )
    parser.add_argument(
        "--email",
        default="",
        help="Email address to send the published report to.",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full result as JSON (includes metadata, publication info).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output (only print the agent response).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_importable()

    from fin_ai.agents.agent_library import library as agent_library
    from fin_ai.core.processor import (
        build_agent_llm_config,
        run_agent_task,
    )
    from fin_ai.core.tools import YAHOO_FINANCE_TOOLS

    parser = build_parser()
    args = parser.parse_args()

    # --list-agents
    if args.list_agents:
        print("Available agent profiles:\n")
        for name, config in agent_library.items():
            desc = config.get("profile", "")[:80].replace("\n", " ").strip()
            tools = config.get("tools", [])
            print(f"  {name}")
            if desc:
                print(f"      {desc}...")
            if tools:
                print(f"      Tools: {', '.join(tools)}")
            print()
        return

    # --list-tools
    if args.list_tools:
        print("Available financial data tools:\n")
        for tool in YAHOO_FINANCE_TOOLS:
            if tool.get("type") == "function":
                fn = tool["function"]
                name = fn.get("name", "?")
                desc = fn.get("description", "")
                params = fn.get("parameters", {}).get("properties", {})
                print(f"  {name}")
                if desc:
                    print(f"      {desc}")
                if params:
                    print(f"      Parameters: {', '.join(params.keys())}")
                print()
        return

    # Resolve prompt
    prompt = args.prompt.strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.print_help()
        print("\nERROR: A prompt is required. Provide it as an argument or via stdin.", file=sys.stderr)
        sys.exit(1)

    # Resolve agent name
    agent_name = args.agent
    if args.publisher:
        agent_name = "Research_Publisher"

    # Validate agent exists
    if agent_name not in agent_library:
        print(
            f"ERROR: Unknown agent '{agent_name}'. "
            f"Use --list-agents to see available profiles.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve model default per provider
    model = args.model
    if not model:
        provider_defaults = {
            "ollama": os.getenv("OLLAMA_MODEL", "llama3.1"),
            "github": os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
            "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "proxied_github": os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
            "proxied_deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        }
        model = provider_defaults.get(args.provider, "llama3.1")

    # Build LLM config
    llm_config = build_agent_llm_config(
        provider=args.provider,
        model=model,
        ollama_base_url=args.ollama_base_url,
        github_endpoint=args.github_endpoint,
        github_token=args.github_token,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_token=args.deepseek_token,
    )

    # Determine if this is a publisher agent
    is_publisher = agent_name == "Research_Publisher"

    if not args.quiet:
        print(f"Agent:      {agent_name}")
        print(f"Provider:   {args.provider}")
        print(f"Model:      {model}")
        print(f"Publisher:  {is_publisher}")
        if is_publisher:
            print(f"Format:     {args.format}")
            if args.email:
                print(f"Email:      {args.email}")
        print(f"Prompt:     {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print("-" * 60)

    # Run the agent
    result = run_agent_task(
        agent_name=agent_name,
        prompt=prompt,
        llm_config=llm_config,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        embedding_base_url=args.embedding_base_url,
        chat_provider=args.provider,
        is_publisher=is_publisher,
        publisher_format=args.format,
        publisher_email=args.email,
    )

    # Output
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("success"):
        response = result.get("response", "")
        if response:
            print(response)
        else:
            print("(no response content)")

        publication = result.get("publication")
        if publication:
            try:
                pub_data = json.loads(publication) if isinstance(publication, str) else publication
                pub_info = pub_data.get("publish", pub_data)
                filepath = pub_info.get("filepath", "")
                if filepath:
                    print(f"\n[Published] {filepath}")
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"\n[Publication] {publication}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""Command line entry point.

    python -m agentic_core serve              # start the API (port 8088)
    python -m agentic_core seed [--reset]     # (re)create the dummy HR database
    python -m agentic_core index [--force]    # (re)index the policy documents
    python -m agentic_core chat "question" [--as E1002] [--model groq/qwen/qwen3.8-27b]
    python -m agentic_core models             # list models the agents can use
    python -m agentic_core graph              # print the orchestration graph (Mermaid)
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _service():
    from .main import build_service, configure_logging
    from .settings import get_settings

    settings = get_settings()
    configure_logging(settings)
    return build_service(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_core")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")

    seed = sub.add_parser("seed", help="create/reset the dummy HR database")
    seed.add_argument("--reset", action="store_true")

    index = sub.add_parser("index", help="index the policy documents")
    index.add_argument("--force", action="store_true")

    chat = sub.add_parser("chat", help="ask the agents a question from the terminal")
    chat.add_argument("message")
    chat.add_argument("--as", dest="employee", default="", help="employee id or email to act as")
    chat.add_argument("--model", default=None)
    chat.add_argument("--quiet", action="store_true", help="hide the agent trace")

    sub.add_parser("models", help="list available models")
    sub.add_parser("graph", help="print the LangGraph orchestration graph as Mermaid")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from .settings import get_settings

        settings = get_settings()
        uvicorn.run(
            "agentic_core.main:app",
            host=args.host or settings.agent_core_host,
            port=args.port or settings.agent_core_port,
            reload=args.reload,
        )
        return 0

    if args.command == "seed":
        from .db.seed import seed_database
        from .db.session import init_engine
        from .settings import get_settings

        init_engine(get_settings().resolved_database_url)
        print(seed_database(reset=args.reset))
        return 0

    service = _service()

    if args.command == "index":
        print(service.knowledge.index_policies(force=args.force))
        return 0

    if args.command == "models":
        async def show():
            default = await service.registry.default_model_id()
            for m in await service.registry.available_models():
                print(("* " if m.id == default else "  ") + m.id)
            print(f"\ndefault: {default}")

        asyncio.run(show())
        return 0

    if args.command == "graph":
        print(service.orchestrator.mermaid())
        return 0

    if args.command == "chat":
        from .service import ChatRequest

        async def run():
            request = ChatRequest(
                messages=[{"role": "user", "content": args.message}],
                model=args.model,
                employee_id=args.employee,
                conversation_id="cli",
            )
            last = None
            async for event in service.stream(request):
                if event.kind == "reasoning" and not args.quiet:
                    if last != "reasoning":
                        sys.stdout.write("\n--- agent trace ---\n")
                    sys.stdout.write(event.text)
                elif event.kind == "content":
                    if last != "content":
                        sys.stdout.write("\n--- answer ---\n")
                    sys.stdout.write(event.text)
                last = event.kind
                sys.stdout.flush()
            sys.stdout.write("\n")

        asyncio.run(run())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

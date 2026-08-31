from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matensemble-dashboard",
        description="Serve the multi-workflow monitoring dashboard.",
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scan-interval", type=float, default=5.0)
    parser.add_argument("--stale-after", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "The dashboard command requires the uvicorn dependency."
        ) from exc
    from matensemble.dashboard import create_dashboard_app

    root = Path(args.root).expanduser().resolve()
    app = create_dashboard_app(
        root,
        scan_interval=args.scan_interval,
        stale_after=args.stale_after,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

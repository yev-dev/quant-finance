#!/usr/bin/env python3
"""
second_model.py

Command-line wrapper that accepts --start-date, --end-date (YYYY-MM-DD), and an
optional --operation-type (string) to customize behavior.

Also simulates a long-running execution with --duration-seconds (default 45),
clamped to a maximum of 60 seconds. Prints periodic progress updates and
handles termination signals gracefully.

Usage:
  python second_model.py --start-date 2025-01-01 --end-date 2025-12-31 \
    --operation-type aggregate
"""

import argparse
import signal
import sys
import time
from datetime import datetime, date
from typing import Optional


def _parse_date(value: str) -> date:
    """Parse a date string in YYYY-MM-DD format into a date object.

    Raises argparse.ArgumentTypeError on invalid format.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format YYYY-MM-DD."
        )


_stop_requested = False


def _handle_signal(signum, frame):
    # Request a graceful stop; loop will notice and exit promptly.
    global _stop_requested
    _stop_requested = True
    print(f"[second_model] Received signal {signum}; stopping...", flush=True)


def main(
    start_date: date,
    end_date: date,
    operation_type: Optional[str],
    duration_seconds: int,
) -> None:
    """Main entrypoint for the second model.

    Simulates work by sleeping and emitting progress for up to duration_seconds
    (capped at 60). Exits early when a termination signal is received.
    """
    op = operation_type or "none"
    # Clamp duration to [1, 60]
    if duration_seconds < 1:
        duration_seconds = 1
    if duration_seconds > 60:
        print(
            f"[second_model] duration_seconds={duration_seconds} exceeds 60; clamping to 60",
            flush=True,
        )
        duration_seconds = 60

    print(
        f"[second_model] Starting run: start_date={start_date}, end_date={end_date}, "
        f"operation_type={op}, duration_seconds={duration_seconds}",
        flush=True,
    )

    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Simple progress loop with 1-second ticks
    start_ts = time.time()
    for i in range(duration_seconds):
        if _stop_requested:
            print("[second_model] Stop requested; exiting.", flush=True)
            break
        elapsed = int(time.time() - start_ts)
        pct = int(((i + 1) / duration_seconds) * 100)
        print(f"[second_model] Progress: {pct}% (elapsed={elapsed}s)", flush=True)
        time.sleep(1)

    total_elapsed = int(time.time() - start_ts)
    if _stop_requested:
        print(f"[second_model] Run terminated after {total_elapsed}s", flush=True)
    else:
        print(f"[second_model] Run completed in {total_elapsed}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the second model over a date range with an optional operation type"
        )
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--operation-type",
        required=False,
        type=str,
        default=None,
        help=(
            "Optional operation type (e.g., 'aggregate', 'train', 'score'); "
            "if omitted, defaults to 'none'"
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        required=False,
        type=int,
        default=45,
        help=(
            "Simulated run duration in seconds (1-60); default 45. "
            "Values above 60 will be clamped."
        ),
    )

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("start-date must be less than or equal to end-date")

    try:
        main(args.start_date, args.end_date, args.operation_type, args.duration_seconds)
    except KeyboardInterrupt:
        # In case SIGINT arrives before our handler is registered
        print("[second_model] KeyboardInterrupt; exiting.", flush=True)
        sys.exit(130)

#!/usr/bin/env python3
"""
first_model.py

A simple command-line wrapper that accepts --start-date and --end-date
(YYYY-MM-DD) and invokes a main(start_date, end_date) function.

Usage:
  python first_model.py --start-date 2025-01-01 --end-date 2025-12-31
"""

import argparse
import time
from datetime import datetime, date


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


def main(start_date: date, end_date: date) -> None:
    """Main entrypoint for the first model.

    Replace this placeholder body with your actual model logic.
    """
    print(f"[first_model] Running main with start_date={start_date} end_date={end_date}")
    # Simulate a long-running task with a 30-second timer
    print("[first_model] Starting 30-second timer...")
    time.sleep(30)
    print("[first_model] 30-second timer complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the first model main function over a date range"
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

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("start-date must be less than or equal to end-date")

    main(args.start_date, args.end_date)

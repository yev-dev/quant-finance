#!/usr/bin/env python3
"""
second_model.py

Command-line wrapper that accepts --start-date, --end-date (YYYY-MM-DD), and an
optional --operation-type (string) to customize behavior.

Usage:
  python second_model.py --start-date 2025-01-01 --end-date 2025-12-31 \
    --operation-type aggregate
"""

import argparse
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


def main(start_date: date, end_date: date, operation_type: Optional[str]) -> None:
    """Main entrypoint for the second model.

    Replace this placeholder body with your actual model logic.
    """
    op = operation_type or "none"
    print(
        f"[second_model] Running main with start_date={start_date} end_date={end_date} "
        f"operation_type={op}"
    )


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

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("start-date must be less than or equal to end-date")

    main(args.start_date, args.end_date, args.operation_type)

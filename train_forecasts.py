"""Generate cleaned demand data, model accuracy metrics, and forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ev_forecasting import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("ev_charging_dataset.xlsx"),
        help="Path to the Excel workbook (default: ev_charging_dataset.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("outputs"),
        help="Directory for CSV outputs (default: ./outputs)",
    )
    parser.add_argument("--horizon", type=int, default=365, help="Forecast days (default: 365)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input workbook not found: {args.input}")
    metrics, forecasts = run_pipeline(args.input, args.output, args.horizon)
    print(f"Saved forecast outputs to {args.output.resolve()}")
    print(metrics.to_string(index=False))
    print(f"Forecast rows: {len(forecasts):,}")


if __name__ == "__main__":
    main()

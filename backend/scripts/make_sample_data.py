"""Generate a Telco-style churn CSV (7,043 rows x 21 columns) for demos and tests.

Matches the dashboard's dummy numbers: ~26.5% churn, 11 missing ``total_charges``,
and ``tenure`` / ``total_charges`` correlated above 0.85.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_churn(rows: int = 7043, seed: int = 7) -> pd.DataFrame:
    """Return a synthetic churn DataFrame."""
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.gamma(1.6, 20.0, rows).round(), 0, 72).astype(int)
    contract_len = rng.choice([0, 1, 2], rows, p=[0.55, 0.21, 0.24])
    monthly = np.clip(rng.normal(64.8, 30.1, rows), 18.3, 118.8).round(2)
    total = (tenure * (0.55 * monthly + 0.45 * 64.8) * rng.normal(1.0, 0.05, rows)).round(2).clip(0, 8684.8)
    support_calls = np.clip(rng.poisson(1.6, rows), 0, 9)
    logit = -0.9 + 0.012 * (monthly - 65) - 0.045 * tenure - 0.7 * contract_len + 0.28 * support_calls
    churn = rng.random(rows) < 1 / (1 + np.exp(-logit - 0.95))
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i:05d}" for i in range(rows)],
            "gender": rng.choice(["Female", "Male"], rows),
            "senior_citizen": rng.choice([0, 1], rows, p=[0.84, 0.16]),
            "partner": rng.choice(["Yes", "No"], rows),
            "dependents": rng.choice(["Yes", "No"], rows, p=[0.3, 0.7]),
            "tenure": tenure,
            "phone_service": rng.choice(["Yes", "No"], rows, p=[0.9, 0.1]),
            "internet_service": rng.choice(["DSL", "Fiber", "NoService"], rows, p=[0.34, 0.44, 0.22]),
            "online_security": rng.choice(["Yes", "No"], rows),
            "tech_support": rng.choice(["Yes", "No"], rows),
            "streaming_tv": rng.choice(["Yes", "No"], rows),
            "paperless_billing": rng.choice(["Yes", "No"], rows, p=[0.6, 0.4]),
            "payment_method": rng.choice(["Card", "Bank", "Check", "Mail"], rows),
            "contract_len": contract_len,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": support_calls,
            "num_products": rng.integers(1, 6, rows),
            "late_payments": rng.poisson(0.6, rows),
            "region": rng.choice(["N", "S", "E", "W"], rows),
            "churn": np.where(churn, "Yes", "No"),
        }
    )
    df.loc[rng.choice(rows, 11, replace=False), "total_charges"] = np.nan
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sample_churn.csv")
    parser.add_argument("--rows", type=int, default=7043)
    args = parser.parse_args()
    frame = make_churn(args.rows)
    Path(args.out).write_text(frame.to_csv(index=False))
    print(f"wrote {args.out}: {frame.shape}, churn={float((frame.churn == 'Yes').mean()):.3f}")

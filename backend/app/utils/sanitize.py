"""JSON safety: turn numpy/pandas values into plain JSON-compatible Python."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from starlette.responses import JSONResponse


def sanitize(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-safe natives (NaN/inf become ``None``)."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, BaseModel):
        return sanitize(value.model_dump())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    return value


class SafeJSONResponse(JSONResponse):
    """JSON response that can never emit NaN/Infinity tokens."""

    def render(self, content: Any) -> bytes:
        """Sanitize then serialize strictly."""
        return json.dumps(
            sanitize(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")

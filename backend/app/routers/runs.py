"""Saved runs, datasets and comparison (persisted in SQLite, independent of sessions)."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from app.deps import RunsDep
from app.schemas.common import ApiModel, ErrorResponse
from pydantic import Field

router = APIRouter(prefix="/api/v1", tags=["runs"], responses={404: {"model": ErrorResponse}})


class CompareRequest(ApiModel):
    """Runs to line up against each other."""

    run_ids: list[str] = Field(min_length=2, max_length=8)


class RunPatch(ApiModel):
    """Rename or annotate a run."""

    label: str | None = Field(default=None, min_length=1, max_length=80)
    notes: str | None = Field(default=None, max_length=1000)


@router.get("/datasets", summary="Datasets seen so far (grouped by content fingerprint)", response_model=None)
async def list_datasets(runs: RunsDep) -> dict[str, Any]:
    """Every uploaded dataset that has at least been trained on once, with its profile and run count."""
    datasets = await asyncio.to_thread(runs.list_datasets)
    return {"datasets": [d for d in datasets if d["n_runs"] > 0], "total": len([d for d in datasets if d["n_runs"] > 0])}


@router.delete("/datasets/{dataset_id}", summary="Delete a dataset and its runs", response_model=None)
async def delete_dataset(dataset_id: str, runs: RunsDep) -> dict[str, Any]:
    """Remove the dataset record and every run trained on it."""
    await asyncio.to_thread(runs.delete_dataset, dataset_id)
    return {"deleted": True, "dataset_id": dataset_id}


@router.get("/runs", summary="Saved training runs, newest first", response_model=None)
async def list_runs(
    runs: RunsDep, dataset_id: str | None = None, task: str | None = None, limit: int = Query(200, ge=1, le=500)
) -> dict[str, Any]:
    """Run summaries (settings, best model, tuned flag)."""
    items = await asyncio.to_thread(runs.list_runs, dataset_id, task, limit)
    return {"runs": items, "total": len(items)}


@router.get("/runs/{run_id}", summary="One saved run with its leaderboard and curves", response_model=None)
async def get_run(run_id: str, runs: RunsDep) -> dict[str, Any]:
    """Full leaderboard rows, split, settings, dataset/target profile and downsampled ROC/PR curves."""
    return await asyncio.to_thread(runs.get_run, run_id)


@router.patch("/runs/{run_id}", summary="Rename or annotate a run", response_model=None)
async def patch_run(run_id: str, body: RunPatch, runs: RunsDep) -> dict[str, Any]:
    """Change the label and/or notes."""
    await asyncio.to_thread(runs.update_run, run_id, body.label, body.notes)
    return await asyncio.to_thread(runs.get_run, run_id, False)


@router.delete("/runs/{run_id}", summary="Delete a run", response_model=None)
async def delete_run(run_id: str, runs: RunsDep) -> dict[str, Any]:
    """Remove one saved run."""
    await asyncio.to_thread(runs.delete_run, run_id)
    return {"deleted": True, "run_id": run_id}


@router.post("/compare", summary="Compare 2-8 runs (models x runs, datasets, curves)", response_model=None)
async def compare(body: CompareRequest, runs: RunsDep) -> dict[str, Any]:
    """Align runs: per-model metrics in every run, dataset profiles, ROC/PR overlay data and caveats."""
    return await asyncio.to_thread(runs.compare, body.run_ids)

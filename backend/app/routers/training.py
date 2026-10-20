"""Model training, job polling and leaderboard endpoints."""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Request

from app.config import Settings
from app.deps import RadioDep, RunsDep, SessionDep, SettingsDep, require_stage
from app.errors import BadRequestError, ConflictError, NotFoundError
from app.schemas.common import ErrorResponse
from app.schemas.training import JobStatus, TrainRequest, TrainResult, TrainStarted
from app.services import modeling
from app.services.analysis import stored_curves
from app.services.runs import RunStore, target_profile
from app.services.model_registry import ModelSpec, resolve_model
from app.state.session_store import JobRecord, Session, Stage
from app.ws.radio import Radio

logger = logging.getLogger("mldash.training")

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["training"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


async def _execute(
    session: Session,
    job: JobRecord,
    prepared: modeling.Prepared,
    specs: list[ModelSpec],
    body: TrainRequest,
    settings: Settings,
    radio: Radio,
    runs: RunStore,
) -> None:
    """Run the training thread, then merge results into the session and save the run."""
    channel = radio.channel(session)
    try:
        outcome = await asyncio.to_thread(
            modeling.run_training,
            prepared,
            specs,
            test_size=body.test_size,
            scale_features=body.scale_features,
            handle_imbalance=body.handle_imbalance,
            average=body.average,
            settings=settings,
            channel=channel,
            job=job,
            tune_trials=body.tune_trials if body.tune else 0,
            cv_folds=body.cv_folds if body.cross_validate else 0,
        )
        async with session.lock:
            session.results.update(outcome["rows"])
            session.fitted.update(outcome["fitted"])
            session.tuning.update(outcome["tuning"])
            session.telemetry.update(outcome["curves"])
            for key in outcome["rows"]:
                session.failed.pop(key, None)
            session.failed.update(outcome["failed"])
            session.split, session.average = outcome["split"], outcome["average"]
            session.exclusions, session.warnings = outcome["exclusions"], outcome["warnings"]
            if session.results:
                session.stage = Stage.TRAINED
            job.result = modeling.build_train_result(
                session.results, session.failed, session.split, session.warnings, session.exclusions,
                session.average, session.task,
            )
            job.status = "COMPLETE" if outcome["rows"] else "FAILED"
            job.error = None if outcome["rows"] else "ALL MODELS FAILED"
            if outcome["rows"]:
                job.run_id = await asyncio.to_thread(_save_run, session, outcome, body, settings, runs)
        top = job.result["top_model"]
        if top:
            channel.success("TRAIN", f"TRAINING COMPLETE TOP MODEL {top['abbr']} {top['score_name']}={top['score']:.3f}")
        else:
            channel.error("TRAIN", "TRAINING FAILED ALL MODELS CRASHED")
    except Exception:  # noqa: BLE001 - never leave a job stuck in RUNNING
        logger.exception("training job %s crashed", job.job_id)
        job.status, job.error = "FAILED", "Training failed unexpectedly."
        channel.error("TRAIN", "TRAINING FAILED UNEXPECTEDLY")


def _save_run(session: Session, outcome: dict, body: TrainRequest, settings: Settings, runs: RunStore) -> str | None:
    """Persist this job's leaderboard so it can be compared later (failures here must never fail the job)."""
    try:
        board = modeling.rank_results(outcome["rows"])
        top = board[0] if board else None
        raw, work = session.raw_df, session.working_df
        payload = {
            "settings": {
                "test_size": body.test_size, "scale_features": body.scale_features, "handle_imbalance": body.handle_imbalance,
                "average": outcome["average"], "tune": body.tune, "tune_trials": body.tune_trials if body.tune else 0,
                "cross_validate": body.cross_validate, "cv_folds": body.cv_folds if body.cross_validate else 0,
            },
            "split": outcome["split"], "leaderboard": board,
            "failed_models": outcome["failed"], "warnings": outcome["warnings"], "rows_used": int(len(work)),
            "raw_rows": int(len(raw)),
            "cleaned": bool(work.shape != raw.shape or int(raw.isna().sum().sum()) != int(work.isna().sum().sum())),
            "target_profile": target_profile(work[session.target], session.task) if session.target else None,
            "top": {"abbr": top["abbr"], "name": top["model"], "score": top.get("score"), "score_name": top.get("score_name")} if top else None,
        }
        curves = {k: stored_curves(session.fitted[k], settings) for k in outcome["rows"] if k in session.fitted}
        label = body.label or f"{session.filename} / {session.target}"
        dataset_id = session.dataset_id or runs.upsert_dataset(raw, session.filename)
        return runs.add_run(dataset_id, label, session.task, session.target or "", payload, {k: v for k, v in curves.items() if v})
    except Exception:  # noqa: BLE001
        logger.exception("could not save run")
        return None


@router.post(
    "/train",
    response_model=TrainStarted,
    status_code=202,
    summary="Start a background training job",
    responses={400: {"model": ErrorResponse}},
)
async def start_training(
    body: TrainRequest, session: SessionDep, settings: SettingsDep, radio: RadioDep, runs: RunsDep, request: Request
) -> TrainStarted:
    """Validate, then train the requested models in a worker thread and return a job id."""
    require_stage(session, Stage.TARGET_SET, "train models")
    assert session.target is not None
    specs: list[ModelSpec] = []
    for name in body.models:
        spec = resolve_model(name)
        available, reason = spec.availability()
        if not available:
            raise BadRequestError(
                f"Model '{spec.abbr}' is unavailable: {reason}.", error_code="MODEL_UNAVAILABLE", details=reason
            )
        if not spec.supports(session.task):
            raise BadRequestError(
                f"Model '{spec.abbr}' only supports classification, but the target is a regression target.",
                error_code="MODEL_NOT_FOR_TASK",
                details={"task": session.task},
            )
        if spec not in specs:
            specs.append(spec)
    async with session.lock:
        if any(job.status == "RUNNING" for job in session.jobs.values()):
            raise ConflictError("A training job is already running for this session.", error_code="TRAINING_IN_PROGRESS")
        prepared = await asyncio.to_thread(modeling.prepare_dataset, session.working_df, session.target, settings, session.task)
        signature = modeling.train_signature(body.test_size, body.scale_features, body.handle_imbalance, body.average)
        if body.tune_trials > settings.tune_max_trials:
            raise BadRequestError(f"At most {settings.tune_max_trials} tuning trials are allowed.", error_code="TOO_MANY_TRIALS")
        channel = radio.channel(session)
        if session.train_signature is not None and session.train_signature != signature:
            session.reset_training()
            channel.warn("TRAIN", "SETTINGS CHANGED - PREVIOUS RESULTS CLEARED")
        session.train_signature = signature
        job = JobRecord(
            job_id=str(uuid.uuid4()),
            models=[s.key for s in specs],
            model_status={s.abbr: ["PENDING", "PENDING", "PENDING"] for s in specs},
        )
        session.jobs[job.job_id] = job
    channel.info("TRAIN", f"TRAINING STARTED {len(specs)} MODELS")
    task = asyncio.create_task(_execute(session, job, prepared, specs, body, settings, radio, runs))
    tasks: set[asyncio.Task[None]] = request.app.state.tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return TrainStarted(job_id=job.job_id, models=job.models)


@router.get("/train/{job_id}", response_model=JobStatus, summary="Job status and results")
async def job_status(job_id: str, session: SessionDep) -> JobStatus:
    """Poll a job: RUNNING (with per-model C/T/E progress), COMPLETE or FAILED."""
    job = session.jobs.get(job_id)
    if job is None:
        raise NotFoundError(f"Job '{job_id}' does not exist.", error_code="JOB_NOT_FOUND")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        models=job.models,
        model_status=job.model_status,
        current=job.current,
        done=job.done,
        total=len(job.models),
        result=job.result,
        error=job.error,
        progress=job.progress,
        run_id=job.run_id,
    )


@router.get("/leaderboard", response_model=TrainResult, summary="Ranked results of everything trained so far")
async def leaderboard(session: SessionDep) -> TrainResult:
    """Merged ranking (F1 desc) with ``gap_to_top`` for every trained model."""
    require_stage(session, Stage.TRAINED, "read the leaderboard")
    return TrainResult(
        **modeling.build_train_result(
            session.results, session.failed, session.split, session.warnings, session.exclusions,
            session.average, session.task,
        )
    )

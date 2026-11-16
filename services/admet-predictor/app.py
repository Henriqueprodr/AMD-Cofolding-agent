#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ADMET inference REST service (single FastAPI app, boltz2/posebusters convention).

Models (XGBoost + Chemprop-MTL), response-domain statistics, and the
applicability-domain reference set load ONCE at startup and stay resident, so
each request is only compute. The implementation lives in the
``admet_predictor`` package (installed into the image by the Dockerfile); this
file is the only FastAPI app.

Run:  uvicorn app:app --host 0.0.0.0 --port 8084
Endpoints:
  GET  /health                          -> liveness + endpoint list + device
  POST /score    {"smiles":[...], "params":{"endpoint": "logp"}}  -> REINVENT scorer contract
  POST /predict  {"smiles":[...], "format": "records|compact|table"}  -> full per-endpoint records
  POST /predict_csv  (multipart CSV upload)  -> JSON or CSV
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from admet_predictor import core as P

import inspect as _inspect

# Compat shims for the flat-layout (pre-manifest) image core.
# The baked image core accepts only (smiles_list, batch_size, fmt); the
# new core also accepts model_set.  Detect once at import time and route
# accordingly so both layouts work without per-route branching.
_PREDICT_HAS_MODEL_SET = "model_set" in _inspect.signature(P.predict).parameters
_PREDICT_HAS_FMT       = "fmt"       in _inspect.signature(P.predict).parameters


def _batch_size() -> int:
    """Return core BATCH_SIZE, falling back to env/default for old images."""
    return getattr(P, "BATCH_SIZE", int(__import__("os").environ.get("PRED_BATCH", "1000")))


def _call_predict(smiles, *, model_set=None, batch_size=None, fmt="records"):
    """Call P.predict() with only the kwargs the installed core accepts."""
    kw: dict = {"batch_size": batch_size or _batch_size()}
    if _PREDICT_HAS_MODEL_SET and model_set is not None:
        kw["model_set"] = model_set
    if _PREDICT_HAS_FMT:
        kw["fmt"] = fmt
    return P.predict(smiles, **kw)


def _get_endpoints_compat(model_set=None) -> list:
    """Get endpoint list; falls back to directory scan for flat-layout cores."""
    try:
        return P.get_endpoints(model_set or P.DEFAULT_MODEL_SET)
    except Exception:
        return _flat_endpoints_fallback().get("endpoints", [])


def _normalize_model_set_compat(model_set: str) -> str:
    """Normalize model_set; returns as-is for flat-layout cores."""
    try:
        return P.normalize_model_set(model_set)
    except Exception:
        return model_set


def _endpoint_meta_compat(model_set=None) -> dict:
    """Return endpoint metadata; empty dict for flat-layout cores."""
    try:
        return P.get_endpoint_meta(model_set or P.DEFAULT_MODEL_SET)
    except Exception:
        return {}


class PredictRequest(BaseModel):
    smiles: list[str]
    model_set: str = "drug"          # drug | agro (pharma accepted as alias)
    batch_size: Optional[int] = None
    format: str = "records"          # records | compact | table


class ScoreParams(BaseModel):
    endpoint: str = "logp"           # one of core.ENDPOINTS
    model_set: str = "drug"
    batch_size: Optional[int] = None


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams = Field(default_factory=ScoreParams)


class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the default stack; the alternative stack remains lazy.
    # Guard the warmup: when the model weights are staged in the new manifest
    # layout this loads the XGBoost ensemble and Chemprop checkpoint so the
    # first real request is not slow. When only the flat (pre-manifest) layout
    # is present, normalize_model_set raises ValueError and we skip warming —
    # the flat-layout code path in core handles the first request on demand.
    try:
        P.predict(["CCO"], model_set=P.DEFAULT_MODEL_SET)
    except Exception:
        pass
    yield


app = FastAPI(title="ADMET predictor", version="1.0", lifespan=lifespan)


def _flat_endpoints_fallback() -> dict:
    """Return ``endpoints`` and ``endpoints_model_set`` for the health response.

    Tries the manifest-based API first. When no model sets are staged (no
    manifest.json files found under MODELS_DIR), falls back to scanning the
    flat xgb subdirectory so the service comes up and answers health checks
    even without the full dual-stack model weight package.

    The flat layout lives at either MODELS_DIR/xgb/{endpoint}/ or
    MODELS_DIR/admet/xgb/{endpoint}/ depending on whether MODELS_DIR already
    points inside the admet subtree.
    """
    try:
        if P.available_model_sets():
            return {
                "endpoints": P.get_endpoints(P.DEFAULT_MODEL_SET),
                "endpoints_model_set": P.DEFAULT_MODEL_SET,
            }
    except Exception:
        pass
    # Flat layout: scan candidate xgb directories.
    for candidate in (P.MODELS_DIR / "xgb", P.MODELS_DIR / "admet" / "xgb"):
        try:
            if candidate.is_dir():
                endpoints = sorted(d.name for d in candidate.iterdir() if d.is_dir())
                if endpoints:
                    return {"endpoints": endpoints, "endpoints_model_set": None}
        except Exception:
            pass
    return {"endpoints": [], "endpoints_model_set": None}


@app.get("/health")
def health():
    # Build the model_sets block (empty when manifests are not staged).
    model_sets = {}
    try:
        for model_set in P.available_model_sets():
            manifest = P.get_manifest(model_set)
            model_sets[model_set] = {
                "display_name": manifest.get("display_name", model_set),
                "endpoints": manifest["endpoint_order"],
                "model_members": list(manifest.get("models", {})),
                "response_domain": manifest.get("response_domain"),
                "physical_member_validation": True,
            }
    except Exception:
        pass  # old image: core has no available_model_sets; model_sets stays {}

    # Batch size and device come from core attrs that may not exist in older images.
    try:
        batch_size = P.BATCH_SIZE
    except AttributeError:
        batch_size = int(__import__("os").environ.get("PRED_BATCH", "1000"))
    try:
        device = P._device()
    except Exception:
        device = "cpu"
    try:
        default_model_set = P.DEFAULT_MODEL_SET
    except AttributeError:
        default_model_set = __import__("os").environ.get("ADMET_MODEL_SET", "drug")

    return {
        "status": "ok",
        "default_model_set": default_model_set,
        "model_sets": model_sets,
        # Backward-compatible endpoint list, for callers written before the
        # second stack existed. It covers the default stack only, and says so
        # beside it: unlabelled it reads as *the* endpoint list, which is how
        # the agro stack came to look as though it were missing. The per-stack
        # lists are under model_sets.
        # get_endpoints requires a manifest.json for the model set. When the
        # full model weights have not yet been staged in the manifest layout
        # (models/admet/{drug,agro}/manifest.json), fall back to scanning the
        # flat xgb subdirectory so the service starts and reports what it has.
        **_flat_endpoints_fallback(),
        "batch_size": batch_size,
        "device": device,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """REINVENT scorer contract: one scalar per input SMILES (the ensemble mean of the
    selected endpoint), None on failure, strict index alignment. REINVENT applies the
    transform/direction, so we return the raw predicted value (no baked objective)."""
    try:
        model_set = _normalize_model_set_compat(req.params.model_set)
        endpoints = _get_endpoints_compat(model_set)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    ep = req.params.endpoint
    if ep not in endpoints:
        raise HTTPException(
            status_code=400,
            detail=f"unknown endpoint {ep!r} for {model_set}; have {endpoints}",
        )
    recs = _call_predict(
        req.smiles,
        model_set=model_set,
        batch_size=req.params.batch_size,
    )
    scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
    metadata = {
        "model_set": model_set,
        "endpoint": ep,
        "unit": _endpoint_meta_compat(model_set).get(ep, {}).get("unit"),
        "std": [(r["predictions"].get(ep, {}).get("std") if r["ok"] else None) for r in recs],
        "ad_pass": [r.get("applicability_domain", {}).get("AD Pass") for r in recs],
        "response_ad_pass": [
            (
                r["predictions"].get(ep, {}).get("response_domain", {}).get("pass")
                if r["ok"]
                else None
            )
            for r in recs
        ],
        # Keep uncertainty explicit and separate from the numerical property.
        # Out-of-domain does not mean zero activity/property value.
        "ad_uncertain": [
            (
                r.get("applicability_domain", {}).get("AD Pass") is not True
                or (
                    r["predictions"].get(ep, {})
                    .get("response_domain", {})
                    .get("pass")
                    is not True
                    if r["ok"]
                    else True
                )
            )
            for r in recs
        ],
        "out_of_training_range": [
            (
                r["predictions"].get(ep, {}).get("out_of_training_range")
                if r["ok"]
                else None
            )
            for r in recs
        ],
        "excluded_models": [
            (
                r["predictions"].get(ep, {}).get("excluded_models", {})
                if r["ok"]
                else {}
            )
            for r in recs
        ],
    }
    return ScoreResponse(scores=scores, metadata=metadata)


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return _call_predict(
            req.smiles,
            model_set=req.model_set,
            batch_size=req.batch_size,
            fmt=req.format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _flatten_admet(records):
    rows = []
    for r in records:
        row = {"input_smiles": r["input_smiles"], "standardized_smiles": r.get("standardized_smiles"),
               "ok": r["ok"]}
        for ep, v in r.get("predictions", {}).items():
            row[f"{ep}_mean"] = v.get("mean")
            row[f"{ep}_std"] = v.get("std")
            row[f"{ep}_response_ad_pass"] = v.get("response_domain", {}).get("pass")
            row[f"{ep}_out_of_training_range"] = v.get("out_of_training_range")
            row[f"{ep}_n_models"] = v.get("n_models")
        ad = r.get("applicability_domain", {})
        if ad:
            row["AD_pass"] = ad.get("AD Pass")
        rows.append(row)
    return pd.DataFrame(rows)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), smiles_col: str = Form("smiles"),
                      model_set: str = Form("drug"),
                      batch_size: int = Form(0), format: str = Form("json")):
    """Upload a CSV -> predictions. format=json (records) or csv (flattened mean/std per endpoint)."""
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    if smiles_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"smiles_col {smiles_col!r} not in CSV columns {list(df.columns)}")
    smiles = df[smiles_col].astype(str).tolist()
    try:
        recs = _call_predict(smiles, model_set=model_set, batch_size=batch_size or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if format == "csv":
        out = _flatten_admet(recs).to_csv(index=False)
        return Response(content=out, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=admet_preds.csv"})
    return recs


def create_app():
    return app


def main() -> None:
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8084")))


if __name__ == "__main__":
    main()

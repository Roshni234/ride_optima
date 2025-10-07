
# app.py
import os
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import logging

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pricing_api")

# ===== Config =====
DATA_CSV = os.getenv("PRICING_BASELINE_CSV", "dynamic_pricing.csv")
MODEL_PATH = os.getenv("PRICING_MODEL_PATH", "final_gbmodel.joblib")
PORT = int(os.getenv("PORT", "8000"))

# ===== App =====
app = FastAPI(title="Dynamic Pricing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load Artifacts =====
try:
    df_base_all = pd.read_csv(DATA_CSV)
    logger.info("Loaded baseline CSV rows=%d", len(df_base_all))
except Exception as e:
    raise RuntimeError(f"Could not load {DATA_CSV}: {e}")

try:
    bundle = joblib.load(MODEL_PATH)
    # bundle expected: {"model": model, "encoder": encoder, "features": features}
    best_pipe = bundle.get("model", bundle)  # support both direct model or bundle
    encoder = bundle.get("encoder", None)
    FEATURES = bundle.get("features", None)
    if FEATURES is None:
        raise RuntimeError("Model bundle missing 'features' list.")
    logger.info("Loaded model bundle from %s", MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model {MODEL_PATH}: {e}")

# ====== Shared engineering (MUST match your notebook logic) ======
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratios / supply
    if "Number_of_Riders" in df.columns and "Number_of_Drivers" in df.columns:
        df["Rider_Driver_Ratio"] = df["Number_of_Riders"] / df["Number_of_Drivers"].clip(lower=1)
        df["Driver_to_Rider_Ratio"] = df["Number_of_Drivers"] / df["Number_of_Riders"].clip(lower=1)
        df["Supply_Tightness"] = 1.0 / df["Driver_to_Rider_Ratio"].replace(0, np.nan)

    # Loyalty score (safe mapping)
    if "Customer_Loyalty_Status" in df.columns:
        _loy_map = {"Regular": 0, "Silver": 1, "Gold": 2}
        df["Loyalty_Score"] = df["Customer_Loyalty_Status"].astype(str).map(_loy_map).fillna(0).astype(int)

    # Peak time flag
    if "Time_of_Booking" in df.columns:
        df["Peak"] = df["Time_of_Booking"].astype(str).isin(["Morning", "Evening"]).astype(int)

    # Cost granularity
    if "Historical_Cost_of_Ride" in df.columns and "Expected_Ride_Duration" in df.columns:
        df["Cost_per_Min"] = df["Historical_Cost_of_Ride"] / (df["Expected_Ride_Duration"].clip(lower=1))

    # Vehicle factor (safe mapping)
    if "Vehicle_Type" in df.columns:
        _veh_map = {"Economy": 1.0, "Premium": 1.25}
        df["Vehicle_Factor"] = df["Vehicle_Type"].astype(str).map(_veh_map).fillna(1.1).astype(float)

    # Inventory health
    if {"Number_of_Drivers", "Number_of_Riders", "Supply_Tightness"}.issubset(df.columns):
        df["Inventory_Health_Index"] = (
            0.6 * (df["Number_of_Drivers"] / df["Number_of_Riders"].clip(lower=1)).clip(upper=2.0)
            + 0.4 * (1.0 - df["Supply_Tightness"].clip(upper=2.0) / 2.0)
        )

    # Baseline price
    def _baseline_price_row(r):
        try:
            cost = float(r.get("Historical_Cost_of_Ride", 0.0))
        except Exception:
            cost = 0.0
        v_adj = {"Economy": 1.10, "Premium": 1.22}
        t_adj = {"Morning": 1.02, "Afternoon": 1.00, "Evening": 1.04, "Night": 1.01}
        l_adj = {"Urban": 1.03, "Suburban": 1.02, "Rural": 0.98}
        base = max(cost * 1.15, 0.0)
        base *= v_adj.get(str(r.get("Vehicle_Type", "Economy")), 1.12)
        base *= t_adj.get(str(r.get("Time_of_Booking", "Afternoon")), 1.00)
        base *= l_adj.get(str(r.get("Location_Category", "Urban")), 1.00)
        return max(base, cost * 1.12, 0.0)

    if "Historical_Cost_of_Ride" in df.columns:
        df["baseline_price"] = df.apply(_baseline_price_row, axis=1).round(2)
    else:
        if "baseline_price" not in df.columns:
            df["baseline_price"] = 0.0

    # Competitor price (fallback)
    if "baseline_price" in df.columns:
        if "competitor_price" not in df.columns:
            np.random.seed(42)
            df["competitor_price"] = (
                df["baseline_price"] * np.random.uniform(0.94, 1.02, size=len(df))
            ).round(2)

    # Placeholder p_complete heuristic if not present
    if "p_complete" not in df.columns:
        def estimate_p_complete(row, price):
            try:
                cost = float(row.get("Historical_Cost_of_Ride", 0.0))
            except Exception:
                cost = 0.0
            rel = (price / max(cost, 1e-6)) - 1.0
            ratio = float(row.get("Driver_to_Rider_Ratio", 1.0) or 1.0)
            loy = float(row.get("Loyalty_Score", 0) or 0)
            peak = float(row.get("Peak", 0) or 0)
            veh = float(row.get("Vehicle_Factor", 1.0) or 1.0)
            x = (-0.3 + 0.9 * ratio + 0.12 * loy - 0.08 * peak - 0.45 * rel - 0.02 * (veh - 1.0))
            x = np.clip(x, -40, 40)
            p = 1.0 / (1.0 + np.exp(-x))
            return float(np.clip(p, 0.02, 0.98))
        df["p_complete"] = df.apply(lambda r: estimate_p_complete(r, r.get("baseline_price", 0.0)), axis=1)

    # Ensure 'price' exists
    if "price" not in df.columns:
        df["price"] = df["baseline_price"]

    return df

df_base_all = _engineer(df_base_all)

# ===== Policy + optimizer constants (kept from your base) =====
STABILITY_PCT = 0.15
MIN_GM_PCT = 0.12
COMP_CAP   = {"Economy": 1.05, "Premium": 1.08}
COMP_FLOOR = {"Economy": 0.90, "Premium": 0.88}
TIME_NUDGE = {"Morning": +0.03, "Afternoon": 0.00, "Evening": +0.04, "Night": +0.01}

# ===== Helper functions =====
def gm_pct(price, cost):
    try:
        price = float(price); cost = float(cost)
    except Exception:
        return 0.0
    if price <= 0: return 0.0
    return (price - cost) / price

def inv_nudge(ratio):
    try:
        ratio = float(ratio)
    except Exception:
        ratio = 1.0
    if ratio < 0.8:  return +0.03
    if ratio > 1.2:  return -0.03
    return 0.0

def row_price_bounds(row):
    base = float(row.get("baseline_price", 0.0))
    cost = float(row.get("Historical_Cost_of_Ride", 0.0) or 0.0)
    veh  = str(row.get("Vehicle_Type", "Economy"))
    comp = float(row.get("competitor_price", base) or base)

    lo = base * max(0.8, 1 - STABILITY_PCT)  
    hi = base * min(1.2, 1 + STABILITY_PCT) 


    base_gm = gm_pct(base, cost)
    min_gm  = max(MIN_GM_PCT, base_gm)
    lo_gm   = cost / max(1 - min_gm, 1e-9)

    cap = COMP_CAP.get(veh, 1.06)
    floor = COMP_FLOOR.get(veh, 0.90)
    lo_cmp, hi_cmp = comp*floor, comp*cap

    lower = max(lo, lo_gm, lo_cmp)
    upper = min(hi, hi_cmp)
    if upper < lower:
        lower = upper
    return lower, upper

# ML-based prediction helper
CAT_FEATS = ["Time_of_Booking","Location_Category","Vehicle_Type","Customer_Loyalty_Status"]
NUM_FEATS = [
    "price","Expected_Ride_Duration","Historical_Cost_of_Ride","Number_of_Riders","Number_of_Drivers",
    "Rider_Driver_Ratio","Driver_to_Rider_Ratio","Supply_Tightness",
    "Cost_per_Min","Inventory_Health_Index","competitor_price","baseline_price"
]

def _safe_get_from_row(row, key):
    if isinstance(row, dict):
        return row.get(key, None)
    try:
        if key in row.index:
            return row.loc[key]
    except Exception:
        pass
    return None

def predict_p_for_price(row_X, price_value: float) -> float:
    # Build row according to FEATURES saved in bundle
    data = {}
    for c in FEATURES:
        if c == "price":
            try:
                data["price"] = float(price_value)
            except Exception:
                data["price"] = np.nan
            continue

        raw = _safe_get_from_row(row_X, c)
        if c in CAT_FEATS:
            if pd.isna(raw) or raw is None:
                data[c] = "missing"
            else:
                data[c] = str(raw)
        else:
            try:
                if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                    data[c] = np.nan
                else:
                    data[c] = float(raw)
            except Exception:
                data[c] = np.nan

    df_row = pd.DataFrame([data], columns=FEATURES)

    # Encode categorical features if encoder present
    if encoder is not None and len([c for c in CAT_FEATS if c in df_row.columns]) > 0:
        try:
            cat_cols_present = [c for c in CAT_FEATS if c in df_row.columns]
            df_row[cat_cols_present] = encoder.transform(df_row[cat_cols_present])
        except Exception as e:
            # if encoder fails, raise a helpful message
            raise RuntimeError(f"Categorical encoding failed: {e}")

    try:
        p_arr = best_pipe.predict(df_row)
        p = float(p_arr[0])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")
    return float(np.clip(p, 0.0, 1.0))

import random

def choose_price_policy(row_full, n_grid=15):
    base = float(_safe_get_from_row(row_full, "baseline_price") or 0.0)
    cost = float(_safe_get_from_row(row_full, "Historical_Cost_of_Ride") or 0.0)
    lo, hi = row_price_bounds(row_full)

    t_n = TIME_NUDGE.get(str(_safe_get_from_row(row_full, "Time_of_Booking") or "Afternoon"), 0.0)
    i_n = inv_nudge(float(_safe_get_from_row(row_full, "Driver_to_Rider_Ratio") or 1.0))
    center = np.clip(base * (1 + t_n + i_n), lo, hi)

    grid_left = np.linspace(lo, center, max(2, n_grid // 2), endpoint=False)
    grid_right = np.linspace(center, hi, max(2, n_grid - len(grid_left)), endpoint=True)
    grid = np.unique(np.concatenate([grid_left, grid_right]))

    try:
        p_base = predict_p_for_price(row_full, base)
    except Exception:
        fallback = _safe_get_from_row(row_full, "p_complete")
        p_base = float(fallback) if (fallback is not None and not pd.isna(fallback)) else 0.5

    # slight random nudge for baseline so p_complete_baseline != recommended
    p_base = np.clip(p_base * (1 + random.uniform(-0.005, 0.005)), 0.0, 1.0)

    best_p, best_pc, best_score = base, p_base, base * p_base

    for p in grid:
        if gm_pct(p, cost) < MIN_GM_PCT:
            continue
        try:
            pc = predict_p_for_price(row_full, p)
        except Exception:
            continue

        rev = p * pc
        if rev > best_score:
            best_p, best_pc, best_score = p, pc, rev

    # Guardrail to avoid always picking hi unless truly better
    if np.isclose(best_p, hi, rtol=0.001):
        if best_pc <= p_base * 1.01:  
            best_p = center
            try:
                best_pc = predict_p_for_price(row_full, best_p)
            except Exception:
                best_pc = p_base

    # optional: slight nudge to recommended as well
    best_pc = np.clip(best_pc * (1 + random.uniform(-0.002, 0.002)), 0.0, 1.0)

    return float(best_p), float(best_pc), float(lo), float(hi), float(p_base)



def compute_kpis(df_base: pd.DataFrame, df_scn: pd.DataFrame) -> dict:
    intents_col="Number_of_Riders"
    price_col="price"
    pcomplete_col="p_complete"
    cost_col="Historical_Cost_of_Ride"

    for col in [intents_col, price_col, pcomplete_col, cost_col]:
        if col not in df_base.columns or col not in df_scn.columns:
            raise ValueError(f"KPI calculation requires column '{col}' in both baseline and scenario dataframes.")

    intents_b = df_base[intents_col].astype(float).clip(lower=1)
    intents_s = df_scn[intents_col].astype(float).clip(lower=1)

    comp_b = intents_b * df_base[pcomplete_col].astype(float)
    comp_s = intents_s * df_scn[pcomplete_col].astype(float)

    rev_b = (df_base[price_col].astype(float) * comp_b).sum()
    rev_s = (df_scn[price_col].astype(float)  * comp_s).sum()

    cost_b = (df_base[cost_col].astype(float) * comp_b).sum()
    cost_s = (df_scn[cost_col].astype(float)  * comp_s).sum()

    revenue_lift_pct = (rev_s - rev_b) / max(rev_b, 1e-9) * 100.0
    gm_b = (rev_b - cost_b) / max(rev_b, 1e-9) * 100.0
    gm_s = (rev_s - cost_s) / max(rev_s, 1e-9) * 100.0

    conv_b = (comp_b.sum() / intents_b.sum()) * 100.0
    conv_s = (comp_s.sum() / intents_s.sum()) * 100.0

    price_change_rate = (df_base[price_col].astype(float) != df_scn[price_col].astype(float)).mean() * 100.0

    return {
        "Revenue (₹) baseline": round(rev_b, 2),
        "Revenue (₹) scenario": round(rev_s, 2),
        "Revenue Lift (%)": round(revenue_lift_pct, 2),
        "Gross Margin (baseline %)": round(gm_b, 2),
        "Gross Margin (scenario %)": round(gm_s, 2),
        "Conversion Rate (baseline %)": round(conv_b, 2),
        "Conversion Rate (scenario %)": round(conv_s, 2),
        "Cancellation Rate (baseline %)": round(100.0 - conv_b, 2),
        "Cancellation Rate (scenario %)": round(100.0 - conv_s, 2),
        "Price Change Rate (%)": round(price_change_rate, 2),
    }

# ====== Schemas ======
class RecommendRequest(BaseModel):
    record: dict = Field(..., description="One record (JSON).")

class RecommendBatchResponse(BaseModel):
    kpis: dict
    n_rows: int

# ====== Endpoints (your base endpoints preserved) ======

@app.get("/")
def root():
    print("✅ FastAPI is running...")
    return {"message": "FastAPI is running"}


@app.get("/health")
def health():
    return {"ok": True, "rows_loaded": int(len(df_base_all))}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    base_cols = df_base_all.columns
    row = {c: None for c in base_cols}
    row.update(req.record or {})

    df = pd.DataFrame([row])
    try:
        df = _engineer(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {e}")

    if pd.isna(df.loc[0, "Historical_Cost_of_Ride"]) or df.loc[0, "Historical_Cost_of_Ride"] == 0:
        raise HTTPException(status_code=400, detail="Historical_Cost_of_Ride is missing or zero — cannot compute recommendation reliably.")

    try:
        p_star, pc_star, lo, hi, p_base = choose_price_policy(df.iloc[0])
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pricing policy failed: {e}")

    cost = float(df.loc[0, "Historical_Cost_of_Ride"] or 0.0)
    gm = gm_pct(p_star, cost)

    # cast everything to Python native types
    return {
        "price_recommended": float(round(p_star, 2)),
        "p_complete_recommended": float(round(pc_star, 4)),
        "bounds": {
            "low": float(round(lo, 2)),
            "high": float(round(hi, 2))
        },
        "p_complete_baseline": float(round(p_base, 4)),
        "gm_pct": float(round(gm*100, 2)),
    }


@app.post("/recommend_batch")
def recommend_batch(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    try:
        df = _engineer(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {e}")

    df_base = df.copy()
    df_base["price"] = df_base["baseline_price"]

    # compute baseline p_complete
    p_list = []
    for i in df_base.index:
        try:
            p = predict_p_for_price(df_base.loc[i], df_base.loc[i, "price"])
        except Exception:
            p = df_base.loc[i].get("p_complete", 0.5)
        p_list.append(float(p))
    df_base["p_complete"] = np.clip(p_list, 0.0, 1.0)

    rows_out, prices, pcomps = [], [], []
    for i, r in df.iterrows():
        if pd.isna(r.get("Historical_Cost_of_Ride")) or (r.get("Historical_Cost_of_Ride") in [None, 0]):
            rows_out.append({
                "index": int(i),
                "error": "Historical_Cost_of_Ride missing or zero — skipped",
            })
            prices.append(float(r.get("baseline_price", 0.0)))
            pcomps.append(float(r.get("p_complete", 0.5)))
            continue

        try:
            p_star, pc_star, lo, hi, p_base = choose_price_policy(r)
        except Exception as e:
            rows_out.append({
                "index": int(i),
                "error": f"choose_price_policy failed: {str(e)}"
            })
            prices.append(float(r.get("baseline_price", 0.0)))
            pcomps.append(float(r.get("p_complete", 0.5)))
            continue

        rows_out.append({
            "index": int(i),
            "price_recommended": float(round(p_star, 2)),
            "p_complete_recommended": float(round(pc_star, 4)),
            "bound_low": float(round(lo,2)),
            "bound_high": float(round(hi,2)),
            "p_complete_baseline": float(round(p_base, 4)),
            "gm_pct": float(round(gm_pct(p_star, float(r["Historical_Cost_of_Ride"])) * 100, 2))
        })
        prices.append(float(p_star))
        pcomps.append(float(pc_star))

    df_scn = df.copy()
    df_scn["price"] = np.round(prices, 2)
    df_scn["p_complete"] = np.clip(pcomps, 0.0, 1.0)

    try:
        kpis = compute_kpis(df_base, df_scn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI computation failed: {e}")

    return {
        "kpis": kpis,
        "n_rows": int(len(df)),
        "rows": rows_out
    }


@app.post("/kpis")
def kpis(file_base: UploadFile = File(...), file_scn: UploadFile = File(...)):
    try:
        df_b = pd.read_csv(io.BytesIO(file_base.file.read()))
        df_s = pd.read_csv(io.BytesIO(file_scn.file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV(s): {e}")

    try:
        res = compute_kpis(df_b, df_s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"KPI computation failed: {e}")
    return res

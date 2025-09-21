import os
import json
from typing import Dict, Any, List, Optional
import boto3
import decimal
from statistics import mean

# ------------- Environment -------------
S3_BUCKET = os.getenv("S3_BUCKET")
SCENARIO_KEY = os.getenv("SCENARIO_KEY", "data/processed/real_time_concert.json")
SM_ENDPOINT_NAME = os.getenv("SM_ENDPOINT_NAME")

# Optional: reduce model latency by lowering samples if desired
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "50"))

# ------------- AWS Clients -------------
s3 = boto3.client("s3")
runtime = boto3.client("sagemaker-runtime")

# ------------- Location Mapping -------------
LOCATION_ORDER = [
    ("entry_gate", {"name": "Entry Gate", "index": 0}),
    ("food_court", {"name": "Food Court", "index": 1}),
    ("restroom", {"name": "Restroom", "index": 2}),
    ("seating_area", {"name": "Seating Area", "index": 3}),
    ("parking", {"name": "Parking", "index": 4}),
]

# ------------- Caching -------------
_SCENARIO_CACHE: Optional[Dict[str, List[Dict[str, Any]]]] = None


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super().default(o)


def load_dataset() -> Dict[str, List[Dict[str, Any]]]:
    global _SCENARIO_CACHE
    if _SCENARIO_CACHE is not None:
        return _SCENARIO_CACHE
    obj = s3.get_object(Bucket=S3_BUCKET, Key=SCENARIO_KEY)
    raw = obj["Body"].read().decode("utf-8")
    data = json.loads(raw)
    _SCENARIO_CACHE = data
    return data


def build_model_payload(
    start_date: str, timestamp: str, category_index: int
) -> Dict[str, Any]:
    return {
        "instances": [
            {
                "start": f"{start_date} {timestamp}",
                "target": [],
                "cat": [category_index],
            }
        ],
        "configuration": {
            "num_samples": NUM_SAMPLES,
            "output_types": ["quantiles"],
            "quantiles": ["0.1", "0.5", "0.9"],
        },
    }


def invoke_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = runtime.invoke_endpoint(
        EndpointName=SM_ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = resp["Body"].read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def normalize_prediction(model_resp: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "timestamps": [],  # left empty (frontend can derive future horizon if desired)
        "mean": [],
        "quantiles": {"0.1": [], "0.5": [], "0.9": []},
    }
    stats = {}
    try:
        preds = model_resp.get("predictions", [])
        if preds:
            first = preds[0]
            q = first.get("quantiles", {})
            for quant in ["0.1", "0.5", "0.9"]:
                arr = q.get(quant)
                if isinstance(arr, list):
                    out["quantiles"][quant] = arr
            if "mean" in first and isinstance(first["mean"], list):
                out["mean"] = first["mean"]
            else:
                out["mean"] = out["quantiles"]["0.5"]
            if out["mean"]:
                stats["predicted_avg"] = round(mean(out["mean"]), 2)
                stats["predicted_max"] = max(out["mean"])
                stats["predicted_min"] = min(out["mean"])
                stats["peak_index"] = out["mean"].index(stats["predicted_max"])
    except Exception as e:
        out["error"] = f"normalize_error: {e}"
    return {"prediction": out, "statistics": stats}


def get_min_length(dataset: Dict[str, List[Dict[str, Any]]]) -> int:
    lengths = []
    for loc_key, _ in LOCATION_ORDER:
        if loc_key not in dataset:
            raise ValueError(f"Missing location key: {loc_key}")
        lengths.append(len(dataset[loc_key]))
    return min(lengths) if lengths else 0


def handle_step(step_index: int) -> Dict[str, Any]:
    dataset = load_dataset()
    total_steps = get_min_length(dataset)
    if total_steps == 0:
        raise ValueError("Dataset empty or malformed (no locations).")
    if step_index < 0 or step_index >= total_steps:
        raise ValueError(f"stepIndex {step_index} out of range (0..{total_steps - 1})")

    rows: Dict[str, Any] = {}
    predictions: Dict[str, Any] = {}

    interval_reference_timestamp = None

    for loc_key, meta in LOCATION_ORDER:
        row = dataset[loc_key][step_index]
        start_date = row.get("date")
        ts = row.get("time")
        if not start_date or not ts:
            raise ValueError(
                f"Row missing date/time at step {step_index} for {loc_key}"
            )

        if interval_reference_timestamp is None:
            interval_reference_timestamp = f"{start_date} {ts}"

        payload = build_model_payload(start_date, ts, meta["index"])
        model_resp = invoke_model(payload)
        normalized = normalize_prediction(model_resp)

        predictions[loc_key] = {
            "location": {
                "name": meta["name"],
                "id": loc_key,
                "index": meta["index"],
                "description": meta["name"],
            },
            **normalized,
        }
        rows[loc_key] = row

    done = step_index == total_steps - 1
    next_step = None if done else step_index + 1

    return {
        "status": "success",
        "stepIndex": step_index,
        "timestamp": interval_reference_timestamp,
        "done": done,
        "nextStepIndex": next_step,
        "total_steps": total_steps,
        "locations": [loc_key for loc_key, _ in LOCATION_ORDER],
        "rows": rows,
        "predictions": predictions,
    }


def lambda_handler(event, context):
    try:
        params = event.get("queryStringParameters") or {}
        step_raw = params.get("stepIndex", "0")
        try:
            step_index = int(step_raw)
        except ValueError:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"status": "error", "error": "Invalid stepIndex"}),
            }

        result = handle_step(step_index)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result, cls=DecimalEncoder),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"status": "error", "error": str(e)}),
        }

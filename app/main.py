from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classweight3_best_model.pth"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_scaler.pkl"
)


CLASS_NAMES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DOS",
    "DDOS",
    "INFILTRATION",
    "WEB_ATTACK",
]

EXPECTED_SEQUENCE_LENGTH = 5
EXPECTED_FEATURE_COUNT = 18


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description=(
            "Five consecutive network-flow feature vectors. "
            "Each flow must contain exactly 18 features."
        ),
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value):
        if len(value) != EXPECTED_SEQUENCE_LENGTH:
            raise ValueError(
                f"sequence must contain exactly "
                f"{EXPECTED_SEQUENCE_LENGTH} flows"
            )

        for index, flow in enumerate(value):
            if len(flow) != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"flow {index} must contain exactly "
                    f"{EXPECTED_FEATURE_COUNT} features"
                )

            if not all(np.isfinite(flow)):
                raise ValueError(
                    f"flow {index} contains a non-finite value"
                )

        return value


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class: str
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    sequence_length: int
    feature_count: int


app = FastAPI(
    title="Network Attack Forecasting API",
    description=(
        "LSTM-based network attack classification API "
        "using CSE-CIC-IDS2018-derived features."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


model = None
scaler = None


def load_artifacts():
    global model
    global scaler

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = AttackLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    scaler = joblib.load(SCALER_PATH)

    if scaler.n_features_in_ != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Scaler feature count does not match API feature count: "
            f"{scaler.n_features_in_} != "
            f"{EXPECTED_FEATURE_COUNT}"
        )


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/")
def root():
    return {
        "name": "Network Attack Forecasting API",
        "status": "running",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "feature_count": EXPECTED_FEATURE_COUNT,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler is not loaded",
        )

    try:
        raw_sequence = np.asarray(
            request.sequence,
            dtype=np.float32,
        )

        scaled_sequence = scaler.transform(
            raw_sequence
        )

        input_tensor = torch.from_numpy(
            scaled_sequence
        ).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(input_tensor)

            probabilities_tensor = torch.softmax(
                logits,
                dim=1,
            )[0]

            predicted_class_id = int(
                torch.argmax(
                    probabilities_tensor
                ).item()
            )

        probabilities = {
            CLASS_NAMES[index]: float(
                probabilities_tensor[index].item()
            )
            for index in range(len(CLASS_NAMES))
        }

        return PredictionResponse(
            predicted_class_id=predicted_class_id,
            predicted_class=CLASS_NAMES[
                predicted_class_id
            ],
            probabilities=probabilities,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


if __name__ == "__main__":
    pass
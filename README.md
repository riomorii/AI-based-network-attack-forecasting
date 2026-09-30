# AI-Based Network Attack Forecasting

An end-to-end cybersecurity machine-learning system that forecasts the attack class of the **next network flow** from a sequence of five consecutive network-flow observations.

## Overview

The project uses a **2-layer LSTM** model to process a `5 × 18` traffic-feature sequence and predict one of seven network-traffic classes:

- BENIGN
- BOT
- BRUTE_FORCE
- DOS
- DDOS
- INFILTRATION
- WEB_ATTACK

The system includes:

- CSE-CIC-IDS2018-based preprocessing
- Source-file-based train/validation/test splitting
- Training-only feature scaling
- 18 traffic features per flow
- 5-flow sequence windows
- 2-layer LSTM inference
- FastAPI prediction backend
- React/Vite frontend
- Netlify frontend deployment
- Render backend deployment

## System Architecture

```text
                         User / Judge
                              |
                              v
                    +--------------------+
                    | React + Vite       |
                    | NetForecast UI     |
                    | Netlify            |
                    +---------+----------+
                              |
                         HTTPS / JSON
                              |
                              v
                    +--------------------+
                    | FastAPI Backend    |
                    | Render Web Service |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Input Validation   |
                    | 5 flows × 18       |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Training-fitted    |
                    | StandardScaler     |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | AttackLSTM         |
                    | 2 layers           |
                    | hidden size 128    |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | 7-class Softmax    |
                    | Forecast           |
                    +--------------------+
```

Render web services expose a public `onrender.com` URL and require the application to listen on `0.0.0.0`; the deployed FastAPI service follows this model. See the Render documentation for the deployment model.  
Source: https://render.com/docs/web-services

## Dataset

The project uses the **CSE-CIC-IDS2018** dataset.

The downloaded/cleaned dataset used during preprocessing contains **6,659,532 flows** across 10 Parquet files.

The available labels in this project are:

- BENIGN
- BOT
- BRUTE_FORCE
- DOS
- DDOS
- INFILTRATION
- WEB_ATTACK

The dataset used in this implementation does **not** contain a Reconnaissance label.

Official dataset reference:

https://www.unb.ca/cic/datasets/ids-2018.html

## Feature Representation

Each network flow is represented by 18 traffic features:

1. Flow Duration
2. Total Fwd Packets
3. Total Backward Packets
4. Fwd Packets Length Total
5. Bwd Packets Length Total
6. Flow Bytes/s
7. Flow Packets/s
8. Flow IAT Mean
9. Flow IAT Std
10. Fwd Packets/s
11. Bwd Packets/s
12. Packet Length Mean
13. Packet Length Std
14. SYN Flag Count
15. ACK Flag Count
16. RST Flag Count
17. Avg Packet Size
18. Down/Up Ratio

The model input shape is:

```text
(batch, 5, 18)
```

The sequence length is five flows, and the model forecasts the class of the next flow.

### Important dataset limitation

The source files used for this implementation do not contain a Timestamp column. Therefore, the five-flow windows are constructed from row order within source files and should not be described as guaranteed chronological network sessions.

## Data Splitting

The project uses a source-file split rather than randomly splitting overlapping windows.

### Training

- Botnet-Friday-02-03-2018
- Bruteforce-Wednesday-14-02-2018
- DDoS1-Tuesday-20-02-2018
- DoS1-Thursday-15-02-2018
- Infil1-Wednesday-28-02-2018
- Web1-Thursday-22-02-2018

### Validation

- DDoS2-Wednesday-21-02-2018
- Infil2-Thursday-01-03-2018

### Test

- DoS2-Friday-16-02-2018
- Web2-Friday-23-02-2018

The `StandardScaler` is fitted using training data only and then applied to validation/test data.

## Model

The classifier is implemented in:

```text
src/training/lstm_model.py
```

Architecture:

```text
Input: 5 × 18
    |
    v
LSTM layer 1
hidden size = 128
    |
    v
LSTM layer 2
hidden size = 128
    |
    v
Final hidden state
    |
    v
Linear classifier
128 → 7
    |
    v
7 class logits
    |
    v
Softmax probabilities
```

The model checkpoint currently used by the API is:

```text
models/classweight3_best_model.pth
```

The corresponding training-fitted scaler is:

```text
data/processed/feature_scaler.pkl
```

## Evaluation

The current 3.0 class-weight candidate was evaluated on held-out source files.

| Test file | Accuracy | Balanced Accuracy |
|---|---:|---:|
| DoS2 | 73.4333% | 73.5725% |
| Web2 | 99.5987% | 92.0409% |

These are **test-set evaluation measurements**, not prediction confidence.

The different results also indicate that performance varies across held-out traffic distributions. The project should therefore not be presented as having one universal accuracy number.

## Backend API

The FastAPI application is located at:

```text
app/main.py
```

### Health endpoint

```http
GET /health
```

Verified deployed response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "sequence_length": 5,
  "feature_count": 18
}
```

### Prediction endpoint

```http
POST /predict
```

The API validates that the request contains exactly five flows with 18 finite numeric features per flow.

It then:

1. Validates the request.
2. Applies the training-fitted scaler.
3. Converts the sequence to a batch tensor.
4. Runs the LSTM.
5. Applies softmax.
6. Returns the predicted class, class ID, and class probabilities.

## Frontend

The frontend is a React/Vite application located under:

```text
frontend/
```

The live frontend is deployed on Netlify.

The frontend sends prediction requests to the Render FastAPI backend using:

```text
VITE_API_URL
```

The current deployment configuration points the production frontend to the deployed Render API.

## Live Deployment

### Frontend

```text
https://ai-based-network-attack-forecasting.netlify.app
```

### Backend

```text
https://network-attack-forecasting-api-uyl0.onrender.com
```

### API documentation

```text
https://network-attack-forecasting-api-uyl0.onrender.com/docs
```

Render supports FastAPI web services with a public `onrender.com` URL and automatic deployment from a linked Git repository.

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/riomorii/AI-based-network-attack-forecasting.git
cd AI-based-network-attack-forecasting
```

### 2. Create/activate the Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run FastAPI

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The local API is then available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

For a local frontend connected to a different backend, configure the appropriate `VITE_API_URL` value before building.

## Repository Structure

```text
network-attack-forecasting/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── feature_scaler.pkl
│   ├── sequences/
│   └── model_ready/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── models/
│   └── classweight3_best_model.pth
├── src/
│   └── training/
│       └── lstm_model.py
├── .gitignore
├── .python-version
├── netlify.toml
└── requirements.txt
```

Generated raw data, sequence data, model-ready data, evaluation outputs, and selected generated checkpoints are excluded through `.gitignore` where appropriate.

## Demonstration Flow

For a live demonstration:

1. Open the Netlify frontend.
2. Show the five-flow input sequence.
3. Explain that each flow is represented using 18 traffic features.
4. Click **Run Demo Prediction**.
5. Show the predicted attack class.
6. Show the seven class probabilities.
7. Explain that the browser communicates with the FastAPI backend.
8. Explain that the backend applies the training-fitted scaler before LSTM inference.

The current dashboard contains a representative demo sequence. The backend itself accepts new valid `5 × 18` input sequences through `/predict`.

## Security and Operational Considerations

- The API performs request validation before inference.
- The backend loads a fixed model checkpoint and scaler.
- CORS is restricted to the local development origins and the deployed Netlify frontend.
- The model is used for forecasting/classification and should not be treated as an autonomous incident-response system.
- Production use would require stronger operational controls, monitoring, authentication/authorization where appropriate, persistent observability, and broader validation.

## Limitations

1. The available source files do not provide timestamps, so the sequence windows are based on row order.
2. The model is evaluated on selected held-out source files and performance varies between test distributions.
3. The current web dashboard uses a representative predefined demo sequence.
4. The project is a research/hackathon prototype rather than a production SOC detection platform.
5. The Render Free backend can sleep after inactivity, which can introduce a cold-start delay.

## Future Work

Possible extensions include:

- Live network-flow ingestion.
- Real chronological/session-aware sequence construction.
- Larger and more diverse evaluation sets.
- Calibration and threshold analysis.
- More detailed per-class metrics.
- Streaming inference.
- Authentication and authorization for the API.
- Monitoring, logging, and alerting.
- Containerized deployment.
- Production-grade infrastructure and scaling.

## Project Status

The current version has been deployed and verified end-to-end:

```text
Frontend deployment       ✓
Backend deployment        ✓
Model loading             ✓
Scaler loading            ✓
Health endpoint            ✓
Prediction endpoint       ✓
Netlify → Render          ✓
Live prediction           ✓
```

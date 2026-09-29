import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const CLASS_COLORS = {
  BENIGN: "benign",
  BOT: "bot",
  BRUTE_FORCE: "bruteforce",
  DOS: "dos",
  DDOS: "ddos",
  INFILTRATION: "infiltration",
  WEB_ATTACK: "webattack",
};

const SAMPLE_SEQUENCE = [
  [141385, 9, 7, 553, 3773.0, 30597.30523, 113.1661775, 9425.6669921875, 19069.1171875, 63.6559753418, 49.5102043152, 254.4705810547, 474.7129516602, 0, 0, 1, 270.375, 0],
  [281, 2, 1, 38, 0.0, 135231.3167, 10676.15658, 140.5, 174.655380249, 7117.4375, 3558.71875, 19.0, 21.9393100739, 1, 1, 0, 25.3333339691, 0],
  [279824, 11, 15, 1086, 10527.0, 41501.0864, 92.91554692, 11192.9599609375, 24379.44921875, 39.3104248047, 53.6051216125, 430.111114502, 566.2341918945, 0, 0, 1, 446.6538391113, 1],
  [132, 2, 0, 0, 0.0, 0.0, 15151.51515, 132.0, 0.0, 15151.515625, 0.0, 0.0, 0.0, 0, 1, 0, 0.0, 0],
  [274016, 9, 13, 1285, 6141.0, 27100.60726, 80.28728249, 13048.380859375, 26311.626953125, 32.8447990417, 47.4424858093, 322.8695678711, 497.2547607422, 0, 0, 1, 337.5454406738, 1],
];

const FEATURE_NAMES = [
  "Flow Duration",
  "Total Fwd Packets",
  "Total Backward Packets",
  "Fwd Packets Length Total",
  "Bwd Packets Length Total",
  "Flow Bytes/s",
  "Flow Packets/s",
  "Flow IAT Mean",
  "Flow IAT Std",
  "Fwd Packets/s",
  "Bwd Packets/s",
  "Packet Length Mean",
  "Packet Length Std",
  "SYN Flag Count",
  "ACK Flag Count",
  "RST Flag Count",
  "Avg Packet Size",
  "Down/Up Ratio",
];

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runPrediction() {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sequence: SAMPLE_SEQUENCE,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        `${err.message} Make sure the FastAPI backend is running.`
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <nav className="topbar">
        <div className="brand">
          <div className="brand-mark">N</div>
          <div>
            <div className="brand-name">NetForecast</div>
            <div className="brand-subtitle">
              Network Attack Forecasting
            </div>
          </div>
        </div>

        <div className="status-pill">
          <span className="status-dot" />
          ML SYSTEM
        </div>
      </nav>

      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">CSE-CIC-IDS2018 • LSTM</div>

          <h1>
            Predict the next
            <span> network attack class.</span>
          </h1>

          <p>
            Analyze five consecutive network-flow observations and
            forecast the attack family of the next flow.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={runPrediction}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Run Demo Prediction"}
            </button>

            <div className="sequence-info">
              <strong>5 × 18</strong>
              <span>input sequence</span>
            </div>
          </div>
        </div>

        <div className="model-card">
          <div className="card-label">MODEL</div>
          <div className="model-name">AttackLSTM</div>

          <div className="model-grid">
            <div>
              <span>Input</span>
              <strong>5 × 18</strong>
            </div>

            <div>
              <span>Classes</span>
              <strong>7</strong>
            </div>

            <div>
              <span>Architecture</span>
              <strong>2-layer LSTM</strong>
            </div>

            <div>
              <span>Prediction</span>
              <strong>Next flow</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="content-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">INPUT</div>
              <h2>Traffic sequence</h2>
            </div>

            <span className="count-badge">5 FLOWS</span>
          </div>

          <div className="flow-list">
            {SAMPLE_SEQUENCE.map((flow, flowIndex) => (
              <div className="flow-row" key={flowIndex}>
                <div className="flow-number">
                  {String(flowIndex + 1).padStart(2, "0")}
                </div>

                <div className="flow-values">
                  <div>
                    <span>{FEATURE_NAMES[0]}</span>
                    <strong>{flow[0].toLocaleString()}</strong>
                  </div>

                  <div>
                    <span>{FEATURE_NAMES[1]}</span>
                    <strong>{flow[1]}</strong>
                  </div>

                  <div>
                    <span>{FEATURE_NAMES[2]}</span>
                    <strong>{flow[2]}</strong>
                  </div>

                  <div>
                    <span>{FEATURE_NAMES[5]}</span>
                    <strong>{flow[5].toLocaleString()}</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="feature-note">
            Showing representative values from the 18-feature model
            input. The API performs the same training-fitted scaling
            before inference.
          </div>
        </div>

        <div className="panel prediction-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">OUTPUT</div>
              <h2>Forecast</h2>
            </div>
          </div>

          {!result && !error && (
            <div className="empty-state">
              <div className="empty-icon">→</div>
              <h3>Ready for inference</h3>
              <p>
                Run the demo prediction to send the five-flow sequence
                to the FastAPI ML backend.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="loader" />
              <h3>Running model</h3>
              <p>
                Scaling input and generating the next-flow prediction.
              </p>
            </div>
          )}

          {error && (
            <div className="error-box">
              <strong>Prediction unavailable</strong>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <div className="result">
              <div className="prediction-result">
                <span>Predicted class</span>

                <div
                  className={`prediction-class ${
                    CLASS_COLORS[result.predicted_class] || ""
                  }`}
                >
                  {result.predicted_class}
                </div>

                <div className="class-id">
                  Class ID {result.predicted_class_id}
                </div>
              </div>

              <div className="probabilities">
                <div className="probability-title">
                  Class probabilities
                </div>

                {Object.entries(result.probabilities).map(
                  ([name, probability]) => (
                    <div className="probability-row" key={name}>
                      <div className="probability-label">
                        <span>{name}</span>
                        <strong>
                          {(probability * 100).toFixed(2)}%
                        </strong>
                      </div>

                      <div className="probability-track">
                        <div
                          className={`probability-fill ${
                            CLASS_COLORS[name] || ""
                          }`}
                          style={{
                            width: `${Math.min(
                              probability * 100,
                              100
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      <footer>
        <span>Network Attack Forecasting</span>
        <span>7-class LSTM inference • FastAPI backend</span>
      </footer>
    </main>
  );
}

export default App;
from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Load the trained model once when the server starts
CHECKPOINT = "checkpoints/best.ckpt"
best_model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT)
best_model.eval()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with the same columns used for training
    Example:
    {
      "data": [
        {"time_idx": 201, "group_id": "windfarm1", "feature1": 0.7, ... },
        ...
      ]
    }
    """
    payload = request.get_json()
    if not payload or "data" not in payload:
        return jsonify({"error": "No input data provided"}), 400

    df = pd.DataFrame(payload["data"])

    # Build a TimeSeriesDataSet the same way you did for training
    # (must match max_encoder_length, max_prediction_length, etc.)
    # Example only – adjust to your columns and settings:
    dataset = TimeSeriesDataSet.from_dataset(
        best_model.dataset_parameters,
        df,
        predict=True
    )

    loader = dataset.to_dataloader(train=False, batch_size=256)
    with torch.no_grad():
        preds = best_model.predict(loader)

    # Convert to regular Python lists for JSON response
    return jsonify({"predictions": preds.squeeze().tolist()})

if __name__ == "__main__":
    app.run(debug=True)


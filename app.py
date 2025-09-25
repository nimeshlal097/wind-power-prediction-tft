import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Load model (rebuild + load weights, as we discussed)
# (Make sure you replicate architecture & dataset settings)
model = torch.load(r"C:\Users\User\Documents\GitHub\wind-power-prediction-tft\model_results\tft_final_model.pth")  


# Load or build datasets for display
# Load the dataset definition
with open("artifacts/training_dataset.pkl", "rb") as f:
    training_dataset = pickle.load(f)

# Load model weights into a model created from that dataset
tft = TemporalFusionTransformer.from_dataset(training_dataset)
tft.load_state_dict(torch.load("artifacts/tft_weights.pth", map_location="cpu"))
tft.eval()

# Load the test set if you saved it
test_df = pd.read_parquet("artifacts/test_df.parquet")

# Build a matching dataset for predictions
test_dataset = training_dataset.__class__.from_dataset(
    training_dataset, test_df, predict=True
)

test_loader = test_dataset.to_dataloader(train=False, batch_size=64)

# Title
st.title("TFT Wind Power Prediction Dashboard")

# Sidebar options
st.sidebar.header("Options")
show_pred = st.sidebar.checkbox("Show predictions vs actuals", value=True)
show_importance = st.sidebar.checkbox("Show encoder variable importance", value=True)

# Prediction vs Actuals
if show_pred:
    st.subheader("Predictions vs Actuals")
    # pick a batch
    batch = next(iter(test_loader))
    # use model.plot_prediction
    fig = model.plot_prediction(batch, idx=0, plot_attention=True)
    st.pyplot(fig)

# Encoder Variable Importance
if show_importance:
    st.subheader("Encoder Variable Importance (%)")
    # get interpret_output
    raw = model.predict(test_loader, mode="raw")
    interpre = model.interpret_output(raw)
    imp = interpre["encoder_variables"]
    # sum dims
    imp_sum = imp.sum(dim=(0,1,2)).detach().cpu().numpy()
    vars = test_dataset.reals  # list of variable names
    imp_series = pd.Series(imp_sum, index=vars)
    # drop target column if present
    if "TARGETVAR" in imp_series.index:
        imp_series = imp_series.drop("TARGETVAR")
    imp_series = 100 * imp_series / imp_series.sum()
    imp_series = imp_series.sort_values(ascending=True)
    fig2, ax = plt.subplots(figsize=(8,5))
    imp_series.plot(kind="barh", ax=ax, color="skyblue")
    ax.set_xlabel("Importance (%)")
    st.pyplot(fig2)

# Add more plots as needed...


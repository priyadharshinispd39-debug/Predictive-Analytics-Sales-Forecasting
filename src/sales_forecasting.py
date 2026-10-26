
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "historical_sales.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load and prepare data
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if not {"Date", "Sales"}.issubset(df.columns):
    raise ValueError("CSV must contain Date and Sales columns.")

df["MonthIndex"] = np.arange(len(df))
df["Month_sin"] = np.sin(2 * np.pi * df["MonthIndex"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["MonthIndex"] / 12)

features = ["MonthIndex", "Month_sin", "Month_cos"]
X = df[features]
y = df["Sales"]

# Chronological 80/20 split
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Sales Forecasting Model Evaluation")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

# Save test predictions
test_results = pd.DataFrame({
    "Date": df["Date"].iloc[split_index:].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": predictions
})
test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# Plot actual vs predicted sales
plt.figure(figsize=(10, 5))
plt.plot(test_results["Date"], test_results["Actual_Sales"],
         marker="o", label="Actual Sales")
plt.plot(test_results["Date"], test_results["Predicted_Sales"],
         marker="o", linestyle="--", label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# Forecast the next six months
future_steps = 6
future_index = np.arange(len(df), len(df) + future_steps)
future_dates = pd.date_range(
    start=df["Date"].max() + pd.offsets.MonthBegin(1),
    periods=future_steps,
    freq="MS"
)

future_features = pd.DataFrame({
    "MonthIndex": future_index,
    "Month_sin": np.sin(2 * np.pi * future_index / 12),
    "Month_cos": np.cos(2 * np.pi * future_index / 12)
})

# Fit final model on all available data for forecasting
final_model = LinearRegression()
final_model.fit(X, y)
future_sales = final_model.predict(future_features)

forecast = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Sales": future_sales
})
forecast.to_csv(OUTPUT_DIR / "future_sales_forecast.csv", index=False)

# Plot historical sales and future forecast
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Sales"], marker="o", label="Historical Sales")
plt.plot(forecast["Date"], forecast["Forecasted_Sales"],
         marker="o", linestyle="--", label="Six-Month Forecast")
plt.title("Historical Sales and Six-Month Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_forecast.png", dpi=300)
plt.close()

print("\nSix-Month Sales Forecast")
print(forecast.to_string(index=False))
print(f"\nOutputs saved to: {OUTPUT_DIR}")

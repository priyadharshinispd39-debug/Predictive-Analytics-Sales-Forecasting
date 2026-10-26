# Predictive Analytics — Sales Forecasting

## Project Overview

This project uses machine learning to analyze historical monthly sales data and forecast future sales using Python and Linear Regression.

It demonstrates data preprocessing, time-based feature engineering, model training, evaluation, visualization, and future sales forecasting.

## Objectives

- Analyze historical monthly sales data.
- Engineer trend and seasonal features.
- Train a Linear Regression model.
- Evaluate model performance using MAE, RMSE, and R².
- Forecast sales for six months.
- Visualize actual and predicted sales.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset

File: `data/historical_sales.csv`

The dataset contains 48 monthly observations from January 2022 to December 2025, with two main columns:

| Column | Description |
|---|---|
| Date | Monthly observation date |
| Sales | Sales value for the month |

## Methodology

1. Load and clean the dataset.
2. Convert dates to datetime format.
3. Create a sequential month index.
4. Encode monthly seasonality using sine and cosine features.
5. Split the dataset chronologically into training and testing sets.
6. Train a Linear Regression model.
7. Evaluate the model using MAE, RMSE, and R².
8. Generate a six-month sales forecast.
9. Export predictions and visualizations.

## Model Performance

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 5.07 |
| Root Mean Squared Error (RMSE) | 6.41 |
| R² Score | 0.836 |

The model explains approximately 83.6% of the variation in the held-out test data. Evaluation used 10 test observations from March to December 2025.

## Forecast Results

| Month | Forecast Sales |
|---|---:|
| January 2026 | 489.45 |
| February 2026 | 506.40 |
| March 2026 | 520.59 |
| April 2026 | 530.39 |
| May 2026 | 535.35 |
| June 2026 | 536.31 |

These are model-generated estimates, not actual sales.

## Repository Structure

- `data/` — Historical sales dataset
- `notebooks/` — Jupyter Notebook analysis
- `src/` — Python forecasting script
- `outputs/` — Prediction CSV files and graphs
- `reports/` — Project report
- `requirements.txt` — Python dependencies

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Predictive-Analytics-Sales-Forecasting.git
cd Predictive-Analytics-Sales-Forecasting
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebooks/Predictive_Analytics.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order.

### 4. Run the Python script

```bash
python src/sales_forecasting.py
```

The script saves predictions, forecast results, and charts to the `outputs/` folder.

## Limitations

- The dataset contains only 48 monthly observations.
- Evaluation uses a single chronological train/test split.
- The model uses a simple linear trend and seasonal features.
- Forecasts may change as new data becomes available.

## Future Improvements

- Compare additional regression and time-series forecasting algorithms.
- Use rolling-origin validation.
- Add prediction intervals.
- Include pricing, promotions, holidays, and other relevant variables.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for monthly sales forecasting. It provides a baseline model that can support data-driven planning and can be improved with more data and additional forecasting techniques.

## Author

Add your name and professional links here.

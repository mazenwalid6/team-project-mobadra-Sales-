from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import logging
import uuid

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to pre-trained SARIMAX model
MODEL_PATH = "SARIMAX_model.pkl"

# Load the SARIMAX model once at startup
try:
    logging.info("Loading SARIMAX model...")
    model = joblib.load(MODEL_PATH)
    logging.info("SARIMAX model loaded successfully.")
    try:
        logging.info(f"Model order: {model.order}")
        logging.info(f"Model seasonal_order: {model.seasonal_order}")
        logging.info(f"Exogenous variables: {model.exog_names}")
    except AttributeError:
        logging.debug("Model metadata attributes not found, skipping metadata logging.")
except Exception as e:
    logging.error(f"Failed to load SARIMAX model: {e}")
    model = None

# In-memory storage for latest forecast data
latest_data = None

# Utility to validate uploaded CSVs
def validate_csv(df):
    required_columns = ['Date', 'Store', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Type', 'Size']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df['Date'] = pd.to_datetime(df['Date'], errors='raise')
    return df

# Utility to aggregate weekly to monthly sums
def aggregate_monthly(df, sales_column='Weekly_Sales'):
    df = df.copy()
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly = df.groupby('YearMonth')[sales_column].sum().reset_index()
    monthly['YearMonth'] = monthly['YearMonth'].dt.to_timestamp()
    return monthly

@app.route('/data', methods=['GET'])
def get_data():
    if latest_data is None:
        return jsonify({"noData": True, "error": "No forecast data available. Please POST to /detect."}), 200
    return jsonify({"status": "success", "request_id": str(uuid.uuid4()), **latest_data})

@app.route('/upload', methods=['POST'])
def upload():
    request_id = str(uuid.uuid4())
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "status": "error", "request_id": request_id}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV", "status": "error", "request_id": request_id}), 400
    try:
        df = pd.read_csv(file)
        df = validate_csv(df)
        file.seek(0)
        file.save('temp_upload.csv')
        return jsonify({"status": "success", "message": "File uploaded successfully", "request_id": request_id})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error", "request_id": request_id}), 400

@app.route('/detect', methods=['POST'])
def detect():
    global latest_data, model
    try:
        request_id = str(uuid.uuid4())
        logging.info(f"New forecast request received. ID: {request_id}")

        if 'file' not in request.files:
            logging.error("No file in request")
            return jsonify({"error": "No file uploaded", "status": "error", "request_id": request_id}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "File must be a CSV", "status": "error", "request_id": request_id}), 400

        # Read and validate CSV
        try:
            logging.info("Reading CSV file...")
            df = pd.read_csv(file)
            logging.info(f"CSV loaded successfully. Shape: {df.shape}")
            df = validate_csv(df)
            logging.info("CSV validation successful")
        except Exception as e:
            logging.error(f"Error reading CSV: {str(e)}")
            return jsonify({"error": f"Error reading CSV: {str(e)}", "status": "error", "request_id": request_id}), 400

        # Clear previous data
        latest_data = None
        logging.info("Cleared previous data")

        # Process deductions
        deductions = request.form.get('deductions')
        if deductions:
            try:
                deductions = eval(deductions)
                logging.info(f"Processing deductions: {deductions}")
                total_deduction = 0
                for d in deductions:
                    value = float(d['value'])
                    if d['type'] == 'percentage':
                        total_deduction += df['Weekly_Sales'].sum() * (value / 100)
                    else:
                        total_deduction += value
                net_revenue = df['Weekly_Sales'].sum() - total_deduction
            except Exception as e:
                logging.error(f"Error processing deductions: {str(e)}")
                return jsonify({"error": f"Error processing deductions: {str(e)}", "status": "error", "request_id": request_id}), 400
        else:
            net_revenue = df['Weekly_Sales'].sum()

        # Prepare data for SARIMAX
        logging.info("Preparing data for model...")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Get the last 43 periods to match model's training data
        sales_data = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        sales_data = sales_data.set_index('Date')
        if len(sales_data) > 43:
            sales_data = sales_data.iloc[-43:]
        logging.info(f"Sales data prepared. Shape: {sales_data.shape}")

        # Prepare exogenous variables
        exog_columns = ['Store', 'Dept', 'IsHoliday', 'Type', 'Size']
        exog_data = df.groupby('Date')[exog_columns].mean()
        if len(exog_data) > 43:
            exog_data = exog_data.iloc[-43:]
        logging.info(f"Exogenous data prepared. Shape: {exog_data.shape}")

        # Ensure both datasets have the same index
        common_dates = sales_data.index.intersection(exog_data.index)
        sales_data = sales_data.loc[common_dates]
        exog_data = exog_data.loc[common_dates]
        logging.info(f"Data aligned. Final shapes - Sales: {sales_data.shape}, Exog: {exog_data.shape}")

        if model is None:
            logging.error("No valid model available.")
            return jsonify({"error": "No valid model available", "status": "error", "request_id": request_id}), 500

        # Generate predictions and forecasts
        try:
            logging.info("Generating historical predictions...")
            historical_predictions = model.get_prediction(start=0, end=len(sales_data)-1, exog=exog_data)
            historical_predicted = historical_predictions.summary_frame()['mean'].values
            logging.info(f"Historical predictions generated. Length: {len(historical_predicted)}")

            logging.info("Generating short-term forecast...")
            forecast_steps = 12
            last_exog = exog_data.iloc[-1][exog_columns].values
            forecast_exog = pd.DataFrame(
                [last_exog] * forecast_steps,
                columns=exog_columns,
                index=pd.date_range(start=sales_data.index[-1] + timedelta(weeks=1), periods=forecast_steps, freq='W')
            )
            
            forecast_result = model.get_forecast(steps=forecast_steps, exog=forecast_exog)
            forecast_values = forecast_result.summary_frame()['mean'].values
            conf_int = forecast_result.conf_int(alpha=0.05)
            confidence = (conf_int['upper Weekly_Sales'] - conf_int['lower Weekly_Sales']) / (2 * forecast_values)
            confidence = np.clip(confidence, 0, 1)
            logging.info(f"Short-term forecast generated. Length: {len(forecast_values)}")

        except Exception as e:
            logging.error(f"Error generating forecast: {str(e)}")
            logging.error(f"Sales data shape: {sales_data.shape}")
            logging.error(f"Exog data shape: {exog_data.shape}")
            return jsonify({"error": f"Error generating forecast: {str(e)}", "status": "error", "request_id": request_id}), 500

        # Create forecast DataFrames
        forecast_dates = pd.date_range(start=sales_data.index[-1] + timedelta(weeks=1), periods=forecast_steps, freq='W')
        
        forecast_sales_df = pd.DataFrame({
            'Date': forecast_dates,
            'Weekly_Sales': forecast_values
        })

        # Aggregate monthly data
        logging.info("Aggregating monthly data...")
        historical_monthly = aggregate_monthly(df)
        forecast_monthly = aggregate_monthly(forecast_sales_df)
        
        all_months = pd.date_range(
            start=df['Date'].min(),
            end=forecast_dates[-1],
            freq='M'
        ).to_period('M').strftime('%Y-%m').tolist()
        
        historical_monthly_sums = historical_monthly.set_index('YearMonth')['Weekly_Sales'].reindex(
            pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='M')
        ).fillna(0).values
        
        forecast_monthly_sums = forecast_monthly.set_index('YearMonth')['Weekly_Sales'].reindex(
            pd.date_range(start=forecast_dates[0], end=forecast_dates[-1], freq='M')
        ).fillna(0).values

        # Calculate metrics
        logging.info("Calculating metrics...")
        
        # Calculate current KPI values first
        gross_revenue = float(df['Weekly_Sales'].sum())
        net_revenue = float(net_revenue)  # Already calculated from deductions
        forecasted_revenue = float(forecast_values.sum())
        
        logging.info(f"KPI Values - Gross Revenue: ${gross_revenue:,.2f}, Net Revenue: ${net_revenue:,.2f}, Forecasted Revenue: ${forecasted_revenue:,.2f}")
        
        # Log actual vs predicted values for verification
        logging.info("Sample of actual vs predicted values:")
        for i in range(min(5, len(sales_data))):
            logging.info(f"Date: {sales_data.index[i]}, Actual: {sales_data['Weekly_Sales'].values[i]:.2f}, Predicted: {historical_predicted[i]:.2f}")
        
        # Calculate and log MAE
        mae = float(np.mean(np.abs(sales_data['Weekly_Sales'].values - historical_predicted)))
        logging.info(f"MAE calculation: {mae:.2f}")
        
        # Calculate and log MSE
        mse = float(np.mean((sales_data['Weekly_Sales'].values - historical_predicted) ** 2))
        logging.info(f"MSE calculation: {mse:.2f}")
        
        # Calculate accuracy for each point
        accuracy_values = []
        for actual, predicted in zip(sales_data['Weekly_Sales'].values, historical_predicted):
            if actual != 0:  # Avoid division by zero
                accuracy = 100 * (1 - abs(actual - predicted) / actual)
                accuracy_values.append(max(0, min(100, accuracy)))  # Clamp between 0 and 100
            else:
                accuracy_values.append(0)
        
        # Calculate overall accuracy
        accuracy = float(np.mean(accuracy_values))
        logging.info(f"Overall accuracy: {accuracy:.2f}%")

        # Calculate and log feature importance
        logging.info("Calculating feature importance...")
        feature_importance = []
        for column in exog_columns:
            correlation = np.corrcoef(exog_data[column], historical_predicted)[0, 1]
            importance = abs(correlation) if not np.isnan(correlation) else 0
            logging.info(f"Feature: {column}, Correlation: {correlation:.4f}, Importance: {importance:.4f}")
            feature_importance.append({
                "name": column,
                "value": float(importance)
            })

        # Store latest data
        logging.info("Preparing response data...")
        latest_data = {
            "forecast": {
                "labels": sales_data.index.strftime('%Y-%m-%d').tolist() + forecast_dates.strftime('%Y-%m-%d').tolist(),
                "historicalData": sales_data['Weekly_Sales'].tolist() + [0] * forecast_steps,
                "historicalPredictions": historical_predicted.tolist() + [0] * forecast_steps,
                "forecastData": [0] * len(sales_data) + forecast_values.tolist(),
                "confidence": [1.0] * len(sales_data) + confidence.tolist(),
                "monthly": {
                    "labels": all_months,
                    "historicalData": historical_monthly_sums.tolist() + [0] * len(forecast_monthly_sums),
                    "historicalPredictions": historical_monthly_sums.tolist() + [0] * len(forecast_monthly_sums),
                    "forecastData": [0] * len(historical_monthly_sums) + forecast_monthly_sums.tolist()
                }
            },
            "accuracyData": {
                "labels": sales_data.index.strftime('%Y-%m-%d').tolist(),
                "data": accuracy_values
            },
            "trendsData": {
                "labels": sales_data.index.strftime('%Y-%m-%d').tolist() + forecast_dates.strftime('%Y-%m-%d').tolist(),
                "data": sales_data['Weekly_Sales'].tolist() + forecast_values.tolist()
            },
            "metrics": {
                "mae": mae,
                "mse": mse,
                "accuracy": accuracy
            },
            "modelMetrics": {
                "mae": float(mae),
                "mse": float(mse),
                "featureImportance": feature_importance
            },
            "tableData": [
                {
                    "date": date.strftime('%Y-%m-%d'),
                    "actualSales": float(actual),
                    "predictedSales": float(predicted),
                    "accuracy": float(acc),
                    "error": float(abs(actual - predicted))
                }
                for date, actual, predicted, acc in zip(
                    sales_data.index,
                    sales_data['Weekly_Sales'].values,
                    historical_predicted,
                    accuracy_values
                )
            ],
            "kpi": {
                "grossRevenue": f"${gross_revenue:,.2f}",
                "netRevenue": f"${net_revenue:,.2f}",
                "forecastedRevenue": f"${forecasted_revenue:,.2f}",
                "forecastAccuracy": f"{accuracy:.1f}%"
            }
        }

        logging.info("Data processing completed successfully")
        logging.info(f"Final model metrics: {latest_data['modelMetrics']}")
        return jsonify({"status": "success", "request_id": request_id, **latest_data})

    except Exception as e:
        logging.error(f"Error in /detect: {str(e)}")
        logging.error(f"Full error details:", exc_info=True)
        return jsonify({"error": str(e), "status": "error", "request_id": request_id}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

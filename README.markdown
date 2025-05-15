# SARIMA Forecasting Dashboard

## Overview

The SARIMA Forecasting Dashboard is a web application for predicting retail sales using a Seasonal ARIMA (SARIMA) model. The frontend, built with React and Vite, provides an intuitive interface to upload CSV files, view key performance indicators (KPIs), and visualize sales forecasts, accuracy charts, and trends. The backend, built with Flask, processes uploaded data, generates forecasts using a pre-trained SARIMA model, and returns metrics such as gross revenue, net revenue, forecasted revenue, and forecast accuracy.

### Project Structure

- **Root Directory**: `team-project-mobadra-Sales-now`
- `client/`: Frontend (React + Vite)
  - `src/components/Dashboard/`: Contains components like `KPICards.jsx`, `Dashboard.jsx`, etc.
  - `package.json`: Lists dependencies (e.g., `react`, `@radix-ui/react-tooltip`, `lucide-react`).
- `server/`: Backend (Flask)
  - `app.py`: Flask application handling `/upload` and `/detect` endpoints.
  - `SARIMAX_model.pkl`: Pre-trained SARIMA model.
  - `requirements.txt`: Lists Python dependencies for the backend.
- **Data Requirements**:
  - CSV files must include columns: `Date`, `Store`, `Dept`, `Weekly_Sales`, `IsHoliday`, `Type`, `Size`.

### Features

- **Upload CSV**: Upload sales data for forecasting.
- **KPIs**: Displays Sales (12-week forecast), Forecast Accuracy, Gross Revenue, and Net Revenue with hover tooltips explaining each metric.
- **Charts**: Visualizes sales forecasts, accuracy, and trends.
- **SARIMA Model**: Uses a pre-trained SARIMAX model to predict sales based on historical data and exogenous variables (`Store`, `Dept`, `IsHoliday`, `Type`, `Size`).

## Prerequisites

- **Node.js**: Version 18 or higher (for Vite frontend).
- **Python**: Version 3.8 or higher (for Flask backend).
- **Git**: For cloning the repository (optional).
- **System Requirements**: Windows, macOS, or Linux with at least 4GB RAM.

## Setup Instructions

### 1. Clone the Repository (Optional)

If you have the project locally, skip this step. Otherwise:

```bash
git clone <repository-url>
cd team-project-mobadra-Sales-now
```

### 2. Set Up the Backend (`server/`)

1. **Navigate to the server directory**:

   ```bash
   cd server
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   - This installs all dependencies listed in `requirements.txt` (e.g., `flask`, `pandas`, `numpy`, `joblib`, `statsmodels`, `flask-cors`).

4. **Verify SARIMA model**:

   - Ensure `SARIMAX_model.pkl` exists in `server/`.
   - Check model compatibility:

     ```bash
     python -c "import joblib; m=joblib.load('SARIMAX_model.pkl'); print(m.order, m.seasonal_order, m.exog_names, m.nobs)"
     ```
     - Expected output: `(1,1,1) (1,1,1,52) ['Store', 'Dept', 'IsHoliday', 'Type', 'Size'] 43`
     - If the model is missing or incompatible, retrain it (see Troubleshooting).

### 3. Set Up the Frontend (`client/`)

1. **Navigate to the client directory**:

   ```bash
   cd ../client
   ```

2. **Install Node.js dependencies**:

   ```bash
   npm install
   ```

   - This installs `react`, `@radix-ui/react-tooltip`, `lucide-react`, and other dependencies listed in `package.json`.

3. **Verify dependencies**:

   ```bash
   npm ls @radix-ui/react-tooltip lucide-react
   ```

   - Ensure no errors and both packages are listed.

### 4. Prepare Sample CSV

- Ensure your CSV has the required columns: `Date`, `Store`, `Dept`, `Weekly_Sales`, `IsHoliday`, `Type`, `Size`.
- Example format:

  ```csv
  Date,Store,Dept,Weekly_Sales,IsHoliday,Type,Size
  2010-02-05,1,1,24924.50,FALSE,A,151315
  2010-02-12,1,1,46039.49,TRUE,A,151315
  ```
- Place the CSV in an accessible location for uploading.

## Starting the Website

### 1. Start the Backend

1. **Navigate to the server directory** (if not already there):

   ```bash
   cd server
   ```

2. **Activate the virtual environment** (if not active):

   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Run the Flask app**:

   ```bash
   python app.py
   ```

   - The backend will run on `http://localhost:5001`.
   - Look for logs like:

     ```
     * Running on http://127.0.0.1:5001
     Loading SARIMAX model...
     SARIMAX model loaded successfully.
     ```

### 2. Start the Frontend

1. **Open a new terminal** and navigate to the client directory:

   ```bash
   cd client
   ```

2. **Run the Vite development server**:

   ```bash
   npm run dev
   ```

   - The frontend will run on `http://localhost:5173`.
   - Look for output like:

     ```
     VITE v4.x.x  ready in 300 ms
     ➜  Local:   http://localhost:5173/
     ```

### 3. Access the Website

- Open a browser (preferably in incognito mode) and go to `http://localhost:5173`.
- Upload a CSV file, add deductions (e.g., Name: "Returns", Value: "10", Type: "percentage"), and click "Generate Forecast".

## Expected Output

- **Dashboard**:
  - **KPICards**: Displays four cards:
    - Sales: \~$177,540 (12-week forecast).
    - Forecast Accuracy: \~96.3%.
    - Gross Revenue: \~$2,114,969.
    - Net Revenue: \~$1,903,473 (after 10% deduction).
  - **Tooltips**: Hovering over each card shows a description (e.g., “Total predicted sales revenue for the next 12 weeks...” for Sales).
  - **Charts**: Visualizes sales forecasts, accuracy, and trends.
- **Console Logs** (browser):
  - `KPICards received data: { forecastedRevenue: "$177,540.00", forecastAccuracy: "96.3%", grossRevenue: "$2,114,969.00", netRevenue: "$1,903,473.10", ... }`
  - `KPICards isLoading: false`
- **Flask Logs**:
  - `KPI Values - Gross Revenue: $2,114,969.00, Net Revenue: $1,903,473.10, Forecasted Revenue: $177,540.00`
  - `Response data` with `kpi` fields.

## Troubleshooting

### Vite Errors

- **Error**: `Failed to resolve import "@radix-ui/react-tooltip"`:
  - **Fix**:

    ```bash
    cd client
    npm install @radix-ui/react-tooltip@latest
    rm -rf node_modules package-lock.json
    npm install
    ```
  - Verify: `npm ls @radix-ui/react-tooltip`.

### KPI Values Show `$0` or `0%`

- **Check**:
  - Browser console logs:
    - `Raw data received:` (should include `kpi` fields).
    - `KPI data for KPICards:` (should match backend `kpi`).
    - `KPICards received data:` (should not be `{ forecastedRevenue: "$0", ... }`).
  - Flask logs:
    - `KPI Values` (should show non-zero values).
    - `Response data` (should include `kpi`).
  - Network tab: `/detect` response JSON (should have `kpi` fields).
- **Fix**:
  - Ensure `/detect` is called (not `/data`).
  - Add debug in `client/src/components/Dashboard/Dashboard.jsx` before `<KPICards>`:

    ```javascript
    console.log("Passing to KPICards:", dashboardData?.kpi);
    ```
  - Share logs and CSV sample (10-20 rows).

### Backend Errors

- **Error**: `/detect` returns 500 (e.g., SARIMAX shape mismatch):
  - **Check**:

    ```bash
    python -c "import joblib; m=joblib.load('server/SARIMAX_model.pkl'); print(m.order, m.seasonal_order, m.exog_names, m.nobs)"
    ```
    - Expected: `(1,1,1) (1,1,1,52) ['Store', 'Dept', 'IsHoliday', 'Type', 'Size'] 43`
  - **Fix**: Retrain the model:

    ```python
    # server/train_model.py
    import pandas as pd
    import joblib
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    df = pd.read_csv("path_to_your_csv.csv")  # Replace with CSV path
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    sales_data = df.groupby('Date')['Weekly_Sales'].sum()
    exog_columns = ['Store', 'Dept', 'IsHoliday', 'Type', 'Size']
    exog_data = df.groupby('Date')[exog_columns].mean()
    common_dates = sales_data.index.intersection(exog_data.index)
    sales_data = sales_data.loc[common_dates].iloc[-43:]
    exog_data = exog_data.loc[common_dates].iloc[-43:]
    model = SARIMAX(
        sales_data,
        exog=exog_data,
        order=(1,1,1),
        seasonal_order=(1,1,1,52)
    ).fit(disp=False)
    joblib.dump(model, "SARIMAX_model.pkl")
    print(model.order, model.seasonal_order, model.exog_names, model.nobs)
    ```

    Run:

    ```bash
    python train_model.py
    ```

### Tooltips Not Appearing

- **Check**: Browser console for errors about `TooltipProvider`.
- **Fix**:
  - Verify `@radix-ui/react-tooltip`:

    ```bash
    npm ls @radix-ui/react-tooltip
    ```
  - Add `z-index` to `client/src/components/Dashboard/KPICards.jsx`:

    ```javascript
    <TooltipContent className="bg-background border border-border p-2 rounded-md max-w-xs z-50">
    ```
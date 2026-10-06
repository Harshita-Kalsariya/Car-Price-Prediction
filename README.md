# Used Car Price Prediction

A machine learning project that predicts the resale price of a used car in **Indian rupees (₹)**. A Random Forest model is trained on used-car listings, and a Flask web app lets anyone enter a car's details and get an instant price estimate with a likely price range.

## Features

- Cleans messy real-world car data (units inside text, missing values, duplicates, impossible values)
- Feature engineering: `car_age`, `km_per_year`, `brand`, `power_per_cc`, `is_first_owner`
- Leakage check: no column that would give away the answer is used for training
- Random Forest Regressor evaluated with **MAE, RMSE and R²**, with the error reported in rupees
- Feature importance: shows which columns drive the price the most
- Web app with a form, a number-plate style result card and a low/high price range

## Tech stack

| Part | Tools |
|---|---|
| Data and ML | Python, pandas, NumPy, scikit-learn, joblib, matplotlib |
| Notebook | Jupyter |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |

## Project structure

```
main-folder/
├── backend/
│   ├── app.py                  # Flask server + prediction API
│   └── price_model.joblib      # trained model (created automatically)
├── frontend/
│   ├── index.html              # web page
│   └── style.css               # styling
├── notebook/
│   └── used_car_price.ipynb    # data cleaning, features, training, evaluation
├── datasets/
│   └── cars.csv                # dataset
└── README.md
```

## Dataset

Download **"Car details v3.csv"** from the Kaggle dataset *Vehicle dataset from CarDekho* and save it as `datasets/cars.csv`.

Expected columns: `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats`.

If `cars.csv` is missing, the notebook generates a sample messy dataset so the project still runs end to end. Replace it with the real file for real results.

## Setup and run

**1. Install the libraries**

```
pip install pandas numpy scikit-learn joblib flask matplotlib jupyter
```

**2. Train the model (notebook)**

Open `notebook/used_car_price.ipynb` from inside the `notebook/` folder and choose **Run All**. This cleans the data, trains the model, prints the metrics and saves `backend/price_model.joblib`.

**3. Start the web app**

From the main folder:

```
python backend/app.py
```

Then open http://127.0.0.1:5000 in your browser.

> The notebook and `app.py` should use the same Python environment. If the saved model was made with a different scikit-learn version, `app.py` detects it and re-trains the model automatically from `datasets/cars.csv` (this takes a short while the first time).

## How it works

### Part 1: Data

| Step | What is done |
|---|---|
| Duplicates | Exact duplicate rows are removed |
| Units | `"23.4 kmpl"`, `"1248 CC"`, `"74 bhp"` become plain numbers |
| Missing values | Rows without a price or year are dropped. Other gaps are filled with the median (numbers) or most frequent value (categories), learned from the training data only |
| Impossible values | Years outside 1990 to 2026 are removed, km above 500,000 is treated as a typo, extreme prices (outside the 0.5th to 99.5th percentile) are trimmed |
| Categories | Text is tidied, then one-hot encoded. Unknown categories are ignored at prediction time |

**Engineered features**

| Feature | Meaning |
|---|---|
| `car_age` | 2026 minus the manufacturing year |
| `km_per_year` | Kilometres driven divided by car age (how heavily it was used) |
| `brand` | First word of the car name. Rare brands are grouped as "Other" |
| `power_per_cc` | Engine power divided by engine size |
| `is_first_owner` | 1 if the car has had only one owner |

**Leakage check.** `selling_price` is the target and is never a feature. `name` is replaced by `brand`, and `year` is replaced by `car_age`. The notebook also prints each numeric feature's correlation with price and warns if any is above 0.95.

### Part 2: Model

- **Model:** `RandomForestRegressor` (300 trees) inside a scikit-learn Pipeline
- **Target:** the model learns on `log(price)` because prices are right-skewed. Predictions are converted back to rupees
- **Split:** 80% train, 20% test (`random_state=42`)
- **Metrics:** MAE (average miss in ₹), RMSE (punishes big misses) and R² (share of variation explained)
- **Feature importance:** one-hot columns are grouped back to the original feature, and the top 5 are printed
- **Price range:** the low and high values on the website are the 10th and 90th percentile of the individual trees' predictions

### Results

Results depend on the dataset. Fill in your own numbers after running the notebook on the real data:

| Metric | Your result |
|---|---|
| MAE | ₹ ________ |
| RMSE | ₹ ________ |
| R² | ________ |

Top 5 features: ____________________

(On the built-in sample data the model gets an MAE of about ₹25,000 and R² of about 0.93, but that data is artificial.)

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Web page |
| GET | `/api/meta` | Dropdown options, model metrics, top feature importances |
| POST | `/api/predict` | Predict a price |

Example request to `/api/predict`:

```json
{
  "brand": "Hyundai", "fuel": "Petrol", "seller_type": "Individual",
  "transmission": "Manual", "owner": "First Owner",
  "year": 2019, "km_driven": 45000
}
```

Example response:

```json
{ "price": 226500.0, "low": 175300.0, "high": 284600.0 }
```

`engine`, `max_power`, `mileage` and `seats` are optional. If left out, typical values from the training data are used.

## Troubleshooting

| Problem | Fix |
|---|---|
| `can't open file 'app.py'` | `app.py` is inside `backend/`. Run `python backend/app.py` from the main folder |
| `datasets/cars.csv nahi mili` | Put the dataset in `datasets/` or run the notebook once to create sample data |
| `SimpleImputer has no attribute '_fill_dtype'` | scikit-learn version mismatch. Use the latest `app.py`, which re-trains the model automatically |
| Page looks old after editing files | Press Ctrl + F5 to clear the browser cache |
| `No module named flask` | Run `pip install flask` with the same Python you use to start the app |
| Two Pythons installed (e.g. Anaconda and python.org) | Start the app with the full path, e.g. `C:\Users\<you>\anaconda3\python.exe backend\app.py` |

## Limitations and ideas for improvement

- Predictions are estimates, not offers. Condition, accident history and city are not in the data
- Compare with Gradient Boosting or XGBoost and tune hyperparameters
- Add more features such as car model, city and service history
- Use cross-validation for a more reliable error estimate
- Deploy the app online (for example on Render)

## Learning outcomes

Feature engineering, handling categories, spotting leakage, training and judging a regression model with MAE, RMSE and R², reading feature importance, and serving a model through a web app.

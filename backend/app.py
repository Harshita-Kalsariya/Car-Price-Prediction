#"""Flask backend for the Used Car Price Prediction project.
#Run (from this backend folder):  python app.py   ->  http://127.0.0.1:5000

#The model lives in price_model.joblib (created by the notebook). If that file is missing or was
#saved by a different scikit-learn version, this app automatically re-trains it from
#../datasets/cars.csv using the SAME steps as the notebook -- so version mismatches can't break it."""
import os, traceback
import joblib, numpy as np, pandas as pd, sklearn
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.join(HERE, "..", "frontend")
CSV = os.path.join(HERE, "..", "datasets", "cars.csv")
PATH = os.path.join(HERE, "price_model.joblib")
app = Flask(__name__, template_folder=FRONT, static_folder=FRONT, static_url_path="/static")
app.json.sort_keys = False

# ---- same feature engineering as the notebook ----
CURRENT_YEAR = 2026
TARGET = "selling_price"
CAT_COLS = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUM_COLS = ["km_driven", "mileage", "engine", "max_power", "seats",
            "car_age", "km_per_year", "power_per_cc", "is_first_owner"]
FEATURES = CAT_COLS + NUM_COLS


def clean(df):
    df = df.drop_duplicates().copy()
    for c in ["mileage", "engine", "max_power"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    for c in [TARGET, "year", "km_driven"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET, "year"])
    df = df[(df["year"].between(1990, CURRENT_YEAR)) & (df[TARGET] > 0)]
    df.loc[(df["km_driven"] < 0) | (df["km_driven"] > 500_000), "km_driven"] = np.nan
    lo, hi = df[TARGET].quantile([0.005, 0.995])
    df = df[df[TARGET].between(lo, hi)]
    for c in ["fuel", "seller_type", "transmission", "owner"]:
        df[c] = df[c].astype(str).str.strip().str.title().replace("Nan", np.nan)
    return df.reset_index(drop=True)


def extract_brand(df, min_count=20):
    df = df.copy()
    df["brand"] = df["name"].astype(str).str.split().str[0].str.title()
    counts = df["brand"].value_counts()
    df["brand"] = df["brand"].where(df["brand"].map(counts) >= min_count, "Other")
    return df


def add_features(df):
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)
    df["power_per_cc"] = df["max_power"] / df["engine"]
    df["is_first_owner"] = (df["owner"] == "First Owner").astype(int)
    return df


def train_and_save():
    """Same pipeline as the notebook, trained with THIS Python's scikit-learn."""
    if not os.path.exists(CSV):
        raise SystemExit("datasets/cars.csv nahi mili. Dataset wahan rakho ya notebook Run All karo.")
    print("Model train ho raha hai (ek baar, thoda time lagega)...")
    df = add_features(extract_brand(clean(pd.read_csv(CSV))))
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS)])
    rf = Pipeline([("prep", prep), ("rf", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42))])
    model = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    names = model.regressor_.named_steps["prep"].get_feature_names_out()
    imp = model.regressor_.named_steps["rf"].feature_importances_
    grouped = {}
    for n, v in zip(names, imp):
        n = n.split("__", 1)[1]
        key = next((c for c in CAT_COLS if n.startswith(c + "_")), n)
        grouped[key] = grouped.get(key, 0) + v
    bundle = {
        "sklearn_version": sklearn.__version__, "model": model,
        "importance": dict(sorted(grouped.items(), key=lambda kv: -kv[1])),
        "metrics": {"mae": float(mean_absolute_error(y_te, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "r2": float(r2_score(y_te, pred))},
        "brands": sorted(X["brand"].unique()),
        "options": {c: sorted(X[c].dropna().unique()) for c in ["fuel", "seller_type", "transmission", "owner"]},
        "defaults": X_tr[["mileage", "engine", "max_power", "seats"]].median().to_dict(),
    }
    joblib.dump(bundle, PATH)
    print(f"Model saved. MAE = Rs {bundle['metrics']['mae']:,.0f}")
    return bundle


def load_model():
    try:
        b = joblib.load(PATH)
        if b.get("sklearn_version") == sklearn.__version__:
            return b
        print(f"Model purane scikit-learn ({b.get('sklearn_version')}) ka hai, ab {sklearn.__version__} chal raha hai. Dobara train kar raha hoon.")
    except Exception as e:
        print("Saved model load nahi hua:", e)
    return train_and_save()


B = load_model()
pipe = B["model"].regressor_


def num(data, key, lo, hi, default=None):
    v = data.get(key)
    if v in (None, ""):
        if default is None:
            raise ValueError(f"{key} is required")
        return default
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}")
    return v


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    return jsonify(brands=B["brands"], options=B["options"], defaults=B["defaults"],
                   metrics=B["metrics"], importance=dict(list(B["importance"].items())[:6]),
                   year_now=CURRENT_YEAR)


@app.post("/api/predict")
def predict():
    d = request.get_json(silent=True) or {}
    try:
        row = {
            "brand": d.get("brand"), "fuel": d.get("fuel"), "seller_type": d.get("seller_type"),
            "transmission": d.get("transmission"), "owner": d.get("owner"),
            "year": num(d, "year", 1995, CURRENT_YEAR),
            "km_driven": num(d, "km_driven", 0, 500_000),
            "mileage": num(d, "mileage", 5, 40, B["defaults"]["mileage"]),
            "engine": num(d, "engine", 600, 6000, B["defaults"]["engine"]),
            "max_power": num(d, "max_power", 30, 600, B["defaults"]["max_power"]),
            "seats": num(d, "seats", 2, 10, B["defaults"]["seats"]),
        }
        if not all(row[k] for k in ["brand", "fuel", "seller_type", "transmission", "owner"]):
            raise ValueError("Please fill in brand, fuel, seller, transmission and owner")
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400

    X = add_features(pd.DataFrame([row]))[FEATURES]
    price = float(B["model"].predict(X)[0])
    try:   # low/high from the spread of the forest's trees; fall back to +/- MAE
        Xt = pipe.named_steps["prep"].transform(X)
        trees = np.expm1([t.predict(Xt)[0] for t in pipe.named_steps["rf"].estimators_])
        low, high = [float(v) for v in np.percentile(trees, [10, 90])]
    except Exception:
        traceback.print_exc()
        mae = B["metrics"]["mae"]
        low, high = max(price - mae, 0), price + mae
    return jsonify(price=round(price, -2), low=round(low, -2), high=round(high, -2))


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=f"{e.code} {e.name}"), e.code
    traceback.print_exc()
    return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    app.run(debug=False)
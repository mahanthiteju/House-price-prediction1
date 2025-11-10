import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


# Try loading expected dataset files. Fall back to train.csv if housing.csv
# is absent.
data_paths = ["data/housing.csv", "data/train.csv"]
data_file = None
for p in data_paths:
	if os.path.exists(p):
		data_file = p
		break

if data_file is None:
	raise FileNotFoundError(
		"No data file found. Expected one of: " + ", ".join(data_paths)
	)

print("Loading data from:", data_file)
df = pd.read_csv(data_file)


# Normalize common column names between different datasets
if "TARGET(PRICE_IN_LACS)" in df.columns:
	df = df.rename(columns={"TARGET(PRICE_IN_LACS)": "price"})

if "SQUARE_FT" in df.columns:
	df = df.rename(columns={"SQUARE_FT": "size"})


# If there is an ADDRESS column but no explicit 'location', extract city
# from ADDRESS
if "ADDRESS" in df.columns and "location" not in df.columns:
	def _extract_city(addr):
		if pd.isnull(addr):
			return "unknown"
		parts = str(addr).split(",")
		return parts[-1].strip() if parts else "unknown"

	df["location"] = df["ADDRESS"].apply(_extract_city)


# If amenities column is missing, fill with a default value so
# get_dummies works
if "amenities" not in df.columns:
	df["amenities"] = "none"


# Ensure required numeric columns exist
if "price" not in df.columns:
	raise ValueError(
		"Could not find target column 'price' in data. "
		f"Columns: {list(df.columns)[:20]}"
	)

if "size" not in df.columns:
	# Try to infer a size-like column
	possible_size = None
	for c in ["area", "sqft", "square_ft", "SQUARE_FT"]:
		if c in df.columns:
			possible_size = c
			break

	if possible_size:
		df = df.rename(columns={possible_size: "size"})
	else:
		raise ValueError(
			"Could not find a size column (expected 'size' or 'SQUARE_FT')."
		)


# Drop rows with missing target or size
df = df.dropna(subset=["price", "size"]) 


# One-hot encode location and amenities (drop_first for simplicity)
df = pd.get_dummies(df, columns=["location", "amenities"], drop_first=True)


# Prepare X, y
X = df.drop("price", axis=1)
y = df["price"]

# Keep only numeric feature columns for training. This avoids errors when
# non-numeric columns (like POSTED_BY) remain in the dataset.
X = X.select_dtypes(include=["number"])  # only numeric columns
if X.shape[1] == 0:
	raise ValueError("No numeric features found after preprocessing.")


# Train model
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)


# Save model (keep same filename as app expects)
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))


# Also save the feature list so you can inspect it later (not used by app
# currently)
with open(os.path.join("models", "features.txt"), "w", encoding="utf-8") as f:
	f.write(",".join(map(str, X.columns.tolist())))


print("Training finished. Model saved to models/model.pkl")
print("Features (first 20):", X.columns.tolist()[:20])
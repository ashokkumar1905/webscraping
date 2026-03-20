import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

class SalesPredictor: 
    def __init__(self, data_path):
        self.data_path = data_path
        self.linear_model = None
        self.xgb_model = None
        self.scaler = None
        self.df = None

    def load_data(self):
        # Load the dataset from the provided path
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d')

    def preprocess_data(self):
        # Feature Engineering (replace with your specific data preprocessing logic)
        parts = self.df["date"].dt
        self.df["year"] = parts.year
        self.df["month"] = parts.month
        self.df["day"] = parts.day

        self.df['m1'] = np.sin(self.df['month'] * (2 * np.pi / 12))
        self.df['m2'] = np.cos(self.df['month'] * (2 * np.pi / 12))

        features = self.df.drop(columns=['sales', 'date'])
        target = self.df['sales']

        return train_test_split(features, target, test_size=0.2, random_state=42)

    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()

        # Normalize the data
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train Linear Regression
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)

        # Train XGBoost
        self.xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        self.xgb_model.fit(X_train, y_train)

        print("Models trained successfully.")

    def save_models(self):
        joblib.dump(self.linear_model, 'linear_model.pkl')
        joblib.dump(self.xgb_model, 'xgb_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Models and scaler saved successfully.")

    def load_models(self):
        self.linear_model = joblib.load('linear_model.pkl')
        self.xgb_model = joblib.load('xgb_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def predict(self, features):
        # Apply scaling to features
        features_scaled = self.scaler.transform(features)

        # Make predictions
        linear_prediction = self.linear_model.predict(features_scaled)
        xgb_prediction = self.xgb_model.predict(features_scaled)

        return linear_prediction, xgb_prediction

# Example usage
if __name__ == "__main__":
    predictor = SalesPredictor(r"C:\Users\aasho\OneDrive\Desktop\major\train.csv")  # Use the correct path to the CSV file
    predictor.load_data()
    predictor.train_models()
    predictor.save_models()

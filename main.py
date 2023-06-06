import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from dask_ml.xgboost import XGBRegressor
import joblib
from dask.distributed import Client

def main():
    # Start a Dask client
    client = Client()

    # Load the advertising data
    advertising_data = dd.read_csv('advertising.csv')

    # Preprocess the data
    X = advertising_data[['TV', 'Radio', 'Newspaper']]
    y = advertising_data['Sales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True)

    # Train the XGBoost model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test).compute().astype(int)  # Convert to NumPy array
    y_test = y_test.compute().values.astype(int)  # Convert to NumPy array

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save the trained model
    joblib.dump(model, 'advertising_sales_model.joblib')

    # Display model performance
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

if __name__ == '__main__':
    main()

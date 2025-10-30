# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Create dataset

road_data = {
    'speed_limit': [50, 70, 90, 100, 110, 60, 80],
    'light_conditions': [1, 2, 3, 1, 2, 3, 2],
    'weather_conditions': [1, 3, 2, 1, 2, 3, 1],
    'road_surface': [1, 2, 3, 1, 2, 3, 2],
    'vehicle_age': [2, 5, 1, 3, 6, 4, 2],
    'driver_experience': [10, 4, 15, 7, 3, 5, 9],
    'num_vehicles_involved': [1, 2, 2, 3, 1, 2, 3],
    'accident_severity': [1, 2, 3, 2, 3, 2, 1]
}

df = pd.DataFrame(road_data)
df.to_csv('road_accidents.csv', index=False)
print("Road accident dataset 'road_accidents.csv' generated successfully!\n")


# Prepare data

X = df[['speed_limit', 'light_conditions', 'weather_conditions', 'road_surface',
        'vehicle_age', 'driver_experience', 'num_vehicles_involved']]
y = df['accident_severity']

# Split dataset into training and test portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# linear regression model

model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model for future use
joblib.dump(model, 'road_accident_model.pkl')

# Evaluate model performance

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance Summary:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coefficient of Determination (R²): {r2:.2f}\n")

#Test model with new scenario

sample_input = pd.DataFrame({
    'speed_limit': [85],
    'light_conditions': [2],
    'weather_conditions': [1],
    'road_surface': [2],
    'vehicle_age': [3],
    'driver_experience': [6],
    'num_vehicles_involved': [2]
})

predicted_value = model.predict(sample_input)[0]
print(f"Estimated Accident Severity Level: {predicted_value:.2f}")

#The code below is for linear regression, the logic and code was written by the user, The direction comments are used to
#explain the code
# ""The code is just modified by AI to look clean""

# 🧠 Core Libraries for Computation and Visualization
import numpy as np                    # For numerical operations (like RMSE calculation)
import pandas as pd                   # For data manipulation and analysis
import matplotlib.pyplot as plt       # For visualizing plots
import seaborn as sns                 # Optional advanced visualization

# 📦 Scikit-learn Imports
from sklearn.datasets import fetch_california_housing   # Built-in California Housing dataset
from sklearn.model_selection import train_test_split     # Function to split dataset
from sklearn.linear_model import LinearRegression        # Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score # Evaluation metrics

# 📥 Step 1: Load the Dataset
# as_frame=True returns data as a pandas DataFrame instead of numpy arrays
california = fetch_california_housing(as_frame=True)

# 🧮 Step 2: Separate Data (X) and Target Labels (y)
X = california.data       # Features (input variables)
y = california.target     # Target (output/label -> Median house value)

# 🖨️ Step 3: Print the Shapes (Dimensions) of X and y
print("X shape:", X.shape)   # Example: (20640, 8) --> 20640 samples, 8 features
print("y shape:", y.shape)   # Example: (20640,)   --> 20640 target values

# 🔀 Step 4: Split Data into Training and Testing Sets
# 80% for training, 20% for testing; random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏋️‍♂️ Step 5: Initialize and Train the Linear Regression Model
lin_reg = LinearRegression()         # Create model object
lin_reg.fit(X_train, y_train)        # Train the model using training data

# 🤖 Step 6: Make Predictions on the Test Set
y_pred = lin_reg.predict(X_test)     # Predict house prices using the model

# 📊 Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)                       # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)             # R² Score (how well model fits)

# 📢 Step 8: Display Evaluation Results
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# 📈 Step 9: Visualize Actual vs Predicted Values Using a Bar Plot
plt.figure(figsize=(15, 6))               # Set plot size
bar_width = 0.5                           # Width of each bar
indices = np.arange(25)                   # Index for first 25 samples

# Plot actual target values
plt.bar(indices, y_test.iloc[:25], width=bar_width, label='Actual', color='skyblue')

# Plot predicted target values
plt.bar(indices + bar_width, y_pred[:25], width=bar_width, label='Predicted', color='salmon')

# Add labels and legends to plot
plt.xlabel('Sample Index')                              # X-axis label
plt.ylabel('Median House Value')                        # Y-axis label
plt.title('Actual vs Predicted House Values (First 25 Samples)')  # Title
plt.xticks(indices + bar_width / 2, indices)            # Center x-axis ticks
plt.legend()                                            # Show legend
plt.tight_layout()                                      # Adjust layout
plt.show()                                              # Display the plot

# ⚠️ Optional: Show Error Between Actual and Predicted (Actual - Predicted)
errors = y_test.iloc[:25].values - y_pred[:25]

# 🔍 Step 10: Visualize Errors
plt.figure(figsize=(15, 4))
plt.bar(indices, errors, color='purple')
plt.axhline(0, color='black', linestyle='--')           # Horizontal zero line
plt.xlabel('Sample Index')
plt.ylabel('Prediction Error')
plt.title('Prediction Errors (Actual - Predicted) for First 25 Samples')
plt.tight_layout()
plt.show()
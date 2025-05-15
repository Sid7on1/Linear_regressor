# *****📊 Linear Regression Housing Price Predictor*****

This project uses **Linear Regression**, a supervised regression algorithm, to predict California housing prices based on features like population, income, house age, and more.

## 🏠 Is it a House Price Estimator?

✅ Yes! This model acts like a house price predictor. It learns the relationship between different neighborhood features and the **median house value** to estimate future prices.

## ⚙️ What Happens in the Code?

1. 📥 Loads the California Housing dataset  
2. 🧹 Splits the data into features (`X`) and target values (`y`)  
3. 🔧 Uses `train_test_split()` to create training and testing sets  
4. 🧪 Trains the Linear Regression model  
5. 🔮 Predicts housing prices on test data  
6. 📈 Evaluates performance using MSE, RMSE, and R² Score  
7. 🎨 Visualizes actual vs predicted values with Matplotlib bar chart  

> Built using Python, Pandas, scikit-learn, Matplotlib, and Seaborn.

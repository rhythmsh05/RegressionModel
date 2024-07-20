from src.data.make_dataset import load_data
from src.features.build_features import train_test_splitter
from src.models.train_model import train_linear_regression, train_decision_tree_regressor, train_random_forest_regressor
from src.models.predict_model import evaluate_model
from src.visualization.visualize import plot_tree
if __name__ == "__main__":
    # Load  the data
    data_path = "RegressionModel/data/raw/final.csv"
    df = load_data(data_path)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_splitter(df)

# train your model
lrmodel = train_linear_regression(X_train,y_train)  #syntax

# Metrics
print("Coefficients of Linear Regression: ", lrmodel.coef_)
print("Intercept of Linear Regression: ", lrmodel.intercept_)

# Display evaluation metrics for linear regression
lr_train_mae, lr_test_mae = evaluate_model(lrmodel, X_train, X_test, y_train, y_test)
print('Linear Regression Train error is', lr_train_mae)
print('Linear Regression Test error is', lr_test_mae)

#Train the Decision Tree Regressor model
dt_model = train_decision_tree_regressor(X_train, y_train)

# Display evaluation metrics for Decision Tree Regressor
dt_train_mae, dt_test_mae = evaluate_model(dt_model, X_train, X_test, y_train, y_test)
print('Decision Tree Regressor Train error is', dt_train_mae)
print('Decision Tree Regressor Test error is', dt_test_mae)

#Plot the decision tree
plot_tree(dt_model, dt_model.feature_names_in_, save_path='RegressionModel/reports/figures/tree.png')

#Train the Random Forest Regressor
rf_model = train_random_forest_regressor(X_train, y_train)

# Display evaluation metrics for Random Forest Regressor
rf_train_mae, rf_test_mae = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
print('Random Forest Regressor Train error is', rf_train_mae)
print('Random Forest Regressor Test error is', rf_test_mae)

#Plot the Random Forest Regressor
plot_tree(rf_model.estimators_[2], dt_model.feature_names_in_, save_path='RegressionModel/reports/figures/tree1.png')
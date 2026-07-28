# # Import required libraries for tuning
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.inspection import permutation_importance


# # Hyperparameter tuning using RandomizedSearchCV
# param_dist = {
#     "n_estimators": [100, 200, 300, 400, 500],
#     "max_depth": [10, 20, 30, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["auto", "sqrt", "log2"],
#     "bootstrap": [True, False],
# }

# # Initialize the model
# rf_model = RandomForestRegressor(random_state=42)

# # Randomized Search
# print("\nTuning Hyperparameters...")
# random_search = RandomizedSearchCV(
#     estimator=rf_model,
#     param_distributions=param_dist,
#     n_iter=50,  # Number of combinations to test
#     cv=3,  # 3-fold cross-validation
#     scoring="r2",  # R-squared as evaluation metric
#     verbose=1,
#     random_state=42,
#     n_jobs=-1,
# )
# random_search.fit(X_train, y_train)

# # Update the model with the best parameters
# rf_model = random_search.best_estimator_
# print("\nBest Parameters Found:")
# print(random_search.best_params_)

# # Train the model with tuned parameters
# print("\nRetraining the Model...")
# rf_model.fit(X_train, y_train)

# # Predict with the tuned model
# y_pred_train = rf_model.predict(X_train)
# y_pred_test = rf_model.predict(X_test)

# # Evaluate the tuned model
# print("\nModel Evaluation on Training Data (After Tuning):")
# print(f"Mean Squared Error: {mean_squared_error(y_train, y_pred_train):.2f}")
# print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train):.2f}")
# print(f"R-squared Score: {r2_score(y_train, y_pred_train):.2f}")

# print("\nModel Evaluation on Test Data (After Tuning):")
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_test):.2f}")
# print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.2f}")
# print(f"R-squared Score: {r2_score(y_test, y_pred_test):.2f}")

# # Re-analyze feature importance using permutation importance
# print("\nAnalyzing Feature Importance...")
# perm_importance = permutation_importance(
#     rf_model, X_test, y_test, n_repeats=10, random_state=42
# )
# importance_df = pd.DataFrame(
#     {"Feature": X.columns, "Importance": perm_importance.importances_mean}
# ).sort_values(by="Importance", ascending=False)

# # Plot updated feature importance
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
# plt.title("Updated Feature Importance After Tuning")
# plt.xlabel("Importance")
# plt.ylabel("Features")
# plt.show()

# # Cross-check feature dependency
# print("\nFeatures Highly Dependent on Model:")
# print(importance_df.head())

# # Visualize predicted vs actual ratings
# plt.figure(figsize=(8, 8))
# sns.scatterplot(x=y_test, y=y_pred_test, color="blue", alpha=0.7)
# plt.title("Predicted vs Actual Aircraft Ratings (After Tuning)")
# plt.xlabel("Actual Ratings")
# plt.ylabel("Predicted Ratings")
# plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Reference line
# plt.grid()
# plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset to check correlations
dataset_path = "average_pilot_dataset.csv"
dataset = pd.read_csv(dataset_path)

# Drop non-numeric and identifier columns for correlation calculation
numeric_dataset = dataset.drop(columns=["Pilot ID", "Performance Level"], errors="ignore")

# Calculate the correlation matrix
correlation_matrix = numeric_dataset.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Pilot Dataset Features")
plt.tight_layout()

plt.show()
correlation_matrix

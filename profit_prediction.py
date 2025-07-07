

# Importing pandas for data manipulation and analysis
import pandas as pd

# Importing numpy for numerical operations
import numpy as np

# Importing matplotlib for data visualization
import matplotlib.pyplot as plt

# Importing seaborn for statistical data visualization
import seaborn as sns

# Importing train_test_split to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Importing LinearRegression for building a linear regression model
from sklearn.linear_model import LinearRegression  # ← Comment added here

# Importing evaluation metrics for model performance
from sklearn.metrics import mean_squared_error, r2_score

# Importing SimpleImputer to handle missing values
from sklearn.impute import SimpleImputer

# Importing encoders to convert categorical data into numeric form
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Importing ColumnTransformer to apply different preprocessing to different columns
from sklearn.compose import ColumnTransformer

# Importing Pipeline to combine preprocessing and modeling into one workflow
from sklearn.pipeline import Pipeline

df= pd.read_csv('profit.csv') # load dataset

df.head() # display the 5 five rows

df.isnull().sum() # check for null column

df.duplicated().sum() # check for dupicate rows

















df.info() # check for information

df.describe() # describe the dataset



df = df.drop_duplicates() # drop duplicate row

df.duplicated().sum()

# Set seaborn style
sns.set(style='whitegrid', palette='pastel')

# Enhanced pairplot
plt.figure(figsize=(12, 8))
sns.pairplot(df, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'k'})

plt.suptitle('Pair Plot of Features', fontsize=16, y=1.02)
plt.show()

# Set the visual style
sns.set(style='whitegrid')

# Create the pairplot
plt.figure(figsize=(10, 8))  # This won't affect pairplot directly, but it's fine to include
pair = sns.pairplot(
    df,
    kind='scatter',
    plot_kws={'alpha': 0.5, 's': 60, 'edgecolor': 'k'},
    corner=True,
    diag_kind='kde',
    palette='husl'
)

# Add a title (optional)
pair.fig.suptitle("Pairplot of Features", y=1.02, fontsize=16)

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set(style='whitegrid')

# Create a scatter plot for R&D Spend vs Profit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='R&D Spend', y='Profit', color='teal', alpha=0.6, edgecolor='k')

# Add title and labels
plt.title('R&D Spend vs Profit', fontsize=14)
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.tight_layout()
plt.show()



from sklearn.preprocessing import StandardScaler


#  Separate features (X) and target (y)
X = df.drop('Profit', axis=1)   # replace 'Profit' with your target column
y = df['Profit']

# Identify column types
categorical = X.select_dtypes(include='object').columns.tolist()
numeric = X.select_dtypes(include='number').columns.tolist()

#  Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Define the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ]
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

#  Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)

print("Categorical columns:", categorical)
print("Numeric columns:", numeric)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
y_pred = pipeline.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8, 6))
# Set the size of the plot figure

sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
# Create a scatter plot to compare actual vs predicted values

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
# Add a dashed red line to show the ideal case where predicted == actual

plt.xlabel('Actual Values')
# Label the x-axis as actual values

plt.ylabel('Predicted Values
# Label the y-axis as predicted values

plt.title('Predicted vs Actual Values')
# Add a title to the plot

plt.grid(True)
# Show grid lines for better readability

plt.show()
# Display the plot


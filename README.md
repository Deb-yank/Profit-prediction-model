# Profit Prediction Model

A machine learning project that predicts company profits using linear regression based on various business metrics including R&D spend, marketing spend, and administrative costs.

## Overview

This project implements a profit prediction model using scikit-learn's Linear Regression algorithm. The model processes both categorical and numerical features with proper scaling and encoding to predict company profits based on business investment data.

## Dataset

The model expects a CSV file named `profit.csv` with the following structure:
- **Numerical Features**: R&D Spend, Marketing Spend, Administration costs, etc.
- **Categorical Features**: Company location, industry type, etc.
- **Target Variable**: `Profit` - the company's profit to be predicted

## Features

### Data Preprocessing
- **Missing Value Detection**: Identifies and handles null values
- **Duplicate Removal**: Eliminates duplicate rows from the dataset
- **Feature Scaling**: StandardScaler applied to numerical features
- **Categorical Encoding**: One-hot encoding for categorical variables

### Exploratory Data Analysis
- **Pair Plots**: Comprehensive visualization of feature relationships
- **Correlation Analysis**: Understanding relationships between variables
- **Scatter Plots**: Specific analysis of R&D Spend vs Profit relationship

### Model Pipeline
- Uses scikit-learn Pipeline for streamlined preprocessing and modeling
- Combines ColumnTransformer for feature preprocessing with LinearRegression
- Automatic handling of different data types (numerical and categorical)

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Ensure your dataset file `profit.csv` is in the same directory as the script

## Usage

```python
python profit_prediction.py
```

The script will:
1. Load and explore the dataset
2. Perform data cleaning and preprocessing
3. Generate visualizations for data understanding
4. Split data into training (80%) and testing (20%) sets
5. Train a linear regression model with proper preprocessing
6. Evaluate model performance with multiple metrics
7. Display results and prediction visualization

## Model Performance Metrics

The model evaluates performance using:
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
- **Root Mean Squared Error (RMSE)**: Square root of average squared differences
- **R² Score**: Coefficient of determination (proportion of variance explained)

## Visualizations

### 1. Pair Plot
- Shows relationships between all numerical features
- Includes kernel density estimation on diagonal
- Helps identify correlations and patterns

### 2. R&D Spend vs Profit Scatter Plot
- Specific analysis of R&D investment impact on profit
- Styled with seaborn for professional appearance

### 3. Predicted vs Actual Values
- Scatter plot comparing model predictions to actual profits
- Red dashed line shows perfect prediction baseline
- Visual assessment of model accuracy

## File Structure

```
project/
├── profit_prediction.py    # Main script
├── profit.csv             # Dataset (required)
└── README.md              # This file
```

## Model Architecture

1. **Data Loading & Exploration**: 
   - Load CSV data using pandas
   - Display basic statistics and info
   - Check for missing values and duplicates

2. **Data Visualization**:
   - Generate pair plots for feature relationships
   - Create specific scatter plots for key relationships

3. **Feature Engineering**:
   - Automatic separation of numerical and categorical features
   - StandardScaler for numerical features (normalization)
   - OneHotEncoder for categorical features

4. **Model Training**:
   - 80/20 train-test split
   - Pipeline-based approach for consistent preprocessing
   - Linear regression model fitting

5. **Evaluation**:
   - Multiple performance metrics calculation
   - Visual comparison of predictions vs actual values

## Data Pipeline Details

### Preprocessing Steps:
1. **Numerical Features**: Standardized using StandardScaler
2. **Categorical Features**: One-hot encoded with first category dropped
3. **Target Variable**: Profit (no transformation applied)

### Model Pipeline:
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])),
    ('model', LinearRegression())
])
```

## Performance Interpretation

- **MAE**: Lower values indicate better average prediction accuracy
- **RMSE**: Lower values indicate better overall prediction quality
- **R² Score**: Higher values (closer to 1.0) indicate better model fit
- **Scatter Plot**: Points closer to the red diagonal line indicate better predictions

## Customization Options

### Modify the model:
- Change the algorithm: Replace `LinearRegression()` with other models
- Adjust preprocessing: Modify scaling or encoding strategies
- Change train-test split: Adjust `test_size` parameter
- Add feature engineering: Create new features from existing ones

### Visualization customization:
- Modify seaborn styles and color palettes
- Adjust plot sizes and layouts
- Add additional plots for deeper analysis

## Code Quality Features

- **Comprehensive Comments**: Every major step is documented
- **Modular Structure**: Clear separation of preprocessing and modeling
- **Error Handling**: Automatic detection of feature types
- **Reproducible Results**: Fixed random state for consistent outputs

## Troubleshooting

### Common Issues:
1. **File Not Found**: Ensure `profit.csv` is in the same directory
2. **Missing Dependencies**: Install all required packages
3. **Data Format Issues**: Check that your CSV has the expected structure
4. **Memory Issues**: For large datasets, consider sampling or chunking

### Data Requirements:
- CSV file with profit as target variable
- Mix of numerical and categorical features supported
- No specific column names required (auto-detected)

## Future Enhancements

- **Advanced Models**: Random Forest, XGBoost, Neural Networks
- **Feature Selection**: Automated feature importance analysis
- **Cross-Validation**: More robust model evaluation
- **Hyperparameter Tuning**: Grid search or random search
- **Time Series Analysis**: If profit data has temporal component
- **Interactive Visualizations**: Plotly or Bokeh integration

## Contributing

Feel free to fork this project and submit pull requests for improvements such as:
- Additional evaluation metrics
- New visualization types
- Model improvements
- Code optimization
- Documentation enhancements

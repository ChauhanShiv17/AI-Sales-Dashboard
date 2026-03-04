# ==========================================
# IMPORTS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/superstore.csv")

print("Initial Shape:", df.shape)

# Convert date columns
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

df['Postal Code'] = df['Postal Code'].fillna(0)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nFinal Shape after cleaning:", df.shape)

print("\nData Types:\n")
print(df.dtypes)

# ==========================================
# MONTHLY SALES TREND
# ==========================================

df['Month-Year'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month-Year')['Sales'].sum()

print("\nMonthly Sales:\n")
print(monthly_sales)

plt.figure(figsize=(12,6))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month-Year")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# TOP 10 PRODUCTS
# ==========================================

top_products = (
    df.groupby('Product Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products:\n")
print(top_products)

plt.figure(figsize=(12,6))
top_products.plot(kind='bar')
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Product Name")
plt.ylabel("Total Sales")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# ==========================================
# CUSTOMER SEGMENTATION (K-MEANS)
# ==========================================

customer_data = df.groupby('Customer ID')['Sales'].sum().reset_index()
customer_data.columns = ['Customer ID', 'Total_Sales']

kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['Total_Sales']])

print("\nCustomer Segmentation Sample:\n")
print(customer_data.head())

plt.figure(figsize=(10,4))
plt.scatter(customer_data['Total_Sales'], 
            np.zeros(len(customer_data)), 
            c=customer_data['Cluster'])
plt.title("Customer Segmentation based on Total Sales")
plt.xlabel("Total Sales")
plt.yticks([])
plt.show()

# ==========================================
# CLUSTER ANALYSIS
# ==========================================

cluster_summary = customer_data.groupby('Cluster')['Total_Sales'].mean()

print("\nAverage Sales per Cluster:\n")
print(cluster_summary)

cluster_mapping = {
    1: "High Value",
    0: "Medium Value",
    2: "Low Value"
}

customer_data["Customer_Segment"] = customer_data["Cluster"].map(cluster_mapping)

print("\nCustomer Segments Distribution:\n")
print(customer_data["Customer_Segment"].value_counts())

# ==========================================
# SALES FORECASTING (LINEAR REGRESSION)
# ==========================================

monthly_sales_df = monthly_sales.reset_index()
monthly_sales_df["Time_Index"] = np.arange(len(monthly_sales_df))

X = monthly_sales_df[["Time_Index"]]
y = monthly_sales_df["Sales"]

model = LinearRegression()
model.fit(X, y)

print("\nModel Trained Successfully")

# Model evaluation
r2 = r2_score(y, model.predict(X))
print("R² Score:", round(r2, 4))
print("Monthly Growth (Slope):", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))

# ==========================================
# FUTURE PREDICTIONS
# ==========================================

future_time_index = pd.DataFrame({
    "Time_Index": np.arange(len(monthly_sales_df), 
                            len(monthly_sales_df) + 6)
})

future_predictions = model.predict(future_time_index)

print("\nNext 6 Months Sales Forecast:\n")
for i, pred in enumerate(future_predictions, 1):
    print(f"Month +{i}: {round(pred, 2)}")

# ==========================================
# FORECAST PLOT
# ==========================================

plt.figure(figsize=(12,6))

# Actual data
plt.plot(monthly_sales_df["Time_Index"], y, label="Actual Sales")

# Forecast line (continuous)
all_time_index = pd.concat(
    [monthly_sales_df["Time_Index"], future_time_index["Time_Index"]]
)

all_predictions = model.predict(
    pd.DataFrame({"Time_Index": all_time_index})
)

plt.plot(all_time_index, all_predictions, 
         linestyle="--", label="Trend + Forecast")

plt.xlabel("Time Index (Months)")
plt.ylabel("Sales")
plt.title("Sales Forecast (Next 6 Months)")
plt.legend()
plt.show()
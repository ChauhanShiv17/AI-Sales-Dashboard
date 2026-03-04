import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Sales Dashboard", layout="wide")

st.title("📊 AI Sales Dashboard")

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    return df

df = load_data()

# =====================================
# KPI SECTION
# =====================================

total_sales = df["Sales"].sum()
total_customers = df["Customer ID"].nunique()
total_orders = df["Order ID"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"{total_sales:,.0f}")
col2.metric("👥 Total Customers", total_customers)
col3.metric("📦 Total Orders", total_orders)

st.divider()

# =====================================
# MONTHLY SALES TREND
# =====================================

st.subheader("📈 Monthly Sales Trend")

df['Month-Year'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month-Year')['Sales'].sum()

fig1, ax1 = plt.subplots(figsize=(10,4))
monthly_sales.plot(ax=ax1)
ax1.set_xlabel("Month")
ax1.set_ylabel("Sales")
ax1.set_title("Monthly Sales Trend")
st.pyplot(fig1)

# =====================================
# TOP PRODUCTS
# =====================================

st.subheader("🏆 Top 10 Products")

top_products = (
    df.groupby('Product Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots(figsize=(10,4))
top_products.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Sales")
ax2.set_title("Top 10 Best-Selling Products")
st.pyplot(fig2)

# =====================================
# CUSTOMER SEGMENTATION
# =====================================

st.subheader("👥 Customer Segmentation")

customer_data = df.groupby('Customer ID')['Sales'].sum().reset_index()
customer_data.columns = ['Customer ID', 'Total_Sales']

kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['Total_Sales']])

cluster_summary = customer_data.groupby('Cluster')['Total_Sales'].mean()

cluster_mapping = {
    cluster_summary.idxmax(): "High Value",
    cluster_summary.idxmin(): "Low Value"
}

for c in cluster_summary.index:
    if c not in cluster_mapping:
        cluster_mapping[c] = "Medium Value"

customer_data["Segment"] = customer_data["Cluster"].map(cluster_mapping)

segment_counts = customer_data["Segment"].value_counts()

fig3, ax3 = plt.subplots()
ax3.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%')
ax3.set_title("Customer Segments Distribution")
st.pyplot(fig3)

# =====================================
# SALES FORECAST
# =====================================

st.subheader("🔮 6-Month Sales Forecast")

monthly_sales_df = monthly_sales.reset_index()
monthly_sales_df["Time_Index"] = np.arange(len(monthly_sales_df))

X = monthly_sales_df[["Time_Index"]]
y = monthly_sales_df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_time_index = pd.DataFrame({
    "Time_Index": np.arange(len(monthly_sales_df),
                            len(monthly_sales_df) + 6)
})

future_predictions = model.predict(future_time_index)

fig4, ax4 = plt.subplots(figsize=(10,4))

ax4.plot(monthly_sales_df["Time_Index"], y, label="Actual")
ax4.plot(future_time_index["Time_Index"], future_predictions,
         linestyle="--", label="Forecast")

ax4.set_title("Sales Forecast")
ax4.legend()

st.pyplot(fig4)

st.success("Dashboard Loaded Successfully 🚀")
import streamlit as st
import pandas as pd

# Initialize session state to store sales data
if "sales_data" not in st.session_state:
    st.session_state.sales_data = pd.DataFrame(columns=["Product", "Category", "Sales"])

st.title("🛒 D-Mart Product Sales Tracker")

# Input section
st.subheader("Add New Product Sale")
product = st.text_input("Product Name:")
category = st.text_input("Category:")
sales = st.number_input("Sales Quantity:", min_value=0, step=1)

if st.button("Add Sale"):
    if product and category and sales > 0:
        new_entry = pd.DataFrame([[product, category, sales]],
                                 columns=["Product", "Category", "Sales"])
        st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_entry], ignore_index=True)
        st.success(f"✅ Added: {product} ({category}) - {sales} units")
    else:
        st.warning("⚠️ Please fill all fields correctly.")

# Show data
st.subheader("📊 Sales Data")
st.dataframe(st.session_state.sales_data)

if not st.session_state.sales_data.empty:
    # Best selling product
    best_product = st.session_state.sales_data.groupby("Product")["Sales"].sum().idxmax()
    best_sales = st.session_state.sales_data.groupby("Product")["Sales"].sum().max()

    # Best category
    best_category = st.session_state.sales_data.groupby("Category")["Sales"].sum().idxmax()
    best_cat_sales = st.session_state.sales_data.groupby("Category")["Sales"].sum().max()

    st.markdown(f"🔥 **Best Selling Product:** {best_product} ({best_sales} units)")
    st.markdown(f"🏆 **Top Category:** {best_category} ({best_cat_sales} units)")

    # Visualization
    st.subheader("📈 Product-wise Sales")
    st.bar_chart(st.session_state.sales_data.groupby("Product")["Sales"].sum())

    st.subheader("📊 Category-wise Sales")
    st.bar_chart(st.session_state.sales_data.groupby("Category")["Sales"].sum())

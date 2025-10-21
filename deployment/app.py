import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

# Title and description
st.title("🛒 SuperKart Sales Forecasting System")
st.markdown("### Predict product sales using Machine Learning")
st.markdown("---")

# Cache the model loading
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders from Hugging Face"""
    try:
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id="aksace/superkart-sales-forecasting-model",
            filename="best_model.pkl"
        )
        
        # Download label encoders
        encoders_path = hf_hub_download(
            repo_id="aksace/superkart-sales-forecasting-model",
            filename="label_encoders.pkl"
        )
        
        # Load model and encoders
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and encoders
with st.spinner("Loading model..."):
    model, label_encoders = load_model_and_encoders()

if model is None or label_encoders is None:
    st.error("Failed to load model. Please check your Hugging Face repository.")
    st.stop()

st.success("✓ Model loaded successfully!")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Product Information")
    
    # Product Weight
    product_weight = st.number_input(
        "Product Weight (kg)",
        min_value=0.0,
        max_value=50.0,
        value=12.5,
        step=0.1
    )
    
    # Product Sugar Content
    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        options=['Low Sugar', 'Regular', 'No Sugar']
    )
    
    # Product Allocated Area
    product_allocated_area = st.slider(
        "Product Allocated Area (ratio)",
        min_value=0.0,
        max_value=0.3,
        value=0.05,
        step=0.001
    )
    
    # Product Type
    product_type = st.selectbox(
        "Product Type",
        options=['Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 
                'Health and Hygiene', 'Snack Foods', 'Meat', 'Household',
                'Hard Drinks', 'Fruits and Vegetables', 'Breads', 'Soft Drinks',
                'Breakfast', 'Others', 'Starchy Foods', 'Seafood']
    )
    
    # Product MRP
    product_mrp = st.number_input(
        "Product MRP ($)",
        min_value=0.0,
        max_value=300.0,
        value=150.0,
        step=1.0
    )

with col2:
    st.subheader("Store Information")
    
    # Store Size
    store_size = st.selectbox(
        "Store Size",
        options=['Small', 'Medium', 'High']
    )
    
    # Store Location City Type
    store_location = st.selectbox(
        "Store Location City Type",
        options=['Tier 1', 'Tier 2', 'Tier 3']
    )
    
    # Store Type
    store_type = st.selectbox(
        "Store Type",
        options=['Supermarket Type1', 'Supermarket Type2', 
                'Departmental Store', 'Food Mart']
    )
    
    # Store Age
    store_age = st.number_input(
        "Store Age (years)",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )
    
    # Price Category
    if product_mrp <= 69:
        price_category = 'Low'
    elif product_mrp <= 136:
        price_category = 'Medium'
    elif product_mrp <= 204:
        price_category = 'High'
    else:
        price_category = 'Very High'
    
    st.info(f"**Calculated Price Category:** {price_category}")

# Prediction button
st.markdown("---")
if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Product_Weight': [product_weight],
            'Product_Sugar_Content': [product_sugar_content],
            'Product_Allocated_Area': [product_allocated_area],
            'Product_Type': [product_type],
            'Product_MRP': [product_mrp],
            'Store_Size': [store_size],
            'Store_Location_City_Type': [store_location],
            'Store_Type': [store_type],
            'Store_Age': [store_age],
            'Price_Category': [price_category]
        })
        
        # Encode categorical variables
        for col in ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 
                   'Store_Location_City_Type', 'Store_Type', 'Price_Category']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            st.metric(
                label="Predicted Sales",
                value=f"${prediction:,.2f}"
            )
        
        with col_pred2:
            daily_sales = prediction / 365
            st.metric(
                label="Estimated Daily Sales",
                value=f"${daily_sales:,.2f}"
            )
        
        with col_pred3:
            monthly_sales = prediction / 12
            st.metric(
                label="Estimated Monthly Sales",
                value=f"${monthly_sales:,.2f}"
            )
        
        # Additional insights
        st.markdown("---")
        st.subheader("💡 Insights")
        
        if prediction > 5000:
            st.success("🎯 **High Sales Expected!** This product-store combination is predicted to perform excellently.")
        elif prediction > 3000:
            st.info("📈 **Good Sales Expected!** This product-store combination shows promising potential.")
        else:
            st.warning("⚠️ **Moderate Sales Expected.** Consider adjusting product placement or pricing strategy.")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Powered by Random Forest Regressor | Model R² Score: 0.9319</p>
        <p>SuperKart Sales Forecasting System © 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)

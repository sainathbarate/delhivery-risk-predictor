import streamlit as st
import pandas as pd
import joblib

# Load saved models
model_delayed = joblib.load('model_delayed.pkl')
model_rto = joblib.load('model_rto.pkl')
model_ontime = joblib.load('model_ontime.pkl')
model_days = joblib.load('model_days.pkl')
model_columns = joblib.load('model_columns.pkl')

# Custom CSS for black/white/red theme
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #FF3D3D;
        color: black;
        font-size: 20px;
        font-weight: 700;
        padding: 12px 40px;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #E62E2E;
        color: black;
    }
    div[data-baseweb="select"] {
        background-color: #000000 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        color: white !important;
        border-radius: 6px;
    }
    input {
        background-color: #000000 !important;
        color: white !important;
        border-radius: 6px;
    }
    li[role="option"] {
        color: white !important;
        background-color: #000000 !important;
    }
    [data-testid="stMetricValue"] {
        color: white !important;
    }
    svg {
        fill: white !important;
    }
    button[data-testid="stNumberInputStepDown"] svg,
    button[data-testid="stNumberInputStepUp"] svg {
        fill: black !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: white;'>Delhivery Shipment Risk Predictor</h1>", unsafe_allow_html=True)
st.write("Enter shipment details to predict delay, RTO, and delivery time risk.")

st.header("Shipment Details")

col_left, col_right = st.columns(2)

with col_left:
    product_category = st.selectbox("Product Category", 
        ['Clothing & Apparel', 'Electronics', 'Furniture & Home Decor', 
         'Grocery & FMCG', 'Pharma & Health', 'Sports & Fitness'])

    weight_kg = st.number_input("Weight (kg)", min_value=0.01, value=1.0)

    destination_city_list = ['Ahmedabad', 'Amritsar', 'Bangalore', 'Bhopal', 'Bhubaneswar', 
        'Bikaner', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Guwahati', 'Hampi', 
        'Hyderabad', 'Indore', 'Jaipur', 'Jodhpur', 'Kochi', 'Kolkata', 'Leh', 'Lucknow', 
        'Madurai', 'Mysore', 'Nagpur', 'Nashik', 'Patna', 'Pune', 'Ranchi', 'Silchar', 
        'Srinagar', 'Surat', 'Varanasi', 'Vijayawada', 'Agra']
    destination_city = st.selectbox("Destination City", destination_city_list)

    distance_band = st.selectbox("Distance Band", ['Short', 'Medium', 'Long', 'Very Long'])

    customer_type = st.selectbox("Customer Type", ['New', 'Returning'])

with col_right:
    package_size = st.selectbox("Package Size", ['Small', 'Medium', 'Large', 'Oversized'])

    order_value = st.number_input("Order Value (INR)", min_value=1.0, value=1000.0)

    destination_tier = st.selectbox("Destination Tier", ['Metro', 'Tier 2', 'Tier 3', 'Rural'])

    payment_type = st.selectbox("Payment Type", ['COD', 'Prepaid'])

    order_date = st.date_input("Order Date")

is_sale_season = st.checkbox("Is Sale Season?")


col_a, col_b, col_c = st.columns([1,1,1])
with col_b:
    predict_clicked = st.button("Predict Risk", use_container_width=True)

if predict_clicked:    
    input_dict = {col: 0 for col in model_columns}
    
    input_dict['weight_kg'] = weight_kg
    input_dict['order_value'] = order_value
    input_dict['payment_type'] = 1 if payment_type == 'Prepaid' else 0
    input_dict['customer_type'] = 1 if customer_type == 'Returning' else 0
    input_dict['is_sale_season'] = 1 if is_sale_season else 0
    input_dict['order_month'] = order_date.month
    input_dict['order_day_of_week'] = order_date.weekday()
    input_dict['is_weekend'] = 1 if order_date.weekday() in [5,6] else 0
    
    cat_col = f'product_category_{product_category}'
    if cat_col in input_dict:
        input_dict[cat_col] = 1
    
    tier_col = f'destination_tier_{destination_tier}'
    if tier_col in input_dict:
        input_dict[tier_col] = 1
    
    dist_col = f'distance_band_{distance_band}'
    if dist_col in input_dict:
        input_dict[dist_col] = 1
    
    pkg_col = f'package_size_{package_size}'
    if pkg_col in input_dict:
        input_dict[pkg_col] = 1
    
    city_col = f'destination_city_{destination_city}'
    if city_col in input_dict:
        input_dict[city_col] = 1
    
    input_df = pd.DataFrame([input_dict])[model_columns]
    
    delay_risk = model_delayed.predict_proba(input_df)[:,1][0] * 100
    rto_risk = model_rto.predict_proba(input_df)[:,1][0] * 100
    ontime_chance = model_ontime.predict_proba(input_df)[:,1][0] * 100
    predicted_days = model_days.predict(input_df)[0]
    
    st.markdown("<h2 style='color: white;'>Prediction Results</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    def risk_color(pct, reverse=False):
        if reverse:
            if pct >= 60: return "🟢"
            elif pct >= 35: return "🟡"
            else: return "🔴"
        else:
            if pct >= 50: return "🔴"
            elif pct >= 25: return "🟡"
            else: return "🟢"
    
    with col1:
        st.metric("Delay Risk", f"{delay_risk:.1f}%")
        st.write(risk_color(delay_risk))
    
    with col2:
        st.metric("RTO Risk", f"{rto_risk:.1f}%")
        st.write(risk_color(rto_risk))
    
    with col3:
        st.metric("On-Time Chance", f"{ontime_chance:.1f}%")
        st.write(risk_color(ontime_chance, reverse=True))
    
    with col4:
        st.metric("Predicted Days", f"{predicted_days:.1f}")
    
    st.markdown("---")
    
    if rto_risk >= 50 or delay_risk >= 40:
        st.error("⚠️ HIGH RISK — Recommend verification before dispatch (confirm address/phone, consider COD restrictions).")
    elif rto_risk >= 25 or delay_risk >= 20:
        st.warning("🟡 MODERATE RISK — Monitor this shipment, standard handling should be fine.")
    else:
        st.success("✅ LOW RISK — Safe to process normally.")
"""
Scorely - Credit Risk Prediction Web Application
CSCI 4050U Machine Learning Final Project

Deploy with: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Scorely - Credit Risk Prediction",
    page_icon="credit_card",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Model Definition (must match training)
# =====================================================
class CreditMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# =====================================================
# Load Model
# =====================================================
@st.cache_resource
def load_model():
    """Load the trained model and scaler parameters."""
    try:
        checkpoint = torch.load('scorely_model.pth', map_location='cpu', weights_only=False)
        
        model = CreditMLP(input_dim=checkpoint['input_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_mean = checkpoint['scaler_mean']
        scaler_scale = checkpoint['scaler_scale']
        feature_cols = checkpoint['feature_cols']
        
        return model, scaler_mean, scaler_scale, feature_cols, True
    except FileNotFoundError:
        return None, None, None, None, False

model, scaler_mean, scaler_scale, feature_cols, model_loaded = load_model()

# =====================================================
# Prediction Function
# =====================================================
def predict_risk(input_data, model, scaler_mean, scaler_scale):
    """Make a prediction for a single applicant."""
    features = np.array(list(input_data.values()), dtype=np.float32).reshape(1, -1)
    features_scaled = (features - scaler_mean) / scaler_scale
    
    with torch.no_grad():
        tensor = torch.tensor(features_scaled, dtype=torch.float32)
        logits = model(tensor)
        probability = torch.sigmoid(logits).item()
    
    return probability

def get_risk_category(probability):
    """Categorize risk level based on probability. Returns (label, color)."""
    if probability < 0.15:
        return "Very Low Risk", "#10B981"
    elif probability < 0.30:
        return "Low Risk", "#34D399"
    elif probability < 0.50:
        return "Moderate Risk", "#FBBF24"
    elif probability < 0.70:
        return "High Risk", "#F97316"
    else:
        return "Very High Risk", "#EF4444"

def risk_badge(label, color):
    """Generate HTML for a colored risk badge."""
    return f'''
    <div style="
        display: inline-block;
        background-color: {color};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1rem;
    ">{label}</div>
    '''

def status_dot(color):
    """Generate HTML for a colored status dot."""
    return f'''
    <span style="
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: {color};
        border-radius: 50%;
        margin-right: 8px;
    "></span>
    '''

# =====================================================
# Custom CSS
# =====================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-display {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .factor-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)





# Header
st.markdown('<h1 class="main-header">Scorely</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Credit Risk Assessment System</p>', unsafe_allow_html=True)

# Check if model is loaded
if not model_loaded:
    st.error("""
    **Model not found.** 
    
    Please ensure `scorely_model.pth` is in the same directory as this app.
    
    Run the training notebook first to generate the model file.
    """)
    st.info("**Running in Demo Mode** - Predictions are simulated")
    demo_mode = True
else:
    st.success("Model loaded successfully!")
    demo_mode = False

st.markdown("---")

# Two column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Applicant Information")
    
    # Personal info
    st.markdown("**Personal Details**")
    
    age = st.slider(
        "Age (years)", 
        min_value=18, 
        max_value=100, 
        value=35,
        help="Applicant's age in years"
    )
    
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of dependents (excluding self)"
    )
    
    st.markdown("**Financial Information**")
    
    monthly_income = st.number_input(
        "Monthly Income ($)",
        min_value=0,
        max_value=500000,
        value=5000,
        step=500,
        help="Gross monthly income"
    )
    
    debt_ratio = st.slider(
        "Debt Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="Monthly debt payments / Monthly income"
    )
    
    credit_utilization = st.slider(
        "Credit Card Utilization",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.01,
        help="Total credit card balance / Total credit limit"
    )
    
    st.markdown("**Credit History**")
    
    open_credit_lines = st.number_input(
        "Open Credit Lines & Loans",
        min_value=0,
        max_value=50,
        value=5,
        help="Number of open loans and credit lines"
    )
    
    real_estate_loans = st.number_input(
        "Real Estate Loans",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of mortgage and real estate loans"
    )
    
    st.markdown("**Payment History (Last 2 Years)**")
    
    late_30_59 = st.number_input(
        "Times 30-59 Days Late",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of times 30-59 days past due"
    )
    
    late_60_89 = st.number_input(
        "Times 60-89 Days Late",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of times 60-89 days past due"
    )
    
    late_90_plus = st.number_input(
        "Times 90+ Days Late",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of times 90 days or more past due"
    )

# Prepareing the input data
input_data = {
    'RevolvingUtilizationOfUnsecuredLines': credit_utilization,
    'age': age,
    'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
    'DebtRatio': debt_ratio,
    'MonthlyIncome': monthly_income,
    'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
    'NumberOfTimes90DaysLate': late_90_plus,
    'NumberRealEstateLoansOrLines': real_estate_loans,
    'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
    'NumberOfDependents': dependents
}

with col2:
    st.markdown("### Risk Assessment")
    
    # Predict button
    if st.button("Analyze Credit Risk", use_container_width=True):
        
        with st.spinner("Analyzing..."):
            # Make prediction
            if demo_mode:
                # Simulated prediction for demo
                late_total = late_30_59 + late_60_89 + late_90_plus
                probability = min(0.05 + late_total * 0.12 + max(0, credit_utilization - 0.5) * 0.3 + max(0, debt_ratio - 0.4) * 0.2, 0.95)
            else:
                probability = predict_risk(input_data, model, scaler_mean, scaler_scale)
            
            risk_category, color = get_risk_category(probability)
        
        # Display results
        st.markdown("---")
        
        # Main probability display
        badge_html = f'<span style="display: inline-block; background-color: {color}; color: white; padding: 0.5rem 1rem; border-radius: 2rem; font-weight: 600; font-size: 1rem;">{risk_category}</span>'
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {color}22 0%, {color}11 100%); border-radius: 1rem; border: 2px solid {color};">
            <h2 style="margin: 0; color: #374151;">Default Probability</h2>
            <p style="font-size: 4rem; font-weight: bold; margin: 0.5rem 0; color: {color};">{probability*100:.1f}%</p>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'steps': [
                    {'range': [0, 15], 'color': '#D1FAE5'},
                    {'range': [15, 30], 'color': '#A7F3D0'},
                    {'range': [30, 50], 'color': '#FEF3C7'},
                    {'range': [50, 70], 'color': '#FED7AA'},
                    {'range': [70, 100], 'color': '#FECACA'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#374151"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("### Recommendation")
        
        if probability < 0.15:
            st.success("""
            **APPROVE - Excellent Credit Profile**
            
            This applicant demonstrates very low default risk. Recommend approval with standard terms.
            """)
        elif probability < 0.30:
            st.success("""
            **APPROVE - Good Credit Profile**
            
            This applicant shows low default risk. Standard approval recommended.
            """)
        elif probability < 0.50:
            st.warning("""
            **REVIEW - Moderate Risk**
            
            This applicant has moderate default risk. Consider:
            - Requesting additional documentation
            - Lower credit limit
            - Higher interest rate
            """)
        elif probability < 0.70:
            st.error("""
            **CAUTION - High Risk**
            
            This applicant presents elevated default risk. Consider:
            - Declining the application
            - Requiring collateral or co-signer
            - Significantly reduced credit limit
            """)
        else:
            st.error("""
            **DECLINE - Very High Risk**
            
            This applicant has a very high probability of default. 
            Approval is not recommended without substantial risk mitigation.
            """)
        
        # Risk factors analysis
        st.markdown("### Key Risk Factors")
        
        factors = []
        if late_90_plus > 0:
            factors.append(("Severe delinquency", f"{late_90_plus} instances of 90+ days late", "#EF4444"))
        if late_60_89 > 0:
            factors.append(("Past due history", f"{late_60_89} instances of 60-89 days late", "#F97316"))
        if late_30_59 > 0:
            factors.append(("Minor delinquency", f"{late_30_59} instances of 30-59 days late", "#FBBF24"))
        if credit_utilization > 0.7:
            factors.append(("High credit utilization", f"{credit_utilization*100:.0f}%", "#EF4444"))
        elif credit_utilization > 0.5:
            factors.append(("Elevated credit utilization", f"{credit_utilization*100:.0f}%", "#F97316"))
        if debt_ratio > 0.5:
            factors.append(("High debt ratio", f"{debt_ratio*100:.0f}%", "#F97316"))
        if age < 25:
            factors.append(("Young applicant", "Limited credit history likely", "#FBBF24"))
        
        if factors:
            for label, detail, dot_color in factors:
                st.markdown(f"""
                <div class="factor-item">
                    {status_dot(dot_color)}<strong>{label}:</strong> {detail}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="factor-item">
                {status_dot("#10B981")}<strong>No significant risk factors identified</strong>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; padding: 1rem;">
    <p><strong>Scorely</strong> - Credit Risk Prediction System</p>
    <p>CSCI 4050U Machine Learning Final Project</p>
    <p style="font-size: 0.8rem;">This is an educational tool and should not be used for actual lending decisions.</p>
</div>
""", unsafe_allow_html=True)

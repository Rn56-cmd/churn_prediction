import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpathces

# Page Config
st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon = "📡",
    layout = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {'About': "Telco Customer Churn Prediction - End-to-End Data Science App"}
)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    return joblib.load(model_path)

data = load_model()
model = data['model']
model_name = data['model_name']
results = data['results']
feat_imp = data['feature_importance']
numeric_cols = data['numeric_cols']
cat_cols = data['cat_cols']


# Sidebar
st.sidebar.title("📡 Churn Predictor")
st.sidebar.markdown('---')
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🔍 Predict Churn",
    "📊 Model Performance",
    "📈 Feature Importance",
    "💡 Business Insights"
])


# Page 1 (Home)
if page == "🏠 Home":
    st.title("📡 Telco Customer Churn Prediction")
    st.markdown("#End-to-End Data Science Application - Real-Time Model Interference*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "7,043")
    col2.metric("Churn Rate", "26.5%")
    col3.metric("Best Model", model_name)
    col4.metric("AUC-ROC", f"{results[model_name]['AUC']:.4f}")

    st.markdown('---')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("About This App")
        st.write("""
        This application predicts whether a telecom customer will ** churn (leave)**
        based on their account information and service usage.
                 
        **Features used**
        - Demographic: Gender, Senior Citizen, Partner, Dependents
        - Services: Internet, Phone, Streaming, Security, Backup
        - Account: Contract type, Payment method, Monthly/Total charges
        - Tenure: How long the customer has been with the company
""")
        
    with col2:
        st.subheader("Pipeline Architecture")
        st.write("""
        '''
        Raw Data
            ↓
        Preprocessing
        ├── Numeric → Median Imputer → Standard Scaler
        └── Categorical → Mode Imputer → One-Hot Encoder
            ↓
        Gradient Boosting Classifier
            ↓
        Churn Prediction + Probability
        ```
""")
        
    st.markdown('---')
    st.subheader("Model Comparison Summary")
    summary = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{v['Accuracy']:.4f}" for v in results.values()],
        'F1 Score': [f"{v['F1']:.4f}" for v in results.values()],
        'AUC-ROC': [f"{v['AUC']:.4f}" for v in results.values()],
    })

    st.dataframe(summary, hide_index=True, use_container_width=True)


# Page 2 (Predict Churn) Real time inference
elif page == "🔍 Predict Churn":
    st.title("🔍 Real-Time Churn Prediction")
    st.markdown('---')
    st.subheader("Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographic**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ['No','Yes'])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.markdown("**Account Info**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two Year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check',
                                                  'Bank Transfer (automatic)', 'Credit card (automatic)'])
        
    with col2:
        st.markdown("**Phone Service**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        st.markdown("**Internet Services**")
        internet = st.selectbox("Internet Service", ["Fiber Optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])


    with col3:
        st.markdown("**Additional Services**")
        device_protect = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("**Charges**")
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total_charges = st.number_input("Total Charges($)", 0.0, 9000.0,
                                        float(monthly_charges*tenure), 1.0)
        
    st.markdown("---")

    if st.button("🔍 Predict Churn", use_container_width=True):
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            "MultipleLines": multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protect,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error(f"⚠️ **HIGH CHURN RISK**\nThis customer is likely to churn!")
            else:
                st.success(f"✅ **LOW CHURN RISK**\nThis customer is likely to stay!")

            churn_prob = probability[1] * 100
            stay_prob = probability[0] * 100
            st.metric("Churn Probability", f"{churn_prob:.1f}")
            st.metric("Stay Probability", f"{stay_prob:.1f}")


        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            colors = ['#55A868', '#C44E52']
            bars = ax.bar(['Will Stay', "Will Churn"],
                            [stay_prob, churn_prob],
                            color=colors, edgecolor='white', width=0.5)
            ax.set_ylim(0,100)
            ax.set_ylabel('Churn Probability', fontweight='bold')
            for bar, val in zip(bars, [stay_prob, churn_prob]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f"{val:.1f}%", ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Risk Level
        st.markdown("---")
        st.subheader("Risk Assesment")
        if churn_prob >= 70:
            st.error("🔴 **CRITICAL RISK** - Immediate retention action needed!")
            st.write("**Recommended actions:** Offer discount, upgrade contract to yearly, as sign dedicated account manager.")
        elif churn_prob >= 40:
            st.warning("🔴 **Medium Risk** - Monitor closely and engage proactively.")
            st.write("**Recommended actions:** Send loyalty rewards, offer tech support upgrade, check service satisfaction.")
        else:
            st.success("🟢 **Low Risk** - Customer is likely satisfied. keep up the good service!")
            st.write("**Recommended actions:** Continue current service level, upsell premium features.")


# Page 3 Model Performance
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")

    #metrics comparison bar chart
    model_names = list(results.keys())
    metrics = ['Accuracy', 'F1', 'AUC']
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    for ax, metric, color in zip(axes, metrics, colors):
        vals = [results[n][metric] for n in model_names]
        bars = ax.bar(model_names, vals, color=color, edgecolor='white')
        ax.set_title(metric, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('---')

    # confusion matrices
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['CM'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Stay', 'Churn'],
                    yticklabels=['Stay', 'Churn'], ax=ax)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("""
    **Metric Guide:**
    - ** Accuracy** - Overall correct predictions
    - ** F1 Score** - Balance between precision and recall (important for imbalanced data)
    - **AUC-ROC** - How well the model separates churners from non-churners (higher = better)
""")
    
# Page 4 - Feature importance
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    st.markdown('---')

    st.subheader("Top 15 Most Influential Features (Random Forest)")

    top_n = st.slider("Show top N Features", 5, 15, 15)
    display_imp = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(10,6))
    colors_bar = ['#C44E52' if i < 3 else '#4C72B0'
                  for i in range(len(display_imp))]
    ax.barh(display_imp['feature'].values[::-1],
            display_imp['importance'].values[::-1],
            color=colors_bar[::-1], edgecolor='white')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance - What Drives Churn', fontweight='bold')

    red_patch = mpathces.Patch(color='#CC4E52', label='Top 3 features')
    blue_patch = mpathces.Patch(color="#4C72B0", label='Other features')
    ax.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        display_imp.rename(columns={
            'feature': 'Feature', 'importance': 'Importance Score'
        }).reset_index(drop=True),
        use_container_width=True
    )

    st.info("""
    **Key Findings:**
    - **TotalChargers, Tenure, MonthlyCharges** are the top 3 drivers - long-term & high-paying
            customers matter most
    - **Month-to-month contracts** significantly increase churn risk
    - **Electronics check payment** is associated with higher churn
    - **No Online Security / Tech Support** correlates with churn
""")
    

# Page 5 - Business Insight
elif page == "💡 Business Insights":
    st.title("💡 Business Insights")
    st.markdown('---')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Rate", "26.5%")
    col2.metric("At-Risk Customer", "-1,869")
    col3.metric("Model AUC-ROC", f"{results[model_name]['AUC']:.4f}")
    col4.metric("Best Model", model_name)

    st.markdown('---')

    st.subheader("🎯 Actionable Business strategies")

    with st.container(border=True):
        st.markdown("**1. 📄 Target Month-to-Month Contract Customers**")
        st.write("""
        Month-to-Month contract is one of the top churn indicators.
        **Strategy:** Offer discounted annual/bi-annual contracts with incentives
        (e.g. 1 month free, lower monthly rate) to convert short-term customers.
""")
        
    with st.container(border=True):
        st.markdown("**2. 💳 Electronic Check users Need Attention**")
        st.write("""
        Customers paying via electronics check have higher churn rates.
        **Strategy:** Offer a discounted cashback for switching to automatic
        bank transfer or credit card payment - also reduces payment failures.
""")
        
    with st.container(border=True):
        st.markdown("**3. 🔐 Bundle Security & Support Services*")
        st.write("""
        Customers without Online Security and Tech Support are more likely to leave.
        **Strategy:** Bundle these services at a reduced rate for new customers,
        or offer a free 3-month trial to increase stickiness.
""")
        
    with st.container(border=True):
        st.markdown("**4. 🌱 Focus on Early Tenure Customers**")
        st.write("""
        Low tenure is strongly linked to churn - customers are most vunerable in their first Year.
        **Strategy:** Implement a 90-day onboarding program with check-in calls,
        tutorials, and loyalty rewarsds to improve early retention.
""")
        
    with st.container(border=True):
        st.markdown("**5. 💰 High Monthly Charges = Higher Risk**")
        st.write("""
        Customers paying high monthly charges but not seeing value are at risk.
        **Strategy:** Proactively offer plan reviews, personalised bundles,
        and usage-based pricing to retain high-value customers.
""")
        
    st.markdown('---')
    st.subheader("📊 Estimated Business Impact")

    col1, col2 = st.columns(2)
    avg_revenue = 65.0
    churners = 1869

    with col1:
        st.metric("Avg Monthly Revenue/Customers", f"${avg_revenue}")
        st.metric("Est. Monthly Revenue at Risk", f"${churners*avg_revenue:,.0f}")

    with col2:
        retention_rate = 0.25
        saved = churners*retention_rate*avg_revenue
        st.metric("If 25% Retained (monthly)", f"${saved:,.0f} saved")
        st.metric("Annual Savings Potential", f"${saved*12:,.0f}")
        
    
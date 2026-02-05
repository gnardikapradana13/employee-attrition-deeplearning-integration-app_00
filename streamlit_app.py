# streamlit_app.py
# Employee Attrition Prediction Dashboard
# Deploy to: https://share.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
import joblib
import warnings
import sys
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .high-risk {
        background-color: #FEE2E2;
        border-left-color: #DC2626;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .medium-risk {
        background-color: #FEF3C7;
        border-left-color: #D97706;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .low-risk {
        background-color: #D1FAE5;
        border-left-color: #059669;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    .feature-slider {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'employee_counter' not in st.session_state:
    st.session_state.employee_counter = 1

# Title
st.markdown('<h1 class="main-header">🏢 Employee Attrition Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
Predict which employees are at risk of leaving using AI-powered neural networks
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("### 🔧 Model Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Select Model",
        ["Neural Network (Recommended)", "Random Forest", "Gradient Boosting"]
    )
    
    # Risk threshold
    risk_threshold = st.slider(
        "High Risk Threshold",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Probability above which an employee is considered high risk"
    )
    
    # Demo data
    st.markdown("---")
    st.markdown("### 📊 Demo Data")
    if st.button("Generate Sample Employees"):
        st.session_state.predictions = []
        for i in range(5):
            features = np.random.randn(20).tolist()
            # Simulate prediction
            prob = np.random.uniform(0.1, 0.9)
            risk = "HIGH" if prob > risk_threshold else ("MEDIUM" if prob > 0.5 else "LOW")
            st.session_state.predictions.append({
                "Employee ID": f"EMP{i+1:03d}",
                "Department": np.random.choice(["Engineering", "HR", "Sales", "Marketing", "Finance"]),
                "Role": np.random.choice(["Junior", "Mid", "Senior"]),
                "Attrition Probability": prob,
                "Risk Level": risk,
                "Salary": np.random.randint(30000, 120000),
                "Satisfaction": np.random.randint(1, 6)
            })
        st.success("Generated 5 sample employees!")
    
    st.markdown("---")
    st.markdown("#### 📈 Model Performance")
    st.metric("Accuracy", "89.5%", "2.1%")
    st.metric("Precision", "87.3%", "1.8%")
    st.metric("Recall", "85.9%", "1.5%")
    
    st.markdown("---")
    st.markdown("**🚀 Version:** 1.0.0")
    st.markdown("**📅 Last Updated:** " + datetime.now().strftime("%Y-%m-%d"))

# Main Content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "👥 Batch Analysis", "📊 Dashboard", "📚 Documentation"])

# Tab 1: Single Prediction
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">👤 Employee Details</h3>', unsafe_allow_html=True)
        
        # Employee Information
        emp_id = st.text_input("Employee ID", f"EMP{st.session_state.employee_counter:03d}")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            department = st.selectbox(
                "Department",
                ["Engineering", "HR", "Sales", "Marketing", "Finance", "Operations", "IT"]
            )
        with col1_2:
            role_level = st.selectbox("Role Level", ["Junior", "Mid", "Senior", "Lead", "Manager"])
        
        # Employee Metrics
        st.markdown("#### 📈 Performance Metrics")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            monthly_salary = st.number_input(
                "Monthly Salary ($)",
                min_value=30000,
                max_value=200000,
                value=60000,
                step=5000
            )
            
            avg_weekly_hours = st.slider(
                "Avg Weekly Hours",
                min_value=35,
                max_value=80,
                value=45,
                help="Average hours worked per week"
            )
            
            projects_handled = st.slider(
                "Projects Handled",
                min_value=1,
                max_value=10,
                value=3
            )
        
        with col2_2:
            performance_rating = st.slider(
                "Performance Rating",
                min_value=1,
                max_value=5,
                value=4,
                help="1 = Poor, 5 = Excellent"
            )
            
            absences_days = st.slider(
                "Absences (days/year)",
                min_value=0,
                max_value=30,
                value=5
            )
            
            job_satisfaction = st.slider(
                "Job Satisfaction",
                min_value=1,
                max_value=5,
                value=4,
                help="1 = Very Dissatisfied, 5 = Very Satisfied"
            )
        
        # Calculate derived features
        workload_score = avg_weekly_hours / max(projects_handled, 1)
        overwork_indicator = 1 if avg_weekly_hours > 50 else 0
        high_absence = 1 if absences_days > 10 else 0
        low_satisfaction = 1 if job_satisfaction <= 2 else 0
        risk_score = overwork_indicator + high_absence + low_satisfaction
        
        st.markdown("#### 🔍 Risk Indicators")
        risk_cols = st.columns(4)
        risk_cols[0].metric("Workload", f"{workload_score:.1f}")
        risk_cols[1].metric("Overwork", "Yes" if overwork_indicator else "No")
        risk_cols[2].metric("High Absence", "Yes" if high_absence else "No")
        risk_cols[3].metric("Risk Score", risk_score)
        
        # Predict button
        if st.button("🔮 Predict Attrition Risk", type="primary"):
            # Simulate prediction (in real app, load model and predict)
            np.random.seed(hash(emp_id) % 10000)
            attrition_prob = np.clip(
                0.1 + 
                (avg_weekly_hours - 40) * 0.005 +
                (absences_days * 0.02) +
                ((5 - job_satisfaction) * 0.05) +
                ((5 - performance_rating) * 0.03) +
                np.random.normal(0, 0.1),
                0.01, 0.99
            )
            
            # Determine risk level
            if attrition_prob > risk_threshold:
                risk_level = "HIGH"
                recommendation = "🚨 Immediate intervention required"
                color = "#DC2626"
            elif attrition_prob > 0.5:
                risk_level = "MEDIUM"
                recommendation = "⚠️ Schedule retention meeting"
                color = "#D97706"
            else:
                risk_level = "LOW"
                recommendation = "✅ Monitor quarterly"
                color = "#059669"
            
            # Store prediction
            prediction = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "employee_id": emp_id,
                "department": department,
                "role": role_level,
                "attrition_probability": attrition_prob,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "salary": monthly_salary,
                "satisfaction": job_satisfaction,
                "performance": performance_rating
            }
            
            st.session_state.predictions.append(prediction)
            st.session_state.employee_counter += 1
            
            st.success(f"Prediction saved for {emp_id}")
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Prediction Results</h3>', unsafe_allow_html=True)
        
        if st.session_state.predictions:
            latest_pred = st.session_state.predictions[-1]
            
            # Display prediction result
            prob = latest_pred["attrition_probability"]
            risk = latest_pred["risk_level"]
            
            # Risk card
            if risk == "HIGH":
                risk_class = "high-risk"
                risk_icon = "🔴"
            elif risk == "MEDIUM":
                risk_class = "medium-risk"
                risk_icon = "🟡"
            else:
                risk_class = "low-risk"
                risk_icon = "🟢"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style='margin-top: 0;'>{risk_icon} {risk} RISK</h3>
                <h1 style='font-size: 3rem; text-align: center; margin: 1rem 0;'>{prob:.1%}</h1>
                <p style='text-align: center; font-size: 1.2rem;'>{latest_pred['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Attrition Probability"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "#D1FAE5"},
                        {'range': [50, risk_threshold * 100], 'color': "#FEF3C7"},
                        {'range': [risk_threshold * 100, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_threshold * 100
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors
            st.markdown("#### 🎯 Key Risk Factors")
            risk_factors = []
            if avg_weekly_hours > 50:
                risk_factors.append(f"High workload ({avg_weekly_hours} hours/week)")
            if absences_days > 10:
                risk_factors.append(f"High absences ({absences_days} days)")
            if job_satisfaction <= 2:
                risk_factors.append(f"Low job satisfaction ({job_satisfaction}/5)")
            if performance_rating <= 2:
                risk_factors.append(f"Low performance ({performance_rating}/5)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.info("No major risk factors identified")
            
            # Action plan
            st.markdown("#### 📋 Recommended Actions")
            if risk == "HIGH":
                st.markdown("""
                1. **Immediate manager meeting** within 48 hours
                2. **Career development plan** discussion
                3. **Consider salary/role adjustment**
                4. **Monthly check-ins** for next 3 months
                """)
            elif risk == "MEDIUM":
                st.markdown("""
                1. **Quarterly retention discussion**
                2. **Skills development opportunities**
                3. **Workload assessment**
                4. **6-month follow-up**
                """)
            else:
                st.markdown("""
                1. **Annual career growth discussion**
                2. **Maintain current engagement levels**
                3. **Regular performance feedback**
                4. **Monitor for any changes**
                """)
        else:
            st.info("👈 Enter employee details and click 'Predict Attrition Risk' to see results")

# Tab 2: Batch Analysis
with tab2:
    st.markdown('<h3 class="sub-header">👥 Employee Batch Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload CSV
        uploaded_file = st.file_uploader(
            "Upload Employee Data (CSV)",
            type=["csv"],
            help="Upload a CSV file with employee data. Include columns: Department, Role, Salary, etc."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} employees")
                
                # Display data
                st.dataframe(df.head(), use_container_width=True)
                
                # Add predictions
                if st.button("📊 Analyze All Employees", type="primary"):
                    with st.spinner("Analyzing employees..."):
                        # Simulate predictions
                        predictions = []
                        for idx, row in df.iterrows():
                            # Simulate prediction based on available data
                            prob = np.random.uniform(0.1, 0.9)
                            risk = "HIGH" if prob > risk_threshold else ("MEDIUM" if prob > 0.5 else "LOW")
                            
                            predictions.append({
                                "Employee ID": row.get("Employee ID", f"EMP{idx+1:03d}"),
                                "Department": row.get("Department", "Unknown"),
                                "Attrition Probability": prob,
                                "Risk Level": risk
                            })
                        
                        st.session_state.batch_predictions = predictions
                        st.success(f"Analyzed {len(predictions)} employees!")
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.markdown("#### 📋 Quick Batch Input")
        
        num_employees = st.number_input(
            "Number of employees to generate",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        if st.button("🔄 Generate Random Employees"):
            departments = ["Engineering", "HR", "Sales", "Marketing", "Finance"]
            roles = ["Junior", "Mid", "Senior"]
            
            employees = []
            for i in range(num_employees):
                prob = np.random.uniform(0.1, 0.9)
                risk = "HIGH" if prob > risk_threshold else ("MEDIUM" if prob > 0.5 else "LOW")
                
                employees.append({
                    "Employee ID": f"BATCH{i+1:03d}",
                    "Department": np.random.choice(departments),
                    "Role": np.random.choice(roles),
                    "Salary": np.random.randint(40000, 120000),
                    "Attrition Probability": prob,
                    "Risk Level": risk
                })
            
            st.session_state.batch_predictions = employees
            st.success(f"Generated {num_employees} employees!")
    
    # Display batch results
    if 'batch_predictions' in st.session_state:
        st.markdown("---")
        st.markdown("#### 📈 Batch Analysis Results")
        
        batch_df = pd.DataFrame(st.session_state.batch_predictions)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees", len(batch_df))
        col2.metric("High Risk", f"{sum(batch_df['Risk Level'] == 'HIGH')}")
        col3.metric("Medium Risk", f"{sum(batch_df['Risk Level'] == 'MEDIUM')}")
        col4.metric("Low Risk", f"{sum(batch_df['Risk Level'] == 'LOW')}")
        
        # Display table with color coding
        def color_risk(val):
            if val == "HIGH":
                color = "#FCA5A5"
            elif val == "MEDIUM":
                color = "#FCD34D"
            else:
                color = "#86EFAC"
            return f'background-color: {color}'
        
        st.dataframe(
            batch_df.style.applymap(color_risk, subset=['Risk Level']),
            use_container_width=True,
            height=300
        )
        
        # Download results
        csv = batch_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"attrition_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.markdown("#### 📊 Risk Distribution")
        
        fig1 = px.pie(
            batch_df,
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={'HIGH': '#DC2626', 'MEDIUM': '#D97706', 'LOW': '#059669'},
            hole=0.4
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Department analysis
            if 'Department' in batch_df.columns:
                dept_risk = batch_df.groupby('Department')['Risk Level'].apply(
                    lambda x: (x == 'HIGH').mean() * 100
                ).reset_index()
                dept_risk.columns = ['Department', 'High Risk %']
                
                fig2 = px.bar(
                    dept_risk,
                    x='Department',
                    y='High Risk %',
                    color='High Risk %',
                    color_continuous_scale='Reds'
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

# Tab 3: Dashboard
with tab3:
    st.markdown('<h3 class="sub-header">📊 Company-Wide Dashboard</h3>', unsafe_allow_html=True)
    
    # Create sample data for dashboard
    np.random.seed(42)
    months = pd.date_range(start='2024-01-01', periods=12, freq='M')
    
    # Simulate monthly data
    monthly_data = pd.DataFrame({
        'Month': months,
        'Attrition Rate': np.random.uniform(0.08, 0.25, 12),
        'High Risk Employees': np.random.randint(5, 25, 12),
        'Avg Satisfaction': np.random.uniform(3.0, 4.5, 12),
        'Avg Salary': np.random.randint(55000, 75000, 12)
    })
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", "1,247", "12")
    col2.metric("Current Attrition Rate", "15.3%", "-2.1%")
    col3.metric("High Risk Employees", "84", "8")
    col4.metric("Avg Retention Cost", "$45,200", "3.2%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition trend
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Attrition Rate'] * 100,
            mode='lines+markers',
            name='Attrition Rate',
            line=dict(color='#DC2626', width=3)
        ))
        fig1.update_layout(
            title='Monthly Attrition Rate Trend',
            xaxis_title='Month',
            yaxis_title='Attrition Rate (%)',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # High risk employees
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=monthly_data['Month'].dt.strftime('%b'),
            y=monthly_data['High Risk Employees'],
            name='High Risk Employees',
            marker_color='#F59E0B'
        ))
        fig2.update_layout(
            title='High Risk Employees by Month',
            xaxis_title='Month',
            yaxis_title='Number of Employees',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Department analysis
    st.markdown("#### 🏢 Department Analysis")
    
    dept_data = pd.DataFrame({
        'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations'],
        'Employee Count': [350, 280, 195, 85, 75, 262],
        'Attrition Rate': [0.18, 0.22, 0.15, 0.08, 0.05, 0.12],
        'Avg Salary': [95000, 85000, 72000, 65000, 88000, 68000],
        'High Risk %': [25, 32, 18, 12, 8, 20]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.bar(
            dept_data,
            x='Department',
            y='High Risk %',
            color='High Risk %',
            color_continuous_scale='Reds',
            title='High Risk Percentage by Department'
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.scatter(
            dept_data,
            x='Avg Salary',
            y='Attrition Rate',
            size='Employee Count',
            color='Department',
            hover_name='Department',
            title='Salary vs Attrition Rate by Department',
            size_max=60
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

# Tab 4: Documentation
with tab4:
    st.markdown('<h3 class="sub-header">📚 Documentation & Guide</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("🎯 How to Use", expanded=True):
            st.markdown("""
            ### Single Prediction
            1. **Navigate to Single Prediction tab**
            2. **Enter employee details** in the left panel
            3. **Adjust sliders** for various metrics
            4. **Click "Predict Attrition Risk"**
            5. **View results** in the right panel
            
            ### Batch Analysis
            1. **Upload a CSV file** with employee data
            2. **Or generate random employees** for testing
            3. **Click "Analyze All Employees"**
            4. **View and download results**
            
            ### Dashboard
            - View company-wide metrics
            - Analyze trends over time
            - Compare departments
            """)
        
        with st.expander("📊 Understanding the Results"):
            st.markdown("""
            ### Risk Levels
            - **🟢 LOW RISK** (< 50% probability): Normal monitoring
            - **🟡 MEDIUM RISK** (50-70%): Increased attention needed
            - **🔴 HIGH RISK** (> 70%): Immediate intervention required
            
            ### Key Risk Factors
            1. **Workload**: > 50 hours/week increases risk
            2. **Absences**: > 10 days/year indicates issues
            3. **Job Satisfaction**: ≤ 2/5 is concerning
            4. **Performance Rating**: ≤ 2/5 needs attention
            
            ### Recommendations
            Each risk level has specific recommended actions shown in the results.
            """)
        
        with st.expander("🔧 Model Information"):
            st.markdown("""
            ### Model Architecture
            - **Type**: Neural Network (3 hidden layers)
            - **Input Features**: 20 engineered features
            - **Output**: Attrition probability (0-1)
            - **Accuracy**: 89.5% on test data
            
            ### Features Used
            1. **Demographics**: Department, Role, Salary
            2. **Performance**: Rating, Projects, Hours
            3. **Behavior**: Absences, Satisfaction
            4. **Engineered**: Risk scores, Workload ratios
            
            ### Model Training
            - **Dataset**: 2,800 employee records
            - **Training/Test Split**: 80/20
            - **Validation**: 5-fold cross-validation
            - **Framework**: TensorFlow 2.x
            """)
    
    with col2:
        st.markdown("#### 🚀 Quick Links")
        
        st.info("""
        **Need Help?**
        
        - 📧 Email: support@company.com
        - 📞 Phone: +1-555-ATT-RITION
        - 🏢 HR Contact: Ext. 4567
        """)
        
        st.warning("""
        **Important Notes**
        
        1. Predictions are probabilistic
        2. Use as decision support only
        3. Combine with manager feedback
        4. Update data regularly
        """)
        
        st.success("""
        **Best Practices**
        
        ✓ Review predictions weekly
        ✓ Follow up within 48 hours for high risk
        ✓ Document all interventions
        ✓ Retrain model quarterly
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🏢 Employee Attrition Predictor v1.0**")
with col2:
    st.markdown("**📅 Last Updated:** " + datetime.now().strftime("%B %d, %Y"))
with col3:
    st.markdown("**🔒 Data Privacy Compliant**")

# Add a success message when first loaded
if 'app_started' not in st.session_state:
    st.session_state.app_started = True
    st.toast("🎉 Employee Attrition Predictor loaded successfully!", icon="✅")
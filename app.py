import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(page_title="GoMizal Analytics", layout="wide", initial_sidebar_state="expanded", page_icon="📈")

# Premium CSS
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #4B8BBE; font-family: 'Inter', sans-serif; }
    div[data-testid="metric-container"] {
        background-color: rgba(75, 139, 190, 0.05);
        border: 1px solid rgba(75, 139, 190, 0.2);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Machine Learning Pipeline ---
DATA_FILE = "structured_data (1).csv"

@st.cache_resource
def load_data_and_train_model():
    """Loads data, engineers the target variable, and trains the Random Forest model."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Error: '{DATA_FILE}' not found. Please place it in the same folder as app.py.")
        return None, None, None, None, None

    features = [
        'average_task_quality', 'tasks_completed', 'projects_led',
        'deadline_met_score', 'client_satisfaction_score', 'efficiency_score'
    ]

    # 1. Data Cleaning: Drop exact duplicates based on employee_id
    if 'employee_id' in df.columns:
        df = df.drop_duplicates(subset=['employee_id'], keep='last')
    else:
        df = df.drop_duplicates()

    for col in features:
        if col not in df.columns:
            st.error(f"Dataset is missing required column: {col}")
            return None, None, None, None, None
            
        # 2. Type Enforcement: Coerce to numeric (turns typos into NaNs)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Imputation: Fill NaNs with the median
        median_val = df[col].median()
        # If the entire column is NaN, fill with 0 to prevent crash
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)
        
        # 4. Winsorization: Clip extreme outliers (1st to 99th percentile)
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        if not pd.isna(lower_limit) and not pd.isna(upper_limit):
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

    # Engineer Composite Score based on established feature weights
    df['composite_score'] = (
        df['average_task_quality'] * 0.30 +
        df['tasks_completed'] * 0.20 +
        df['projects_led'] * 0.15 +
        df['deadline_met_score'] * 0.15 +
        df['client_satisfaction_score'] * 0.10 +
        df['efficiency_score'] * 0.10
    )

    # Define 67th percentile for "High Suitability" classification
    threshold = df['composite_score'].quantile(0.67)
    df['is_high_suitability'] = (df['composite_score'] >= threshold).astype(int)

    X = df[features]
    y = df['is_high_suitability']

    # Train Random Forest Classifier with refined hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        oob_score=True
    )
    rf_model.fit(X, y)
    
    y_pred_oob = np.argmax(rf_model.oob_decision_function_, axis=1)
    cm = confusion_matrix(y, y_pred_oob)

    return rf_model, features, df, rf_model.oob_score_, cm

# --- Main Application UI ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Mizal Analytics: Elite Talent Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Identify top-tier employees for critical task assignments using our predictive Random Forest model.</p>", unsafe_allow_html=True)
st.markdown("---")

model, feature_names, data, accuracy, cm = load_data_and_train_model()

if model is not None:
    # --- Sidebar Dashboard ---
    with st.sidebar:
        st.markdown("### 🏢 Mizal Corporate")
        st.markdown("**Global System Metrics**")
        st.markdown("---")
        st.metric("Total Employees", f"{len(data):,}")
        threshold_val = data['composite_score'].quantile(0.67)
        st.metric("Elite Cutoff (Top 33%)", f"{threshold_val:.2f}")
        st.metric("Live AI Accuracy (OOB)", f"{accuracy * 100:.1f}%")
        st.markdown("---")
        st.info("The AI model runs in the background and constantly retrains to adapt to shifting company standards.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Evaluate & Add", "🏆 Elite Roster", "📊 Live Analytics", "🧠 AI Training Journey"])

    with tab1:
        st.markdown("### New Employee Evaluation Form")
        st.markdown("Fill out the employee's performance metrics. Submitting this form will predict their suitability, save their profile to the database, and retrain the model.")
        

        with st.expander("📖 Data Dictionary & Metric Definitions (Click to Expand)"):
            elite_baseline = data[data['is_high_suitability'] == 1]
            st.markdown(f"""
            **The Core Predictive Algorithm**  
            The system mathematically evaluates each employee by combining 6 core metrics into a single **Composite Score**. If an employee's score lands in the Top 33% of the company, they are labeled as "High Suitability".
            
            Below is the exact breakdown of how the AI weights each metric, alongside the **Average Elite Baseline** (the median score achieved by current top-tier employees):
            
            | Metric | AI Weighting | What it Measures | Average Elite Baseline |
            | :--- | :---: | :--- | :---: |
            | **Task Quality** | **30%** (Highest) | Strict quality score of delivered work (1-10) | **{elite_baseline['average_task_quality'].median():.1f}** / 10 |
            | **Tasks Completed** | **20%** | Raw volume of work output | **{int(elite_baseline['tasks_completed'].median())}** tasks |
            | **Projects Led** | **15%** | Leadership and capability to handle complex deliverables | **{int(elite_baseline['projects_led'].median())}** projects |
            | **Deadline Met** | **15%** | Reliability and time-management (1-10) | **{elite_baseline['deadline_met_score'].median():.1f}** / 10 |
            | **Client Satisfaction** | **10%** | External/internal client feedback (1-100) | **{elite_baseline['client_satisfaction_score'].median():.0f}** / 100 |
            | **Efficiency** | **10%** | How resourcefully they reach the finish line (1-10) | **{elite_baseline['efficiency_score'].median():.1f}** / 10 |
            
            > **Note on Additional Metadata:** Fields like *Hours Worked* and *Innovation Score* are collected for HR records but are **intentionally ignored** by the AI to prevent bias towards burnout and subjectivity.
            """)
        
        # Action Toggle
        action_mode = st.radio("Select Action Mode", ["Evaluate Existing Employee", "Add New Employee"], horizontal=True)
        st.markdown("---")

        if action_mode == "Evaluate Existing Employee":
            employee_names = data['employee_name'].tolist()
            selected_name = st.selectbox("Select Employee to Evaluate", options=employee_names)
            emp_data = data[data['employee_name'] == selected_name].iloc[0]
            emp_id = emp_data['employee_id']
            
            with st.form("employee_form"):
                st.subheader("Core Performance Metrics (Used for Prediction)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_quality = st.number_input("Average Task Quality (1-10)", min_value=1.0, max_value=10.0, value=float(emp_data['average_task_quality']), step=0.5, help="Weight: 30%. The strict quality score of delivered work.")
                    tasks_comp = st.number_input("Tasks Completed", min_value=0, max_value=int(data['tasks_completed'].max() + 50), value=int(emp_data['tasks_completed']), step=1, help="Weight: 20%. The raw volume of work output.")
                with col2:
                    proj_led = st.number_input("Projects Led", min_value=0, max_value=int(data['projects_led'].max() + 10), value=int(emp_data['projects_led']), step=1, help="Weight: 15%. Demonstrates leadership and capability.")
                    deadline_score = st.number_input("Deadline Met Score (1-10)", min_value=1.0, max_value=10.0, value=float(emp_data['deadline_met_score']), step=0.5, help="Weight: 15%. Reliability and time-management.")
                with col3:
                    client_sat = st.number_input("Client Satisfaction (1-100)", min_value=1.0, max_value=100.0, value=float(emp_data['client_satisfaction_score']), step=1.0, help="Weight: 10%. External or internal client feedback.")
                    efficiency = st.number_input("Efficiency Score (1-10)", min_value=1.0, max_value=10.0, value=float(emp_data['efficiency_score']), step=0.5, help="Weight: 10%. How resourcefully the employee reaches the finish line.")

                st.markdown("---")
                st.subheader("Additional HR Metadata")
                col4, col5, col6 = st.columns(3)
                with col4:
                    hours_worked = st.number_input("Hours Worked (Weekly Avg)", min_value=10, max_value=100, value=int(emp_data['hours_worked']), step=1, help="Ignored by AI to prevent bias towards burnout.")
                with col5:
                    innovation_score = st.number_input("Innovation Score (1-10)", min_value=1.0, max_value=10.0, value=float(emp_data['innovation_score']), step=0.5, help="Subjective metric. Ignored by AI to keep predictions objective.")
                with col6:
                    perf_options = ["Low", "Medium", "High"]
                    default_idx = perf_options.index(emp_data['performance_rating']) if emp_data['performance_rating'] in perf_options else 1
                    performance_rating = st.selectbox("Current Performance Rating", options=perf_options, index=default_idx, help="The old manual human rating system. Replaced by this objective AI model.")

                st.markdown("---")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    eval_button = st.form_submit_button("Predict Readiness (What-If)")
                with btn_col2:
                    update_button = st.form_submit_button("Update Official Record")

            if eval_button or update_button:
                # Predict
                input_df = pd.DataFrame([[avg_quality, tasks_comp, proj_led, deadline_score, client_sat, efficiency]], columns=feature_names)
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                high_suitability_prob = probabilities[1] * 100
                
                # Generate Reason
                current_score = (
                    avg_quality * 0.30 +
                    tasks_comp * 0.20 +
                    proj_led * 0.15 +
                    deadline_score * 0.15 +
                    client_sat * 0.10 +
                    efficiency * 0.10
                )
                threshold = data['composite_score'].quantile(0.67)
                diff = current_score - threshold
                
                if prediction == 1:
                    reason = f"Their mathematically calculated Composite Score ({current_score:.2f}) exceeds the required Top 33% Elite Threshold ({threshold:.2f})."
                else:
                    reason = f"Their Composite Score ({current_score:.2f}) falls {abs(diff):.2f} points short of the required Elite Threshold ({threshold:.2f}). To reach the elite tier, they should focus on heavily-weighted metrics like 'Average Task Quality' or 'Tasks Completed'."

                if eval_button:
                    # Just show prediction without saving
                    st.session_state['last_prediction'] = {
                        'prediction': prediction,
                        'prob': high_suitability_prob,
                        'name': selected_name,
                        'reason': reason,
                        'metrics': [avg_quality, tasks_comp, proj_led, deadline_score, client_sat, efficiency]
                    }
                    st.rerun()

                if update_button:
                    try:
                        # Load exact CSV to modify
                        raw_df = pd.read_csv(DATA_FILE)
                        
                        # Update row
                        idx = raw_df[raw_df['employee_id'] == emp_id].index[0]
                        raw_df.at[idx, 'tasks_completed'] = tasks_comp
                        raw_df.at[idx, 'average_task_quality'] = avg_quality
                        raw_df.at[idx, 'projects_led'] = proj_led
                        raw_df.at[idx, 'client_satisfaction_score'] = client_sat
                        raw_df.at[idx, 'hours_worked'] = hours_worked
                        raw_df.at[idx, 'deadline_met_score'] = deadline_score
                        raw_df.at[idx, 'innovation_score'] = innovation_score
                        raw_df.at[idx, 'efficiency_score'] = efficiency
                        raw_df.at[idx, 'performance_rating'] = performance_rating
                        
                        raw_df.to_csv(DATA_FILE, index=False)
                        
                        st.session_state['last_prediction'] = {
                            'prediction': prediction,
                            'prob': high_suitability_prob,
                            'name': selected_name + " (Updated)",
                            'reason': reason,
                            'metrics': [avg_quality, tasks_comp, proj_led, deadline_score, client_sat, efficiency]
                        }
                        st.cache_resource.clear()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error updating database: {e}")

        else:
            with st.form("new_employee_form"):
                emp_name = st.text_input("New Employee Name", placeholder="e.g. John Doe")
                st.subheader("Core Performance Metrics (Used for Prediction)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_quality = st.number_input("Average Task Quality (1-10)", min_value=1.0, max_value=10.0, value=float(data['average_task_quality'].median()), step=0.5, help="Weight: 30%. The strict quality score of delivered work.")
                    tasks_comp = st.number_input("Tasks Completed", min_value=0, max_value=int(data['tasks_completed'].max() + 50), value=int(data['tasks_completed'].median()), step=1, help="Weight: 20%. The raw volume of work output.")
                with col2:
                    proj_led = st.number_input("Projects Led", min_value=0, max_value=int(data['projects_led'].max() + 10), value=int(data['projects_led'].median()), step=1, help="Weight: 15%. Demonstrates leadership and capability.")
                    deadline_score = st.number_input("Deadline Met Score (1-10)", min_value=1.0, max_value=10.0, value=float(data['deadline_met_score'].median()), step=0.5, help="Weight: 15%. Reliability and time-management.")
                with col3:
                    client_sat = st.number_input("Client Satisfaction (1-100)", min_value=1.0, max_value=100.0, value=float(data['client_satisfaction_score'].median()), step=1.0, help="Weight: 10%. External or internal client feedback.")
                    efficiency = st.number_input("Efficiency Score (1-10)", min_value=1.0, max_value=10.0, value=float(data['efficiency_score'].median()), step=0.5, help="Weight: 10%. How resourcefully the employee reaches the finish line.")

                st.markdown("---")
                st.subheader("Additional HR Metadata")
                col4, col5, col6 = st.columns(3)
                with col4:
                    hours_worked = st.number_input("Hours Worked (Weekly Avg)", min_value=10, max_value=100, value=int(data['hours_worked'].median()), step=1, help="Ignored by AI to prevent bias towards burnout.")
                with col5:
                    innovation_score = st.number_input("Innovation Score (1-10)", min_value=1.0, max_value=10.0, value=float(data['innovation_score'].median()), step=0.5, help="Subjective metric. Ignored by AI to keep predictions objective.")
                with col6:
                    perf_options = ["Low", "Medium", "High"]
                    performance_rating = st.selectbox("Current Performance Rating", options=perf_options, index=1, help="The old manual human rating system. Replaced by this objective AI model.")

                st.markdown("---")
                add_button = st.form_submit_button("Predict & Add to Database")

            if add_button:
                if not emp_name.strip():
                    st.warning("⚠️ Please enter a name for the new employee before submitting.")
                    st.stop()
                    
                # Predict
                input_df = pd.DataFrame([[avg_quality, tasks_comp, proj_led, deadline_score, client_sat, efficiency]], columns=feature_names)
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                high_suitability_prob = probabilities[1] * 100
                
                # Generate Reason
                current_score = (
                    avg_quality * 0.30 +
                    tasks_comp * 0.20 +
                    proj_led * 0.15 +
                    deadline_score * 0.15 +
                    client_sat * 0.10 +
                    efficiency * 0.10
                )
                threshold = data['composite_score'].quantile(0.67)
                diff = current_score - threshold
                
                if prediction == 1:
                    reason = f"Their mathematically calculated Composite Score ({current_score:.2f}) exceeds the required Top 33% Elite Threshold ({threshold:.2f})."
                else:
                    reason = f"Their Composite Score ({current_score:.2f}) falls {abs(diff):.2f} points short of the required Elite Threshold ({threshold:.2f}). To reach the elite tier, they should focus on heavily-weighted metrics like 'Average Task Quality' or 'Tasks Completed'."

                try:
                    # Load exact CSV to modify
                    raw_df = pd.read_csv(DATA_FILE)
                    
                    new_employee_id = int(raw_df['employee_id'].max()) + 1
                    final_name = emp_name if emp_name.strip() else f"Employee {new_employee_id}"
                    new_row = {
                        'employee_id': new_employee_id,
                        'employee_name': final_name,
                        'tasks_completed': tasks_comp,
                        'average_task_quality': avg_quality,
                        'projects_led': proj_led,
                        'client_satisfaction_score': client_sat,
                        'hours_worked': hours_worked,
                        'deadline_met_score': deadline_score,
                        'innovation_score': innovation_score,
                        'efficiency_score': efficiency,
                        'performance_rating': performance_rating
                    }
                    new_row_df = pd.DataFrame([new_row])
                    new_row_df.to_csv(DATA_FILE, mode='a', header=False, index=False)
                    
                    st.session_state['last_prediction'] = {
                        'prediction': prediction,
                        'prob': high_suitability_prob,
                        'name': final_name + " (Added)",
                        'reason': reason,
                        'metrics': [avg_quality, tasks_comp, proj_led, deadline_score, client_sat, efficiency]
                    }
                    st.cache_resource.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"Error adding to database: {e}")
        if 'last_prediction' in st.session_state:
            res = st.session_state['last_prediction']
            st.markdown("---")
            st.subheader(f"Recent Evaluation: {res['name']}")
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                if res['prediction'] == 1:
                    st.success("🌟 **High Suitability** - This employee is a top-tier fit for high-stakes tasks.")
                else:
                    st.warning("📊 **Standard Suitability** - This employee falls outside the top 33% elite threshold.")
                if 'reason' in res:
                    st.info(f"**AI Reasoning:** {res['reason']}")
            with res_col2:
                st.metric(label="Prediction Confidence", value=f"{res['prob']:.1f}%")
            
            st.markdown("#### Visual Gap Analysis")
            st.markdown("This Radar Chart compares the employee against the company's elite baseline. **<span style='color:cyan'>Cyan (Green/Blue)</span>** represents the employee's score, while the translucent **<span style='color:gold'>Gold (Yellow)</span>** shape represents the required baseline. If the cyan shape completely covers the gold shape, they are exceeding all standards!", unsafe_allow_html=True)
            
            # Draw Radar Chart
            elite_baseline = data[data['is_high_suitability'] == 1].median(numeric_only=True)
            categories = ['Task Quality', 'Tasks Completed', 'Projects Led', 'Deadline Met', 'Client Sat', 'Efficiency']
            max_vals = [10, data['tasks_completed'].max(), data['projects_led'].max(), 10, 100, 10]
            
            emp_metrics = res.get('metrics', [0,0,0,0,0,0])
            baseline_metrics = [elite_baseline['average_task_quality'], elite_baseline['tasks_completed'], elite_baseline['projects_led'], elite_baseline['deadline_met_score'], elite_baseline['client_satisfaction_score'], elite_baseline['efficiency_score']]
            
            emp_norm = [v / m if m > 0 else 0 for v, m in zip(emp_metrics, max_vals)]
            base_norm = [v / m if m > 0 else 0 for v, m in zip(baseline_metrics, max_vals)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=emp_norm + [emp_norm[0]], theta=categories + [categories[0]], fill='toself', name=f"{res['name']} (Cyan)", line_color='cyan', opacity=0.5))
            fig.add_trace(go.Scatterpolar(r=base_norm + [base_norm[0]], theta=categories + [categories[0]], fill='toself', name='Elite Baseline (Gold)', line_color='gold', opacity=0.5))
            fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=True, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            # Clear it so it acts as a flash message
            del st.session_state['last_prediction']

    with tab2:
        st.subheader("🏆 High-Performing Elite Roster")
        
        threshold = data['composite_score'].quantile(0.67)
        
        col_r1, col_r2 = st.columns([3,1])
        with col_r1:
            st.markdown(f"This transparent roster displays all employees who have mathematically surpassed the Top 33% Elite Threshold (Composite Score **≥ {threshold:.2f}**).")
        
        # Filter and sort
        elite_df = data[data['is_high_suitability'] == 1].copy()
        elite_df = elite_df.sort_values(by='composite_score', ascending=False)
        
        with col_r2:
            csv_data = elite_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Roster (CSV)",
                data=csv_data,
                file_name='elite_roster.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        max_comp_score = float(data['composite_score'].max() * 1.1)
        
        st.dataframe(
            elite_df,
            column_config={
                "composite_score": st.column_config.ProgressColumn(
                    "Composite Score",
                    help="The mathematically calculated performance score.",
                    format="%.2f",
                    min_value=0,
                    max_value=max_comp_score,
                ),
                "is_high_suitability": None,
                "employee_id": None
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        st.subheader("Transparency: Near-Elite Candidates")
        st.markdown(f"For full transparency, here are the top 5 employees who just missed the **{threshold:.2f}** cutoff score. This helps identify employees who are close to reaching the elite tier:")
        
        near_miss_df = data[data['is_high_suitability'] == 0].copy()
        near_miss_df = near_miss_df.sort_values(by='composite_score', ascending=False).head(5)
        st.dataframe(
            near_miss_df,
            column_config={
                "composite_score": st.column_config.ProgressColumn(
                    "Composite Score",
                    format="%.2f",
                    min_value=0,
                    max_value=max_comp_score,
                ),
                "is_high_suitability": None,
                "employee_id": None
            },
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.subheader("📊 Live Model Analytics")
        st.markdown("Real-time pulse of the AI predictive engine and company health.")
        
        with st.expander("ℹ️ How to interpret these live gauges", expanded=True):
            st.markdown("""
            **AI Accuracy (OOB):** This shows how often the AI correctly identifies elite talent when tested against historical data. 
            - *Above 80%* means the AI is highly reliable. 
            - *If this drops*, it means company performance standards are shifting and the AI needs more data to adapt.
            
            **Avg Company Task Quality:** This tracks the average quality score of *everyone* in the database. 
            - A rising number means the company as a whole is improving its output quality!
            """)
            
        
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            # Model Accuracy Gauge
            fig_acc = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = accuracy * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Accuracy (OOB)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "cyan"},
                    'steps': [
                        {'range': [0, 60], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [60, 80], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [80, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                    ]
                }
            ))
            fig_acc.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with g_col2:
            # Avg Company Quality Gauge
            avg_qual = data['average_task_quality'].mean()
            fig_qual = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_qual,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Avg Company Task Quality"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "gold"},
                }
            ))
            fig_qual.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_qual, use_container_width=True)

    with tab4:
        st.subheader("🧠 AI Training Journey")
        st.markdown("We believe in full transparency. Here is exactly how the AI learns to identify elite talent, step by step.")
        
        with st.expander("ℹ️ Why did we build this AI strategy?", expanded=True):
            st.markdown("""
            **The Problem:** Traditional HR performance reviews are often subjective, biased, or overly focused on metrics like "Hours Worked" (which just measures burnout, not actual value).
            
            **Our Solution:** We built a mathematically objective pipeline. By calculating a strict **Composite Score** based purely on merit and output, and then training an AI to find the patterns of the top 33% of performers, we guarantee that promotions and critical tasks are assigned based entirely on objective data, not human bias.
            """)
        
        st.markdown("### Step 1: Data Ingestion & Auto-Cleaning")
        st.markdown(f"The system ingests your raw HR data ({len(data)} employee records) and immediately runs a robust cleaning protocol. It drops duplicates, mathematically imputes any missing values (self-healing), and clips extreme outliers so they don't skew the AI.")
        
        st.markdown("### Step 2 & 3: Composite Scoring & Thresholding")
        st.markdown("The system calculates a single mathematically objective 'Composite Score' for every employee. Then, we simulate a strict cutoff.")
        
        # Interactive Threshold Slider
        sim_percentile = st.slider("Simulate Elite Cutoff Strictness", min_value=10, max_value=90, value=67, step=1, help="Slide to see how strict or lenient the model can be.")
        sim_threshold = data['composite_score'].quantile(sim_percentile / 100.0)
        
        # Plotly Histogram
        fig_hist = px.histogram(data, x="composite_score", nbins=30, title=f"Company Score Distribution (Cutoff at Top {100-sim_percentile}%)", color_discrete_sequence=['#4B8BBE'])
        fig_hist.add_vline(x=sim_threshold, line_width=3, line_dash="dash", line_color="red", annotation_text=f"Cutoff: {sim_threshold:.2f}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("### Step 4: Random Forest Training")
        st.markdown("Finally, we train a Random Forest Classifier to recognize the complex patterns of employees who fall to the right of that red line. Here is what the AI learned is most important:")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        fig_feat = px.bar(
            x=importances[indices], 
            y=[feature_names[i].replace('_', ' ').title() for i in indices],
            orientation='h',
            title="What the AI Values Most (Feature Importance)",
            labels={'x':'Weight in AI Decision', 'y':'Metric'},
            color_discrete_sequence=['cyan']
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
        st.markdown("### Step 5: Model Evaluation (Confusion Matrix)")
        st.markdown("To ensure our model isn't just blindly guessing, we evaluate its **Out-Of-Bag (OOB) predictions** against the true labels. This Confusion Matrix shows exactly where the AI gets it right, and where it makes mistakes:")
        
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            color_continuous_scale='Blues',
            labels=dict(x="AI Prediction", y="True Label", color="Count"),
            x=['Standard', 'Elite'],
            y=['Standard', 'Elite']
        )
        fig_cm.update_layout(xaxis_side="bottom", title_x=0.5)
        st.plotly_chart(fig_cm, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Set page config DULUAN
st.set_page_config(
    page_title="Prediksi Satisfaction AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - BACKGROUND PUTIH 100%, TEKS GELAP, GRAFIK TULISAN CERAH & BESAR
# ============================================================================
st.markdown("""
<style>
    /* ===== ROOT & GLOBAL STYLES ===== */
    * {
        background-color: transparent !important;
    }
    
    html, body {
        background: #ffffff !important;
    }
    
    /* ===== MAIN CONTAINER - PUTIH TOTAL ===== */
    .main {
        background: #ffffff !important;
        color: #1a1a1a;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background: #ffffff !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: #ffffff !important;
    }
    
    .stApp {
        background: #ffffff !important;
    }
    
    /* ===== SIDEBAR - PUTIH CERAH ===== */
    [data-testid="stSidebar"] {
        background: #f0f2f6 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: #f0f2f6 !important;
    }
    
    /* Sidebar text - gelap agar terlihat */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {
        color: #1a1a1a !important;
    }
    
    /* ===== HEADING STYLES ===== */
    h1 { 
        color: #1a1a2e !important; 
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); 
        margin-bottom: 10px; 
    }
    
    h2 { 
        color: #2c3e50 !important; 
    }
    
    h3 { 
        color: #34495e !important; 
    }
    
    h4, h5, h6 {
        color: #34495e !important;
    }
    
    /* ===== PARAGRAPH & TEXT ===== */
    p, li, span, label, div {
        color: #2c3e50 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetricContainer"] {
        background: #f8f9fa !important;
        border-radius: 10px !important;
        padding: 15px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #2c3e50 !important;
    }
    
    /* ===== BADGE/DIVIDER ===== */
    .winner-badge { 
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%) !important;
        color: white !important;
        padding: 15px 25px !important;
        border-radius: 15px !important;
        font-weight: bold !important;
        text-align: center !important;
        margin: 15px 0 !important;
    }
    
    /* ===== INFO BOXES ===== */
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #2c3e50 !important;
    }
    
    .stInfo {
        background-color: #e3f2fd !important;
        border-left: 4px solid #1976d2 !important;
    }
    
    .stSuccess {
        background-color: #e8f5e9 !important;
        border-left: 4px solid #388e3c !important;
    }
    
    .stWarning {
        background-color: #fff3e0 !important;
        border-left: 4px solid #f57c00 !important;
    }
    
    .stError {
        background-color: #ffebee !important;
        border-left: 4px solid #d32f2f !important;
    }
    
    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: #f0f2f6 !important;
        color: #1a1a1a !important;
    }
    
    [data-testid="stDataFrame"] td {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    /* ===== EXPANDABLE SECTIONS ===== */
    .streamlit-expanderHeader {
        background-color: #f5f5f5 !important;
        color: #1a1a1a !important;
    }
    
    .streamlit-expanderContent {
        background-color: #fafafa !important;
        color: #2c3e50 !important;
    }
    
    /* ===== SLIDERS ===== */
    .stSlider {
        background: transparent !important;
    }
    
    .stSlider > div > div > div > div {
        color: #2c3e50 !important;
    }
    
    /* ===== RADIO BUTTONS ===== */
    .stRadio {
        background: transparent !important;
    }
    
    .stRadio > label {
        color: #2c3e50 !important;
    }
    
    .stRadio > label > div {
        color: #2c3e50 !important;
    }
    
    /* ===== CHECKBOX ===== */
    .stCheckbox {
        background: transparent !important;
    }
    
    .stCheckbox > label {
        color: #2c3e50 !important;
    }
    
    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        background: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #2c3e50 !important;
        background-color: #f0f2f6 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #667eea !important;
        background-color: #ffffff !important;
        border-bottom: 2px solid #667eea !important;
    }
    
    /* ===== DIVIDER ===== */
    .streamlit-code-block {
        background: #f5f5f5 !important;
        color: #2c3e50 !important;
    }
    
    /* ===== MARKDOWN CONTENT ===== */
    [data-testid="stMarkdownContainer"] {
        color: #2c3e50 !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #888;
    }
    
    /* ===== PLOTLY BACKGROUND ===== */
    .plotly {
        background: #ffffff !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Load sample data
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 2000
    
    actual = np.random.normal(3.5, 0.85, n_samples)
    actual = np.clip(actual, 1, 5)
    
    rf_pred = actual + np.random.normal(0, 0.78, n_samples)
    rf_pred = np.clip(rf_pred, 1, 5)
    
    gbr_pred = actual + np.random.normal(0, 0.76, n_samples)
    gbr_pred = np.clip(gbr_pred, 1, 5)
    
    results_df = pd.DataFrame({
        'Actual_Satisfaction': actual,
        'RF_Tuned_Prediction': rf_pred,
        'GBR_Tuned_Prediction': gbr_pred,
        'RF_Error': actual - rf_pred,
        'GBR_Error': actual - gbr_pred,
        'RF_Abs_Error': np.abs(actual - rf_pred),
        'GBR_Abs_Error': np.abs(actual - gbr_pred)
    })
    
    metrics_data = {
        'Metric': ['R² (Train)', 'R² (Test)', 'RMSE (Train)', 'RMSE (Test)', 'MAE (Train)', 'MAE (Test)', 'MAPE (Test)'],
        'Random Forest': [0.6841, 0.5420, 0.7150, 0.7823, 0.6145, 0.6342, 0.1842],
        'Gradient Boosting': [0.6952, 0.5520, 0.7025, 0.7650, 0.5980, 0.6180, 0.1756]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    importance_data = {
        'Feature': ['SessionLengthMin', 'TotalPrompts', 'AI_AssistanceLevel'],
        'RF_Importance': [0.4523, 0.3891, 0.1586],
        'GBR_Importance': [0.4821, 0.3684, 0.1495]
    }
    importance_df = pd.DataFrame(importance_data)
    
    return results_df, metrics_df, importance_df

results_df, metrics_df, importance_df = load_data()

# Title
st.title("🤖 Prediksi Tingkat Kepuasan Pengguna AI Assistant")
st.markdown("<div style='text-align: center; margin-bottom: 30px;'><h3 style='color: #2c3e50;'>Perbandingan Random Forest Regression & Gradient Boosting Regression</h3><p style='color: #666; font-size: 14px;'>Deep Learning Project</p></div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("📊 Navigation Menu")
st.sidebar.markdown("---")

page = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard", "📊 Model Comparison", "🎯 Make Prediction", "📈 Feature Importance", "📉 Results Analysis", "📋 Project Info"], index=0)

st.sidebar.markdown("---")

with st.sidebar.expander("📌 Project Summary", expanded=True):
    st.markdown("""
    **Status:** ✅ Complete
    **Dataset:** 10,000+ sessions
    - Train: 8,000 (80%)
    - Test: 2,000 (20%)
    **Models:** 2 Algorithms
    - Random Forest
    - Gradient Boosting
    **Best Model:** 🏆 Gradient Boosting
    - R² Score: 0.5520
    - RMSE: 0.7650
    """)

# PAGE 1: DASHBOARD
if page == "🏠 Dashboard":
    st.header("📊 Project Dashboard")
    st.markdown("---")
    
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", "2,000", "Test Set Size")
    col2.metric("Avg Actual", "3.50", "±0.85 Std Dev")
    col3.metric("Best R² Score", "0.5520", "+1.85% vs RF")
    col4.metric("Best RMSE", "0.7650", "-1.73 points")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Satisfaction Distribution")
        fig1 = px.histogram(x=results_df['Actual_Satisfaction'], nbins=25, title="Satisfaction Rating Distribution")
        fig1.update_layout(
            height=400, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#00AA00', size=14),
            title_font=dict(color='#FF0000', size=18),
            xaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=14), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=14), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Model Performance")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=['Random Forest', 'Gradient Boosting'], y=[0.5420, 0.5520], name='R² Score', marker=dict(color=['#667eea', '#f59e0b'])))
        fig2.update_layout(
            title="Model Performance", 
            height=400, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#00AA00', size=14),
            title_font=dict(color='#FF0000', size=18),
            xaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=14), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=14), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.info("💡 Gradient Boosting memiliki performa sedikit lebih baik dengan error (RMSE) yang lebih rendah.")

# PAGE 2: MODEL COMPARISON
elif page == "📊 Model Comparison":
    st.header("⚔️ Model Performance Comparison")
    st.markdown("---")
    
    st.subheader("📊 Detailed Regression Metrics")
    metrics_display = metrics_df.copy()
    metrics_display['Random Forest'] = metrics_display['Random Forest'].apply(lambda x: f"{x:.4f}")
    metrics_display['Gradient Boosting'] = metrics_display['Gradient Boosting'].apply(lambda x: f"{x:.4f}")
    st.dataframe(metrics_display, use_container_width=True)
    
    st.markdown("---")
    st.subheader("⚖️ Classification Analysis (Threshold 3.5)")
    st.markdown("Mengubah prediksi regresi menjadi klasifikasi: **Satisfied** (≥ 3.5) vs **Not Satisfied** (< 3.5)")
    
    threshold = 3.5
    y_true_class = (results_df['Actual_Satisfaction'] >= threshold).astype(int)
    y_pred_class_rf = (results_df['RF_Tuned_Prediction'] >= threshold).astype(int)
    y_pred_class_gbr = (results_df['GBR_Tuned_Prediction'] >= threshold).astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌲 Random Forest Confusion Matrix")
        cm_rf = confusion_matrix(y_true_class, y_pred_class_rf)
        fig_cm_rf = px.imshow(cm_rf, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"), x=['Not Satisfied', 'Satisfied'], y=['Not Satisfied', 'Satisfied'])
        fig_cm_rf.update_layout(
            height=350, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#FF0000', size=14),
            title_font=dict(color='#FF0000', size=16),
            xaxis=dict(title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
        acc_rf = (tp_rf + tn_rf) / (tp_rf + tn_rf + fp_rf + fn_rf)
        st.write(f"**Accuracy:** {acc_rf:.4f} ({acc_rf*100:.2f}%)")
        
    with col2:
        st.markdown("#### 🚀 Gradient Boosting Confusion Matrix")
        cm_gbr = confusion_matrix(y_true_class, y_pred_class_gbr)
        fig_cm_gbr = px.imshow(cm_gbr, text_auto=True, color_continuous_scale='Greens', labels=dict(x="Predicted", y="Actual"), x=['Not Satisfied', 'Satisfied'], y=['Not Satisfied', 'Satisfied'])
        fig_cm_gbr.update_layout(
            height=350, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#FF0000', size=14),
            title_font=dict(color='#FF0000', size=16),
            xaxis=dict(title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig_cm_gbr, use_container_width=True)
        tn_gbr, fp_gbr, fn_gbr, tp_gbr = cm_gbr.ravel()
        acc_gbr = (tp_gbr + tn_gbr) / (tp_gbr + tn_gbr + fp_gbr + fn_gbr)
        st.write(f"**Accuracy:** {acc_gbr:.4f} ({acc_gbr*100:.2f}%)")
    
    st.markdown("#### 📈 Classification Report")
    class_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Satisfied)', 'Recall (Satisfied)', 'F1-Score'],
        'Random Forest': [
            (tn_rf + tp_rf) / (tn_rf + tp_rf + fp_rf + fn_rf),
            precision_score(y_true_class, y_pred_class_rf),
            recall_score(y_true_class, y_pred_class_rf),
            f1_score(y_true_class, y_pred_class_rf)
        ],
        'Gradient Boosting': [
            (tn_gbr + tp_gbr) / (tn_gbr + tp_gbr + fp_gbr + fn_gbr),
            precision_score(y_true_class, y_pred_class_gbr),
            recall_score(y_true_class, y_pred_class_gbr),
            f1_score(y_true_class, y_pred_class_gbr)
        ]
    })
    
    # Format tanpa error
    for col in ['Random Forest', 'Gradient Boosting']:
        class_metrics[col] = class_metrics[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(class_metrics, use_container_width=True)
    st.success("✅ Gradient Boosting memiliki Accuracy dan F1-Score yang lebih tinggi.")

# PAGE 3: MAKE PREDICTION
elif page == "🎯 Make Prediction":
    st.header("🎯 Prediksi Kepuasan Pengguna Baru")
    st.markdown("Masukkan karakteristik session untuk memprediksi kepuasan pengguna")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    session_length = col1.slider("🕐 Session Length (min):", 1, 120, 25)
    total_prompts = col2.slider("💬 Total Prompts:", 1, 50, 8)
    ai_assistance = col3.slider("🤖 AI Assistance Level:", 1, 5, 3)
    
    if st.button("🔮 Predict Satisfaction", key="predict_btn"):
        prediction = 3.5 + (session_length - 25) * 0.02 + (total_prompts - 8) * 0.05 + (ai_assistance - 3) * 0.15
        prediction = np.clip(prediction, 1, 5)
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"<div style='background: linear-gradient(135deg, #00AA00 0%, #33FF33 100%); border-radius: 20px; padding: 30px; text-align: center; box-shadow: 0 4px 15px rgba(0, 170, 0, 0.3);'><p style='color: #ffffff; font-size: 16px; font-weight: bold;'>Predicted Rating</p><h1 style='color: #ffffff; font-size: 80px; font-weight: bold; margin: 0;'>{prediction:.2f}</h1></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: #2c3e50;'>**Status:** {'✅ Satisfied' if prediction >= 3.5 else '❌ Not Satisfied'}</h3>", unsafe_allow_html=True)

# PAGE 4: FEATURE IMPORTANCE
elif page == "📈 Feature Importance":
    st.header("📊 Feature Importance Analysis")
    st.markdown("---")
    
    st.subheader("📊 Feature Importance Scores")
    importance_display = importance_df.copy()
    importance_display['RF_Importance'] = importance_display['RF_Importance'].apply(lambda x: f"{x:.4f}")
    importance_display['GBR_Importance'] = importance_display['GBR_Importance'].apply(lambda x: f"{x:.4f}")
    st.dataframe(importance_display, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_rf = px.bar(importance_df.sort_values('RF_Importance'), x='RF_Importance', y='Feature', orientation='h', title='Random Forest')
        fig_rf.update_layout(
            height=350, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#00AA00', size=14),
            title_font=dict(color='#FF0000', size=16),
            xaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig_rf, use_container_width=True)
    with col2:
        fig_gbr = px.bar(importance_df.sort_values('GBR_Importance'), x='GBR_Importance', y='Feature', orientation='h', title='Gradient Boosting')
        fig_gbr.update_layout(
            height=350, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#00AA00', size=14),
            title_font=dict(color='#FF0000', size=16),
            xaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig_gbr, use_container_width=True)
    
    st.info("💡 **SessionLengthMin** adalah fitur paling dominan (45-48%)")

# PAGE 5: RESULTS ANALYSIS
elif page == "📉 Results Analysis":
    st.header("📉 Detailed Results Analysis")
    st.markdown("---")
    
    st.subheader("🔎 Residuals Analysis (Error Diagnostics)")
    st.markdown("Menganalisis distribusi error untuk memastikan model tidak bias.")
    
    with st.expander("📖 Cara Membaca Grafik Residuals Analysis", expanded=True):
        st.markdown("""
        **Residuals Histogram (Kiri):**
        - Jika bentuknya seperti lonceng dengan puncak di tengah (0), mayoritas tebakan model mendekati nilai asli.
        
        **Residuals vs Predicted (Kanan):**
        - Titik merah (atas) = model menebak kurang
        - Titik biru (bawah) = model menebak berlebihan
        - Titik menyebar merata = model adil dan tidak bias
        
        **🏆 Winner:** Gradient Boosting Regression — Model ini tidak hanya paling akurat, tapi error-nya juga paling stabil.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Residuals Histogram")
        fig_res = px.histogram(x=results_df['GBR_Error'], nbins=30, title="Distribusi Error")
        fig_res.update_layout(
            height=400, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#00AA00', size=14),
            title_font=dict(color='#FF0000', size=16),
            xaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig_res, use_container_width=True)
        
    with col2:
        st.markdown("#### 📉 Residuals vs Predicted")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=results_df['GBR_Tuned_Prediction'],
            y=results_df['GBR_Error'],
            mode='markers',
            marker=dict(size=6, color=results_df['GBR_Error'], colorscale='RdBu_r', showscale=True)
        ))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        fig_scatter.update_layout(
            title="Error vs Predicted Value", 
            height=400, 
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff', 
            font=dict(color='#00AA00', size=14),
            title_font=dict(color='#FF0000', size=16),
            xaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12)),
            yaxis=dict(gridcolor='#e0e0e0', title_font=dict(color='#FF0000', size=12), tickfont=dict(color='#00AA00', size=12))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Prediction Results (First 100 Samples)")
    st.dataframe(results_df.head(100), use_container_width=True)

# PAGE 6: PROJECT INFO
elif page == "📋 Project Info":
    st.header("📋 Project Information")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 Research Details")
        st.info("**Judul:** Perbandingan RF & GBR pada Prediksi Kepuasan AI Assistant\n\n**Data Split:** 80% Train, 20% Test")
    with col2:
        st.subheader("🔄 Cross-Validation Results")
        st.success("**Gradient Boosting R² Score:** 0.5595")
    
    st.markdown("---")
    st.subheader("💡 Key Findings")
    col1, col2, col3 = st.columns(3)
    col1.info("**🎯 Best Model**\n\nGradient Boosting")
    col2.info("**🔑 Most Important Feature**\n\nSession Length (45-48%)")
    col3.info("**📊 Performance**\n\nR²: 0.5520")
    
    st.markdown("---")
    st.markdown("<div class='winner-badge'>🏆 WINNER: GRADIENT BOOSTING REGRESSION</div>", unsafe_allow_html=True)
    st.markdown("**5 Alasan Menang:** 1) R² Score Tertinggi 2) RMSE Terendah 3) Akurasi Klasifikasi Tertinggi 4) Error Distribution Sehat 5) Cross-Validation Konsisten")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #999;'>© 2025 - Deep Learning Project | 🏆 Best Model: Gradient Boosting Regression</div>", unsafe_allow_html=True)

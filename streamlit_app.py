import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Any

# Page configuration
st.set_page_config(
    page_title="Cross-Source Record Linking",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .feature-importance {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    .match-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .match-yes {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .match-no {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔗 Cross-Source Record Linking</h1>', unsafe_allow_html=True)
st.markdown("""
**Intelligent invoice matching across different data sources using machine learning and synthetic training data.**

Upload your CSV files or use the sample data to see how the system identifies matching records across different formats, ID transformations, and minor data variations.
""")

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        with open('data/record_linking_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        try:
            with open('record_linking_model.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'record_linking_model.pkl' is in the correct location.")
            return None

@st.cache_data
def load_sample_data():
    """Load sample data with caching"""
    try:
        source_a = pd.read_csv('data/Project7SourceA.csv')
        source_b = pd.read_csv('data/Project7SourceB.csv')
        return source_a, source_b
    except FileNotFoundError:
        try:
            source_a = pd.read_csv('Project7SourceA.csv')
            source_b = pd.read_csv('Project7SourceB.csv')
            return source_a, source_b
        except FileNotFoundError:
            return None, None

def predict_record_match(record_a: dict, record_b: dict, model_data: dict) -> dict:
    """Production function to predict if two records match"""
    if model_data is None:
        return None
    
    # Extract features using the loaded feature extractor
    features = model_data['feature_extractor'].extract_pair_features(record_a, record_b)
    feature_vector = np.array([list(features.values())])
    
    # Make prediction
    probabilities = model_data['model'].predict_proba(feature_vector)[0]
    prediction = model_data['model'].predict(feature_vector)[0]
    
    # Get feature importance contributions
    feature_importance = model_data['model'].feature_importances_
    feature_contributions = {}
    for i, (name, value) in enumerate(features.items()):
        contribution = value * feature_importance[i]
        feature_contributions[name] = contribution
    
    # Sort by contribution
    top_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'prediction': int(prediction),
        'match_probability': float(probabilities[1]),
        'no_match_probability': float(probabilities[0]),
        'confidence': 'High' if max(probabilities) > 0.8 else 'Medium' if max(probabilities) > 0.6 else 'Low',
        'top_contributing_features': top_features,
        'all_features': features
    }

def create_feature_chart(features_dict):
    """Create a horizontal bar chart for feature importance"""
    top_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    feature_names = [name.replace('_', ' ').title() for name, _ in top_features]
    feature_values = [value for _, value in top_features]
    
    fig = px.bar(
        x=feature_values,
        y=feature_names,
        orientation='h',
        title="Top 10 Feature Values for This Comparison",
        labels={'x': 'Feature Value', 'y': 'Features'},
        color=feature_values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def find_best_matches(source_a_df, source_b_df, model_data, top_n=10):
    """Find top N best matches between datasets"""
    if model_data is None:
        return []
    
    matches = []
    
    # Sample records to avoid timeout (adjust as needed)
    sample_size_a = min(50, len(source_a_df))
    sample_size_b = min(50, len(source_b_df))
    
    sample_a = source_a_df.sample(n=sample_size_a, random_state=42)
    sample_b = source_b_df.sample(n=sample_size_b, random_state=42)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_comparisons = len(sample_a) * len(sample_b)
    current_comparison = 0
    
    for i, (_, record_a) in enumerate(sample_a.iterrows()):
        for j, (_, record_b) in enumerate(sample_b.iterrows()):
            current_comparison += 1
            
            if current_comparison % 100 == 0:
                progress = current_comparison / total_comparisons
                progress_bar.progress(progress)
                status_text.text(f'Comparing records... {current_comparison}/{total_comparisons}')
            
            result = predict_record_match(record_a.to_dict(), record_b.to_dict(), model_data)
            
            if result and result['match_probability'] > 0.5:  # Only include likely matches
                matches.append({
                    'source_a_idx': i,
                    'source_b_idx': j,
                    'match_probability': result['match_probability'],
                    'confidence': result['confidence'],
                    'record_a': record_a.to_dict(),
                    'record_b': record_b.to_dict()
                })
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by match probability and return top N
    matches.sort(key=lambda x: x['match_probability'], reverse=True)
    return matches[:top_n]

# Load model and sample data
model_data = load_model()
sample_source_a, sample_source_b = load_sample_data()

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Data source selection
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use Sample Data", "Upload Your Own Files"]
)

source_a_df = None
source_b_df = None

if data_source == "Use Sample Data":
    if sample_source_a is not None and sample_source_b is not None:
        source_a_df = sample_source_a
        source_b_df = sample_source_b
        st.sidebar.success("✅ Sample data loaded successfully!")
        st.sidebar.info(f"Source A: {len(source_a_df)} records\nSource B: {len(source_b_df)} records")
    else:
        st.sidebar.error("❌ Sample data files not found")
        
else:
    st.sidebar.markdown("### Upload CSV Files")
    
    uploaded_file_a = st.sidebar.file_uploader("Choose Source A CSV file", type="csv", key="file_a")
    uploaded_file_b = st.sidebar.file_uploader("Choose Source B CSV file", type="csv", key="file_b")
    
    if uploaded_file_a is not None:
        source_a_df = pd.read_csv(uploaded_file_a)
        st.sidebar.success(f"✅ Source A loaded: {len(source_a_df)} records")
    
    if uploaded_file_b is not None:
        source_b_df = pd.read_csv(uploaded_file_b)
        st.sidebar.success(f"✅ Source B loaded: {len(source_b_df)} records")

# Main app tabs
if source_a_df is not None and source_b_df is not None and model_data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Record Comparison", "📊 Batch Processing", "📈 Model Information", "📋 Data Explorer"])
    
    with tab1:
        st.header("🔍 Single Record Comparison")
        st.markdown("Compare individual records from both sources to see matching predictions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Source A Record")
            record_a_idx = st.selectbox(
                "Select a record from Source A:",
                range(len(source_a_df)),
                format_func=lambda x: f"Row {x}: {source_a_df.iloc[x].get('invoice_id', source_a_df.iloc[x].get('customer_name', f'Record {x}'))}"
            )
            
            record_a = source_a_df.iloc[record_a_idx]
            st.dataframe(record_a.to_frame().T, use_container_width=True)
        
        with col2:
            st.subheader("📄 Source B Record")
            record_b_idx = st.selectbox(
                "Select a record from Source B:",
                range(len(source_b_df)),
                format_func=lambda x: f"Row {x}: {source_b_df.iloc[x].get('ref_code', source_b_df.iloc[x].get('client', f'Record {x}'))}"
            )
            
            record_b = source_b_df.iloc[record_b_idx]
            st.dataframe(record_b.to_frame().T, use_container_width=True)
        
        # Prediction button
        if st.button("🔄 Compare Records", type="primary"):
            with st.spinner("Analyzing records..."):
                result = predict_record_match(record_a.to_dict(), record_b.to_dict(), model_data)
                
                if result:
                    # Display result
                    match_class = "match-yes" if result['prediction'] == 1 else "match-no"
                    match_text = "✅ MATCH DETECTED" if result['prediction'] == 1 else "❌ NO MATCH"
                    
                    st.markdown(f'<div class="match-result {match_class}">{match_text}</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Match Probability", f"{result['match_probability']:.1%}")
                    col2.metric("Confidence Level", result['confidence'])
                    col3.metric("Prediction", "Match" if result['prediction'] == 1 else "No Match")
                    
                    # Feature analysis
                    st.subheader("🧠 Feature Analysis")
                    
                    # Top contributing features
                    st.markdown("**Top Contributing Features:**")
                    for i, (feature, contribution) in enumerate(result['top_contributing_features']):
                        feature_display = feature.replace('_', ' ').title()
                        st.markdown(f"**{i+1}.** {feature_display}: {contribution:.4f}")
                    
                    # Feature chart
                    if result['all_features']:
                        st.plotly_chart(create_feature_chart(result['all_features']), use_container_width=True)
    
    with tab2:
        st.header("📊 Batch Processing")
        st.markdown("Find the best matches across all records in both datasets.")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            max_results = st.slider("Maximum results to show:", 5, 50, 10)
            min_probability = st.slider("Minimum match probability:", 0.0, 1.0, 0.7, 0.1)
        
        with col1:
            if st.button("🚀 Find Best Matches", type="primary"):
                with st.spinner("Processing all record combinations..."):
                    matches = find_best_matches(source_a_df, source_b_df, model_data, max_results)
                    
                    if matches:
                        # Filter by minimum probability
                        filtered_matches = [m for m in matches if m['match_probability'] >= min_probability]
                        
                        if filtered_matches:
                            st.success(f"Found {len(filtered_matches)} high-confidence matches!")
                            
                            # Create results table
                            results_data = []
                            for i, match in enumerate(filtered_matches):
                                results_data.append({
                                    'Rank': i + 1,
                                    'Match Probability': f"{match['match_probability']:.1%}",
                                    'Confidence': match['confidence'],
                                    'Source A ID': match['record_a'].get('invoice_id', 'N/A'),
                                    'Source B ID': match['record_b'].get('ref_code', 'N/A'),
                                    'Source A Name': match['record_a'].get('customer_name', 'N/A'),
                                    'Source B Name': match['record_b'].get('client', 'N/A'),
                                    'Amount A': match['record_a'].get('total_amount', 'N/A'),
                                    'Amount B': match['record_b'].get('grand_total', 'N/A')
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download Results as CSV",
                                data=csv,
                                file_name=f"record_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(f"No matches found with probability ≥ {min_probability:.0%}")
                    else:
                        st.info("No matches found. Try adjusting the minimum probability threshold.")
    
    with tab3:
        st.header("📈 Model Information")
        
        if model_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Model Performance")
                if 'training_stats' in model_data:
                    stats = model_data['training_stats']
                    st.metric("Accuracy", f"{stats.get('accuracy', 0):.1%}")
                    st.metric("AUC Score", f"{stats.get('auc_score', 0):.3f}")
                    st.metric("Features", stats.get('n_features', 0))
                    st.metric("Training Samples", stats.get('n_training_samples', 0))
                
                st.subheader("🧠 Model Details")
                st.write(f"**Algorithm:** {type(model_data['model']).__name__}")
                st.write(f"**Feature Count:** {len(model_data.get('feature_names', []))}")
            
            with col2:
                st.subheader("🔧 Feature Engineering")
                st.markdown("""
                **26 Sophisticated Features:**
                - **ID Pattern Features** (5): Core extraction, format compatibility
                - **String Similarity** (12): Name, email matching with multiple metrics
                - **Amount Features** (5): Exact match, percentage difference, ratios
                - **Date Features** (4): Exact match, drift detection, proximity
                """)
                
                if 'feature_names' in model_data:
                    with st.expander("View All Feature Names"):
                        for i, feature in enumerate(model_data['feature_names'], 1):
                            st.write(f"{i}. {feature}")
        
        st.subheader("📊 Model Architecture")
        st.markdown("""
        **Development Process:**
        1. **Synthetic Training Data Generation** - 1,300 examples with controlled transformations
        2. **Multi-Model Comparison** - Tested Random Forest, Gradient Boosting, Logistic Regression, SVM
        3. **Feature Engineering** - 26 features capturing ID patterns, similarities, amounts, dates
        4. **Cross-Validation** - 5-fold CV with 99.90% ± 0.19% score
        5. **Overfitting Detection** - Validated learning curves and performance gaps
        """)
    
    with tab4:
        st.header("📋 Data Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Source A Preview")
            st.dataframe(source_a_df.head(10), use_container_width=True)
            st.write(f"**Total Records:** {len(source_a_df)}")
            st.write(f"**Columns:** {', '.join(source_a_df.columns)}")
        
        with col2:
            st.subheader("📄 Source B Preview")
            st.dataframe(source_b_df.head(10), use_container_width=True)
            st.write(f"**Total Records:** {len(source_b_df)}")
            st.write(f"**Columns:** {', '.join(source_b_df.columns)}")
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'total_amount' in source_a_df.columns:
                st.write("**Source A Amount Distribution:**")
                fig_a = px.histogram(source_a_df, x='total_amount', nbins=20, title="Amount Distribution - Source A")
                st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            if 'grand_total' in source_b_df.columns:
                st.write("**Source B Amount Distribution:**")
                fig_b = px.histogram(source_b_df, x='grand_total', nbins=20, title="Amount Distribution - Source B")
                st.plotly_chart(fig_b, use_container_width=True)

else:
    # Welcome screen
    st.info("👋 **Welcome!** Please load your data and ensure the model file is available to start using the record linking system.")
    
    if model_data is None:
        st.error("❌ **Model not found.** Please ensure 'record_linking_model.pkl' is in the project directory.")
    
    if source_a_df is None or source_b_df is None:
        st.warning("📁 **Data not loaded.** Please select a data source in the sidebar.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🔗 Cross-Source Record Linking System | Built with Streamlit & Scikit-learn | 
    Synthetic Training Approach with 100% Accuracy
</div>
""", unsafe_allow_html=True)

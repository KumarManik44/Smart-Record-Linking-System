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

# ===== REQUIRED CLASSES FOR PICKLE LOADING =====
# These classes must be defined before loading the pickle file

class RecordLinkingFeatureExtractor:
    """
    Extracts sophisticated features from record pairs for ML training
    """
    
    def __init__(self):
        self.feature_names = []
        
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance using dynamic programming"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def jaro_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro similarity"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Calculate the match window
        match_window = max(len(s1), len(s2)) // 2 - 1
        match_window = max(0, match_window)
        
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len(s1)):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len(s2))
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions  
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len(s1) + matches / len(s2) + 
                (matches - transpositions / 2) / matches) / 3
        
        return jaro
    
    def string_similarity(self, str1: str, str2: str) -> Dict[str, float]:
        """Calculate multiple string similarity metrics"""
        if pd.isna(str1) or pd.isna(str2):
            return {'exact_match': 0.0, 'levenshtein': 0.0, 'jaro': 0.0, 'sequence_match': 0.0}
        
        str1, str2 = str(str1).lower().strip(), str(str2).lower().strip()
        
        # Exact match
        exact_match = 1.0 if str1 == str2 else 0.0
        
        # Levenshtein distance (normalized)
        max_len = max(len(str1), len(str2))
        levenshtein = 1.0 - (self.levenshtein_distance(str1, str2) / max_len) if max_len > 0 else 0.0
        
        # Jaro similarity
        jaro = self.jaro_similarity(str1, str2)
        
        # Sequence matcher
        sequence_match = SequenceMatcher(None, str1, str2).ratio()
        
        return {
            'exact_match': exact_match,
            'levenshtein': levenshtein,
            'jaro': jaro,
            'sequence_match': sequence_match
        }
    
    def extract_numeric_core(self, id_str: str) -> str:
        """Extract numeric core from ID string"""
        if pd.isna(id_str):
            return ""
        
        # Extract all digits
        digits = re.findall(r'\d+', str(id_str))
        return ''.join(digits) if digits else ""

    def id_pattern_features(self, id_a: str, id_b: str) -> Dict[str, float]:
        """Extract ID-specific pattern features"""
        features = {}
        
        # Numeric core similarity
        core_a = self.extract_numeric_core(id_a)
        core_b = self.extract_numeric_core(id_b)
        
        # Check if one core contains the other (for transformations like INV-123 → 2025123)
        if core_a and core_b:
            features['id_core_exact'] = 1.0 if core_a == core_b else 0.0
            features['id_core_contains'] = 1.0 if (core_a in core_b or core_b in core_a) else 0.0
            
            # Calculate similarity even if not exact match
            core_similarity = self.string_similarity(core_a, core_b)
            features['id_core_levenshtein'] = core_similarity['levenshtein']
        else:
            features.update({'id_core_exact': 0.0, 'id_core_contains': 0.0, 'id_core_levenshtein': 0.0})
        
        # Pattern type matching
        def get_pattern_type(id_str):
            if pd.isna(id_str):
                return "null"
            id_str = str(id_str)
            if 'INV-' in id_str:
                return "inv_format"
            elif 'REF-' in id_str:
                return "ref_format"
            elif '#' in id_str and '::' in id_str:
                return "hash_format"
            elif '/' in id_str:
                return "slash_format"
            elif '-' in id_str and 'INV-' not in id_str and 'REF-' not in id_str:
                return "dash_format"
            elif re.match(r'^\d+$', id_str):
                return "numeric_only"
            else:
                return "other"
        
        pattern_a = get_pattern_type(id_a)
        pattern_b = get_pattern_type(id_b)
        features['id_same_pattern'] = 1.0 if pattern_a == pattern_b else 0.0
        features['id_pattern_compatibility'] = 1.0 if (
            (pattern_a == "inv_format" and pattern_b in ["numeric_only", "dash_format", "slash_format"]) or
            (pattern_b == "inv_format" and pattern_a in ["numeric_only", "dash_format", "slash_format"]) or
            pattern_a == pattern_b
        ) else 0.0
        
        return features

    def amount_features(self, amount_a: float, amount_b: float) -> Dict[str, float]:
        """Extract amount-related features"""
        features = {}
        
        try:
            amt_a = float(amount_a) if not pd.isna(amount_a) else 0.0
            amt_b = float(amount_b) if not pd.isna(amount_b) else 0.0
            
            # Exact match
            features['amount_exact_match'] = 1.0 if abs(amt_a - amt_b) < 0.01 else 0.0
            
            # Percentage difference
            if max(amt_a, amt_b) > 0:
                pct_diff = abs(amt_a - amt_b) / max(amt_a, amt_b)
                features['amount_pct_diff'] = min(pct_diff, 1.0)  # Cap at 100%
                features['amount_close_match'] = 1.0 if pct_diff < 0.01 else 0.0  # Within 1%
                features['amount_reasonable_match'] = 1.0 if pct_diff < 0.05 else 0.0  # Within 5%
            else:
                features['amount_pct_diff'] = 1.0
                features['amount_close_match'] = 0.0
                features['amount_reasonable_match'] = 0.0
            
            # Amount magnitude similarity
            if amt_a > 0 and amt_b > 0:
                ratio = min(amt_a, amt_b) / max(amt_a, amt_b)
                features['amount_ratio'] = ratio
            else:
                features['amount_ratio'] = 0.0
                
        except (ValueError, TypeError):
            features.update({
                'amount_exact_match': 0.0, 'amount_pct_diff': 1.0, 'amount_close_match': 0.0,
                'amount_reasonable_match': 0.0, 'amount_ratio': 0.0
            })
        
        return features

    def date_features(self, date_a: str, date_b: str) -> Dict[str, float]:
        """Extract date-related features"""
        features = {}
        
        try:
            dt_a = pd.to_datetime(date_a)
            dt_b = pd.to_datetime(date_b)
            
            # Exact date match
            features['date_exact_match'] = 1.0 if dt_a.date() == dt_b.date() else 0.0
            
            # Date difference in days
            date_diff = abs((dt_a - dt_b).days)
            features['date_diff_days'] = min(date_diff, 365) / 365  # Normalize to [0,1]
            features['date_within_1_day'] = 1.0 if date_diff <= 1 else 0.0
            features['date_within_7_days'] = 1.0 if date_diff <= 7 else 0.0
            
        except (ValueError, TypeError):
            features.update({
                'date_exact_match': 0.0, 'date_diff_days': 1.0, 
                'date_within_1_day': 0.0, 'date_within_7_days': 0.0
            })
        
        return features

    def extract_pair_features(self, record_a: Dict, record_b: Dict) -> Dict[str, float]:
        """Extract all features for a record pair"""
        features = {}
        
        # ID features
        id_features = self.id_pattern_features(record_a.get('invoice_id'), record_b.get('ref_code'))
        features.update(id_features)
        
        # Name similarity features
        name_features = self.string_similarity(record_a.get('customer_name'), record_b.get('client'))
        features.update({f'name_{k}': v for k, v in name_features.items()})
        
        # Email similarity features  
        email_features = self.string_similarity(record_a.get('customer_email'), record_b.get('email'))
        features.update({f'email_{k}': v for k, v in email_features.items()})
        
        # Amount features
        amount_features = self.amount_features(record_a.get('total_amount'), record_b.get('grand_total'))
        features.update(amount_features)
        
        # Date features
        date_features = self.date_features(record_a.get('invoice_date'), record_b.get('doc_date'))
        features.update(date_features)
        
        # PO number similarity
        po_features = self.string_similarity(record_a.get('po_number'), record_b.get('purchase_order'))
        features.update({f'po_{k}': v for k, v in po_features.items()})
        
        return features

# ===== END OF REQUIRED CLASSES =====

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
        with open('record_linking_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file 'record_linking_model.pkl' not found. Please ensure it's in the project directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data with caching"""
    try:
        source_a = pd.read_csv('Project7SourceA.csv')
        source_b = pd.read_csv('Project7SourceB.csv')
        return source_a, source_b
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading sample data: {str(e)}")
        return None, None

def predict_record_match(record_a: dict, record_b: dict, model_data: dict) -> dict:
    """Production function to predict if two records match"""
    if model_data is None:
        return None
    
    try:
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
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

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

# Main app
if source_a_df is not None and source_b_df is not None and model_data is not None:
    st.success("🎉 **System Ready!** Model and data loaded successfully.")
    
    # Single Record Comparison
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

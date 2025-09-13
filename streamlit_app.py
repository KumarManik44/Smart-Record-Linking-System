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
import json

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
        with open('data/record_linking_model.pkl', 'rb') as f:
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
        source_a = pd.read_csv('data/Project7SourceA.csv')
        source_b = pd.read_csv('data/Project7SourceB.csv')
        return source_a, source_b
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading sample data: {str(e)}")
        return None, None

def predict_record_match_with_rules(record_a: dict, record_b: dict, model_data: dict, rule_config: dict = None) -> dict:
    """Enhanced prediction function that applies custom rule weights"""
    if model_data is None:
        return None
    
    try:
        # Extract features using the loaded feature extractor
        features = model_data['feature_extractor'].extract_pair_features(record_a, record_b)
        
        # Apply custom weights if provided
        if rule_config and 'feature_weights' in rule_config:
            weighted_features = {}
            weights = rule_config['feature_weights']
            
            for feature_name, feature_value in features.items():
                # Map feature names to weight keys
                weight_key = None
                if 'id_core_contains' in feature_name:
                    weight_key = 'id_core_contains'
                elif 'id_core_levenshtein' in feature_name:
                    weight_key = 'id_core_levenshtein'
                elif 'id_pattern_compatibility' in feature_name:
                    weight_key = 'id_pattern_compatibility'
                elif 'name_exact_match' in feature_name:
                    weight_key = 'name_exact'
                elif 'name_' in feature_name and ('levenshtein' in feature_name or 'jaro' in feature_name):
                    weight_key = 'name_similarity'
                elif 'email_exact_match' in feature_name:
                    weight_key = 'email_exact'
                elif 'email_' in feature_name and ('levenshtein' in feature_name or 'jaro' in feature_name):
                    weight_key = 'email_similarity'
                elif 'amount_exact_match' in feature_name:
                    weight_key = 'amount_exact'
                elif 'amount_close_match' in feature_name:
                    weight_key = 'amount_close'
                elif 'amount_reasonable_match' in feature_name:
                    weight_key = 'amount_reasonable'
                elif 'date_exact_match' in feature_name:
                    weight_key = 'date_exact'
                elif 'date_within_1_day' in feature_name:
                    weight_key = 'date_within_1_day'
                elif 'date_within_7_days' in feature_name:
                    weight_key = 'date_within_7_days'
                elif 'po_exact_match' in feature_name:
                    weight_key = 'po_exact'
                elif 'po_' in feature_name:
                    weight_key = 'po_similarity'
                
                # Apply weight
                if weight_key and weight_key in weights:
                    weighted_features[feature_name] = feature_value * weights[weight_key]
                else:
                    weighted_features[feature_name] = feature_value
            
            features = weighted_features
        
        feature_vector = np.array([list(features.values())])
        
        # Make prediction
        probabilities = model_data['model'].predict_proba(feature_vector)[0]
        prediction = model_data['model'].predict(feature_vector)[0]
        
        # Calculate tier-based score if tiers are configured
        tier_score = None
        matched_tier = None
        
        if rule_config and 'tiers' in rule_config:
            for tier_name in ['tier1', 'tier2', 'tier3']:
                tier_config = rule_config['tiers'][tier_name]
                tier_features = []
                
                # Extract relevant features for this tier
                for rule in tier_config['rules']:
                    if rule == "ID Exact" and features.get('id_core_exact', 0) == 1.0:
                        tier_features.append(1.0)
                    elif rule == "ID Contains" and features.get('id_core_contains', 0) == 1.0:
                        tier_features.append(1.0)
                    elif rule == "Amount Exact" and features.get('amount_exact_match', 0) == 1.0:
                        tier_features.append(1.0)
                    # Add more rule mappings as needed
                
                if tier_features:
                    tier_score = sum(tier_features) / len(tier_features)
                    if tier_score >= tier_config['min_score']:
                        matched_tier = tier_name
                        break
        
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
            'all_features': features,
            'tier_score': tier_score,
            'matched_tier': matched_tier,
            'custom_weighted': rule_config is not None
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

# Initialize session state for performance tracking
if 'previous_runs' not in st.session_state:
    st.session_state['previous_runs'] = []

if 'accepted_matches' not in st.session_state:
    st.session_state['accepted_matches'] = []

if 'rejected_matches' not in st.session_state:
    st.session_state['rejected_matches'] = []

if 'adopted_patterns' not in st.session_state:
    st.session_state['adopted_patterns'] = []

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
    
    # Configuration section
    st.header("⚙️ Matching Configuration")
    
    # Load configuration from session state if available
    if 'loaded_config' in st.session_state:
        config = st.session_state['loaded_config']
        default_high = config.get('thresholds', {}).get('high_confidence', 0.8)
        default_suspect = config.get('thresholds', {}).get('suspect', 0.3)
        default_max = config.get('thresholds', {}).get('max_candidates', 3)
    else:
        default_high = 0.8
        default_suspect = 0.3
        default_max = 3
    
    col1, col2, col3 = st.columns(3)
    with col1:
        high_confidence_threshold = st.slider("High Confidence Threshold", 0.5, 1.0, default_high, 0.05)
        st.caption("Records above this threshold are automatically matched")
    
    with col2:
        suspect_confidence_threshold = st.slider("Suspect Threshold", 0.1, 0.8, default_suspect, 0.05)
        st.caption("Records between thresholds need manual review")
    
    with col3:
        max_candidates = st.slider("Max Candidates per Record", 1, 10, default_max)
        st.caption("Maximum number of potential matches to consider")
    
    # Advanced Rule Configuration
    with st.expander("🔧 **Advanced Rule Configuration** - Control Matching Logic", expanded=False):
        st.markdown("**Configure rule weights and priorities for different matching criteria**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Feature Weights")
            
            # ID Pattern Rules
            st.markdown("**ID Pattern Rules**")
            id_core_contains_weight = st.slider("ID Core Contains", 0.0, 2.0, 1.0, 0.1, 
                                               help="Weight for when one ID contains the other's numeric core")
            id_core_levenshtein_weight = st.slider("ID Similarity", 0.0, 2.0, 1.0, 0.1,
                                                  help="Weight for ID string similarity")
            id_pattern_compatibility_weight = st.slider("Pattern Compatibility", 0.0, 2.0, 1.0, 0.1,
                                                       help="Weight for compatible ID formats (INV vs numeric)")
            
            # String Matching Rules
            st.markdown("**String Matching Rules**")
            name_exact_weight = st.slider("Name Exact Match", 0.0, 3.0, 1.0, 0.1)
            name_similarity_weight = st.slider("Name Similarity", 0.0, 2.0, 1.0, 0.1)
            email_exact_weight = st.slider("Email Exact Match", 0.0, 2.0, 1.0, 0.1)
            email_similarity_weight = st.slider("Email Similarity", 0.0, 2.0, 1.0, 0.1)
        
        with col2:
            st.subheader("💰 Business Logic Rules")
            
            # Amount Rules
            st.markdown("**Amount Matching Rules**")
            amount_exact_weight = st.slider("Amount Exact Match", 0.0, 3.0, 1.0, 0.1)
            amount_close_weight = st.slider("Amount Close Match (±1%)", 0.0, 2.0, 1.0, 0.1)
            amount_reasonable_weight = st.slider("Amount Reasonable (±5%)", 0.0, 1.5, 1.0, 0.1)
            
            # Date Rules
            st.markdown("**Date Matching Rules**")
            date_exact_weight = st.slider("Date Exact Match", 0.0, 2.0, 1.0, 0.1)
            date_within_1_day_weight = st.slider("Date Within 1 Day", 0.0, 1.5, 1.0, 0.1)
            date_within_7_days_weight = st.slider("Date Within 7 Days", 0.0, 1.0, 1.0, 0.1)
            
            # PO Rules
            st.markdown("**Purchase Order Rules**")
            po_exact_weight = st.slider("PO Exact Match", 0.0, 2.0, 1.0, 0.1)
            po_similarity_weight = st.slider("PO Similarity", 0.0, 1.5, 1.0, 0.1)
        
        # Rule Tier Configuration
        st.subheader("🏆 Rule Tier Priorities")
        st.markdown("**Define progressive matching tiers (Tier 1 = Strictest, Tier 3 = Most Flexible)**")
        
        tier_col1, tier_col2, tier_col3 = st.columns(3)
        
        with tier_col1:
            st.markdown("**Tier 1: Exact Matches**")
            tier1_rules = st.multiselect(
                "Select Tier 1 Rules:",
                ["ID Exact", "Name Exact", "Email Exact", "Amount Exact", "Date Exact", "PO Exact"],
                default=["ID Exact", "Amount Exact"],
                help="Rules that must match exactly"
            )
            tier1_min_score = st.slider("Tier 1 Min Score", 0.8, 1.0, 0.95, 0.01)
        
        with tier_col2:
            st.markdown("**Tier 2: Pattern Matches**")
            tier2_rules = st.multiselect(
                "Select Tier 2 Rules:",
                ["ID Contains", "Name Similarity", "Email Similarity", "Amount Close", "Date 1-Day"],
                default=["ID Contains", "Name Similarity", "Amount Close"],
                help="Rules with pattern-based matching"
            )
            tier2_min_score = st.slider("Tier 2 Min Score", 0.6, 0.9, 0.75, 0.01)
        
        with tier_col3:
            st.markdown("**Tier 3: Fuzzy Matches**")
            tier3_rules = st.multiselect(
                "Select Tier 3 Rules:",
                ["Name Fuzzy", "Email Fuzzy", "Amount Reasonable", "Date 7-Days", "PO Similarity"],
                default=["Name Fuzzy", "Amount Reasonable"],
                help="Flexible rules for edge cases"
            )
            tier3_min_score = st.slider("Tier 3 Min Score", 0.3, 0.7, 0.5, 0.01)
        
        # Tie-breaker Configuration
        st.subheader("⚖️ Tie-Breaker Rules")
        st.markdown("**When multiple candidates have similar scores, use these tie-breakers in order:**")
        
        tie_breaker_priority = st.multiselect(
            "Tie-breaker Priority (drag to reorder):",
            ["Date Proximity", "Amount Accuracy", "Name Similarity", "Email Match", "ID Similarity"],
            default=["Amount Accuracy", "Date Proximity", "Name Similarity"],
            help="First tie-breaker has highest priority"
        )
        
        # Reset to defaults
        col_reset1, col_reset2 = st.columns([1, 4])
        with col_reset1:
            if st.button("🔄 Reset to Defaults"):
                st.experimental_rerun()
        with col_reset2:
            st.caption("Reset all rule weights and configurations to default values")
    
    # Store configuration in session state
    rule_config = {
        'feature_weights': {
            'id_core_contains': id_core_contains_weight,
            'id_core_levenshtein': id_core_levenshtein_weight,
            'id_pattern_compatibility': id_pattern_compatibility_weight,
            'name_exact': name_exact_weight,
            'name_similarity': name_similarity_weight,
            'email_exact': email_exact_weight,
            'email_similarity': email_similarity_weight,
            'amount_exact': amount_exact_weight,
            'amount_close': amount_close_weight,
            'amount_reasonable': amount_reasonable_weight,
            'date_exact': date_exact_weight,
            'date_within_1_day': date_within_1_day_weight,
            'date_within_7_days': date_within_7_days_weight,
            'po_exact': po_exact_weight,
            'po_similarity': po_similarity_weight,
        },
        'tiers': {
            'tier1': {'rules': tier1_rules, 'min_score': tier1_min_score},
            'tier2': {'rules': tier2_rules, 'min_score': tier2_min_score},
            'tier3': {'rules': tier3_rules, 'min_score': tier3_min_score},
        },
        'tie_breakers': tie_breaker_priority
    }
    
    # Display current configuration summary
    if st.checkbox("📋 Show Current Configuration Summary"):
        st.json(rule_config)
    
    # Configuration Management
    st.subheader("💾 Configuration Management")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Configuration", help="Save current rule settings as JSON"):
            config_to_save = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'thresholds': {
                    'high_confidence': high_confidence_threshold,
                    'suspect': suspect_confidence_threshold,
                    'max_candidates': max_candidates
                },
                'rule_config': rule_config,
                'description': f"Configuration saved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            # Convert to JSON string for download
            config_json = json.dumps(config_to_save, indent=2)
            st.download_button(
                label="📥 Download Configuration",
                data=config_json,
                file_name=f"record_linking_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ Configuration ready for download!")
    
    with col2:
        uploaded_config = st.file_uploader("📂 Load Configuration", type="json", help="Upload a saved configuration file")
        if uploaded_config is not None:
            try:
                loaded_config = json.load(uploaded_config)
                st.session_state['loaded_config'] = loaded_config
                st.success(f"✅ Configuration loaded from {loaded_config.get('timestamp', 'unknown time')}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Error loading configuration: {str(e)}")
    
    with col3:
        if st.button("🔄 Reset All Settings", help="Reset to default configuration"):
            # Clear session state to reset all sliders
            for key in list(st.session_state.keys()):
                if key.startswith('slider_') or key.startswith('config_') or key == 'loaded_config':
                    del st.session_state[key]
            st.experimental_rerun()
    
    with col4:
        st.markdown("**Quick Presets:**")
        preset = st.selectbox(
            "Load Preset:",
            ["Custom", "Strict Matching", "Balanced", "Flexible"],
            help="Load predefined rule configurations"
        )
        
        if preset != "Custom":
            if st.button(f"Apply {preset}"):
                if preset == "Strict Matching":
                    preset_config = {
                        'timestamp': datetime.now().isoformat(),
                        'thresholds': {'high_confidence': 0.95, 'suspect': 0.80, 'max_candidates': 3},
                        'rule_config': {
                            'feature_weights': {k: 2.0 if 'exact' in k else 1.0 for k in rule_config['feature_weights']},
                            'tiers': rule_config['tiers'],
                            'tie_breakers': rule_config['tie_breakers']
                        }
                    }
                elif preset == "Balanced":
                    preset_config = {
                        'timestamp': datetime.now().isoformat(),
                        'thresholds': {'high_confidence': 0.75, 'suspect': 0.50, 'max_candidates': 5},
                        'rule_config': rule_config
                    }
                elif preset == "Flexible":
                    preset_config = {
                        'timestamp': datetime.now().isoformat(),
                        'thresholds': {'high_confidence': 0.60, 'suspect': 0.30, 'max_candidates': 10},
                        'rule_config': {
                            'feature_weights': {k: 0.5 if 'exact' in k else 1.0 for k in rule_config['feature_weights']},
                            'tiers': rule_config['tiers'],
                            'tie_breakers': rule_config['tie_breakers']
                        }
                    }
                
                st.session_state['loaded_config'] = preset_config
                st.experimental_rerun()

    # Run batch matching
    if st.button("🚀 Run Batch Record Linking", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            st.info("🔄 **Processing all record combinations...**")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Process all combinations
        all_results = []
        total_combinations = min(len(source_a_df) * len(source_b_df), 10000)  # Cap at 10k for demo
        current_combo = 0
        
        sample_size_a = min(50, len(source_a_df))  # Sample for demo
        sample_size_b = min(50, len(source_b_df))
        
        sample_a = source_a_df.sample(n=sample_size_a, random_state=42)
        sample_b = source_b_df.sample(n=sample_size_b, random_state=42)
        
        for i, (_, record_a) in enumerate(sample_a.iterrows()):
            for j, (_, record_b) in enumerate(sample_b.iterrows()):
                current_combo += 1
                
                # Update progress every 50 combinations
                if current_combo % 50 == 0:
                    progress = min(current_combo / (sample_size_a * sample_size_b), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f'Processed {current_combo}/{sample_size_a * sample_size_b} combinations')
                
                # Get prediction with custom rules
                result = predict_record_match_with_rules(record_a.to_dict(), record_b.to_dict(), model_data, rule_config)
                
                if result and result['match_probability'] > suspect_confidence_threshold:
                    all_results.append({
                        'source_a_idx': record_a.name,
                        'source_b_idx': record_b.name,
                        'record_a': record_a.to_dict(),
                        'record_b': record_b.to_dict(),
                        'match_probability': result['match_probability'],
                        'confidence_level': result['confidence'],
                        'prediction': result['prediction'],
                        'top_features': result['top_contributing_features'],
                        'all_features': result['all_features'],
                        'tier_score': result.get('tier_score'),
                        'matched_tier': result.get('matched_tier'),
                        'custom_weighted': result.get('custom_weighted', False)
                    })
        
        # Clear progress
        progress_container.empty()
        
        # Categorize results
        matched_results = [r for r in all_results if r['match_probability'] >= high_confidence_threshold]
        suspect_results = [r for r in all_results if suspect_confidence_threshold <= r['match_probability'] < high_confidence_threshold]
        
        # Sort by probability
        matched_results.sort(key=lambda x: x['match_probability'], reverse=True)
        suspect_results.sort(key=lambda x: x['match_probability'], reverse=True)
        
        # Display results in tabs
        st.success(f"✅ **Processing Complete!** Found {len(matched_results)} matches and {len(suspect_results)} suspects")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            f"✅ Matched ({len(matched_results)})", 
            f"❓ Suspects ({len(suspect_results)})", 
            f"❌ Unmatched", 
            f"📊 Analytics"
        ])
        
        with tab1:
            st.header("✅ High-Confidence Matches")
            st.markdown(f"Records with **≥{high_confidence_threshold:.0%}** match probability")
            
            if matched_results:
                for i, match in enumerate(matched_results[:10]):  # Show top 10
                    tier_info = f" (Tier: {match['matched_tier']})" if match['matched_tier'] else ""
                    weight_info = " 🎛️" if match['custom_weighted'] else ""
                    
                    with st.expander(f"Match {i+1}: {match['match_probability']:.1%} confidence{tier_info}{weight_info}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📄 Source A")
                            st.json(match['record_a'])
                        
                        with col2:
                            st.subheader("📄 Source B") 
                            st.json(match['record_b'])
                        
                        st.subheader("🧠 Why This Matched")
                        for j, (feature, score) in enumerate(match['top_features'][:3]):
                            st.write(f"**{j+1}.** {feature.replace('_', ' ').title()}: {score:.4f}")
                        
                        if match['tier_score']:
                            st.write(f"**Tier Score:** {match['tier_score']:.3f}")
                
                # Export functionality for matched results
                st.subheader("📤 Export Results")
                
                export_format = st.radio("Export Format:", ["CSV", "JSON", "Excel"], horizontal=True)
                
                if st.button("📥 Download Matched Results"):
                    # Prepare export data
                    export_data = []
                    for i, match in enumerate(matched_results):
                        export_data.append({
                            'match_rank': i + 1,
                            'match_probability': match['match_probability'],
                            'confidence_level': match['confidence_level'],
                            'matched_tier': match.get('matched_tier', 'N/A'),
                            'source_a_id': match['record_a'].get('invoice_id', ''),
                            'source_a_name': match['record_a'].get('customer_name', ''),
                            'source_a_email': match['record_a'].get('customer_email', ''),
                            'source_a_amount': match['record_a'].get('total_amount', ''),
                            'source_a_date': match['record_a'].get('invoice_date', ''),
                            'source_b_id': match['record_b'].get('ref_code', ''),
                            'source_b_name': match['record_b'].get('client', ''),
                            'source_b_email': match['record_b'].get('email', ''),
                            'source_b_amount': match['record_b'].get('grand_total', ''),
                            'source_b_date': match['record_b'].get('doc_date', ''),
                            'top_feature_1': match['top_features'][0][0] if match['top_features'] else '',
                            'top_feature_1_score': match['top_features'][0][1] if match['top_features'] else 0,
                            'top_feature_2': match['top_features'][1][0] if len(match['top_features']) > 1 else '',
                            'top_feature_2_score': match['top_features'][1][1] if len(match['top_features']) > 1 else 0,
                            'top_feature_3': match['top_features'][2][0] if len(match['top_features']) > 2 else '',
                            'top_feature_3_score': match['top_features'][2][1] if len(match['top_features']) > 2 else 0,
                            'processing_timestamp': datetime.now().isoformat(),
                            'configuration_used': 'custom_weights' if match.get('custom_weighted') else 'default'
                        })
                    
                    if export_format == "CSV":
                        df = pd.DataFrame(export_data)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"matched_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "JSON":
                        full_export = {
                            'export_metadata': {
                                'timestamp': datetime.now().isoformat(),
                                'total_matches': len(matched_results),
                                'configuration': rule_config,
                                'thresholds': {
                                    'high_confidence': high_confidence_threshold,
                                    'suspect': suspect_confidence_threshold
                                }
                            },
                            'matches': export_data
                        }
                        
                        json_data = json.dumps(full_export, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"matched_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    st.success(f"✅ Export prepared! {len(export_data)} matched records with full provenance.")
                    
            else:
                st.info("No high-confidence matches found. Consider lowering the threshold.")
        
        with tab2:
            st.header("❓ Suspect Matches - Need Review")
            st.markdown(f"Records with **{suspect_confidence_threshold:.0%} - {high_confidence_threshold:.0%}** match probability")
            
            if suspect_results:
                for i, suspect in enumerate(suspect_results[:10]):  # Show top 10
                    tier_info = f" (Tier: {suspect['matched_tier']})" if suspect['matched_tier'] else ""
                    weight_info = " 🎛️" if suspect['custom_weighted'] else ""
                    
                    with st.expander(f"Suspect {i+1}: {suspect['match_probability']:.1%} confidence{tier_info}{weight_info}"):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.subheader("📄 Source A")
                            st.json(suspect['record_a'])
                        
                        with col2:
                            st.subheader("📄 Source B")
                            st.json(suspect['record_b'])
                        
                        with col3:
                            st.subheader("👥 Review Decision")
                            
                            if st.button("✅ Accept Match", key=f"accept_{i}"):
                                st.session_state['accepted_matches'].append({
                                    'pair': suspect,
                                    'timestamp': datetime.now().isoformat(),
                                    'analyst_decision': 'accepted'
                                })
                                st.success("✅ Match accepted and recorded!")
                            
                            if st.button("❌ Reject Match", key=f"reject_{i}"):
                                st.session_state['rejected_matches'].append({
                                    'pair': suspect,
                                    'timestamp': datetime.now().isoformat(),
                                    'analyst_decision': 'rejected'
                                })
                                st.error("❌ Match rejected and recorded!")
                            
                            if st.button("📝 Adopt Pattern", key=f"adopt_{i}"):
                                pattern = {
                                    'top_features': suspect['top_features'][:3],
                                    'match_probability': suspect['match_probability'],
                                    'timestamp': datetime.now().isoformat(),
                                    'pattern_type': 'analyst_adopted'
                                }
                                st.session_state['adopted_patterns'].append(pattern)
                                st.info("📝 Pattern adopted for future matching!")
                        
                        st.subheader("🧠 Why This was Flagged")
                        for j, (feature, score) in enumerate(suspect['top_features'][:3]):
                            st.write(f"**{j+1}.** {feature.replace('_', ' ').title()}: {score:.4f}")
                
                # Show analyst feedback summary
                if st.session_state['accepted_matches'] or st.session_state['rejected_matches']:
                    st.subheader("📋 Analyst Feedback Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accepted", len(st.session_state['accepted_matches']))
                    col2.metric("Rejected", len(st.session_state['rejected_matches']))
                    col3.metric("Patterns Adopted", len(st.session_state['adopted_patterns']))
                    
            else:
                st.info("No suspect matches found.")
        
        with tab3:
            st.header("❌ Unmatched Records")
            st.markdown(f"Records with **<{suspect_confidence_threshold:.0%}** match probability")
            
            # Calculate unmatched counts
            matched_source_a = {r['source_a_idx'] for r in all_results}
            matched_source_b = {r['source_b_idx'] for r in all_results}
            
            unmatched_a_count = len(sample_a) - len(matched_source_a)
            unmatched_b_count = len(sample_b) - len(matched_source_b)
            
            col1, col2 = st.columns(2)
            col1.metric("Unmatched Source A Records", unmatched_a_count)
            col2.metric("Unmatched Source B Records", unmatched_b_count)
            
            st.info("💡 **Tip:** These records had no good matches. Consider adjusting thresholds or adding new matching rules.")
            
            # Export unmatched records
            if st.button("📥 Export Unmatched Records"):
                unmatched_a_records = sample_a[~sample_a.index.isin(matched_source_a)]
                unmatched_b_records = sample_b[~sample_b.index.isin(matched_source_b)]
                
                unmatched_export = {
                    'export_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'unmatched_source_a': len(unmatched_a_records),
                        'unmatched_source_b': len(unmatched_b_records),
                        'threshold_used': suspect_confidence_threshold
                    },
                    'unmatched_source_a': unmatched_a_records.to_dict('records'),
                    'unmatched_source_b': unmatched_b_records.to_dict('records')
                }
                
                json_data = json.dumps(unmatched_export, indent=2, default=str)
                st.download_button(
                    label="📥 Download Unmatched Records",
                    data=json_data,
                    file_name=f"unmatched_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with tab4:
            st.header("📊 Matching Analytics")
            
            # Summary metrics
            total_processed = len(all_results)
            match_rate = len(matched_results) / total_processed * 100 if total_processed > 0 else 0
            custom_weighted_count = sum(1 for r in all_results if r.get('custom_weighted', False))
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Comparisons", f"{sample_size_a * sample_size_b:,}")
            col2.metric("Candidates Found", total_processed)
            col3.metric("Match Rate", f"{match_rate:.1f}%")
            col4.metric("Custom Weighted", custom_weighted_count)
            
            # Distribution chart
            if all_results:
                probabilities = [r['match_probability'] for r in all_results]
                
                fig = px.histogram(
                    x=probabilities,
                    nbins=20,
                    title="Distribution of Match Probabilities",
                    labels={'x': 'Match Probability', 'y': 'Count'}
                )
                fig.add_vline(x=high_confidence_threshold, line_dash="dash", line_color="green", 
                             annotation_text="High Confidence")
                fig.add_vline(x=suspect_confidence_threshold, line_dash="dash", line_color="orange",
                             annotation_text="Suspect Threshold")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance tracking and improvement metrics
            st.subheader("📈 Performance Tracking")
            
            current_run = {
                'timestamp': datetime.now().isoformat(),
                'total_comparisons': sample_size_a * sample_size_b,
                'candidates_found': total_processed,
                'matches_found': len(matched_results),
                'suspects_found': len(suspect_results),
                'match_rate': match_rate,
                'configuration': {
                    'high_threshold': high_confidence_threshold,
                    'suspect_threshold': suspect_confidence_threshold,
                    'custom_weights_used': custom_weighted_count > 0
                }
            }
            
            if st.button("📊 Record This Run"):
                st.session_state['previous_runs'].append(current_run)
                st.success("✅ Run recorded for performance tracking!")
            
            # Show improvement over time
            if len(st.session_state['previous_runs']) > 1:
                st.subheader("📈 Improvement Over Time")
                
                runs_df = pd.DataFrame(st.session_state['previous_runs'])
                runs_df['run_number'] = range(1, len(runs_df) + 1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Match rate improvement
                    fig_improvement = px.line(
                        runs_df, 
                        x='run_number', 
                        y='match_rate',
                        title="Match Rate Improvement Over Runs",
                        labels={'run_number': 'Run Number', 'match_rate': 'Match Rate (%)'}
                    )
                    st.plotly_chart(fig_improvement, use_container_width=True)
                
                with col2:
                    # Suspects trend
                    fig_suspects = px.line(
                        runs_df,
                        x='run_number',
                        y='suspects_found',
                        title="Suspects Requiring Review (Should Decrease)",
                        labels={'run_number': 'Run Number', 'suspects_found': 'Suspects Count'}
                    )
                    st.plotly_chart(fig_suspects, use_container_width=True)
                
                # Improvement metrics
                latest_run = runs_df.iloc[-1]
                previous_run = runs_df.iloc[-2]
                
                match_rate_change = latest_run['match_rate'] - previous_run['match_rate']
                suspects_change = latest_run['suspects_found'] - previous_run['suspects_found']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Match Rate Change", f"{match_rate_change:+.1f}%", delta=f"{match_rate_change:+.1f}%")
                col2.metric("Suspects Change", f"{suspects_change:+.0f}", delta=f"{suspects_change:+.0f}")
                col3.metric("Total Runs", len(runs_df))
                
                if match_rate_change > 0:
                    st.success("🎉 **Improvement Detected!** Match rate increased from previous run.")
                elif suspects_change < 0:
                    st.success("🎉 **Improvement Detected!** Fewer suspects requiring manual review.")
            
            # Export analytics
            if st.button("📊 Export Analytics Report"):
                analytics_report = {
                    'report_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'report_type': 'matching_analytics'
                    },
                    'current_run': current_run,
                    'historical_runs': st.session_state['previous_runs'],
                    'analyst_feedback': {
                        'accepted_matches': len(st.session_state['accepted_matches']),
                        'rejected_matches': len(st.session_state['rejected_matches']),
                        'adopted_patterns': len(st.session_state['adopted_patterns'])
                    },
                    'configuration': rule_config
                }
                
                json_data = json.dumps(analytics_report, indent=2)
                st.download_button(
                    label="📥 Download Analytics Report",
                    data=json_data,
                    file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

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
    Progressive Matching with Full Analyst Control & Export Capabilities
</div>
""", unsafe_allow_html=True)

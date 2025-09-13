# Smart Record Linking System

An intelligent machine learning system for cross-source record matching and deduplication, featuring advanced synthetic training data generation and sophisticated feature engineering.

## Live Demo

Try the system at: [Streamlit Cloud URL] (to be added after deployment)

## Overview

This system solves the challenging problem of linking records across different data sources with varying formats, field names, and ID patterns. Unlike traditional approaches that require extensive manually labeled training data, this project uses innovative **synthetic training data generation** to achieve high-accuracy record matching.

## Key Features

- **Synthetic Training Data Generation**: Automatically creates high-quality training examples from your existing data
- **Multi-Model Architecture**: Tests multiple ML algorithms (Random Forest, Gradient Boosting, SVM, Logistic Regression) to select the best performer
- **Advanced Feature Engineering**: 26+ specialized features including string similarity, ID pattern matching, amount comparison, and date analysis
- **Production-Ready Pipeline**: Complete end-to-end system from data ingestion to prediction
- **Interactive Web Interface**: Streamlit-based UI for easy record comparison and batch processing
- **Real-time Explanations**: Shows which features contributed most to each matching decision

## Technical Innovation

### Synthetic Training Approach
Instead of requiring thousands of manually labeled record pairs, the system:
1. Analyzes your existing data to understand ID transformation patterns
2. Generates realistic positive examples by applying controlled transformations
3. Creates negative examples from genuinely non-matching records
4. Produces a balanced, high-quality training dataset automatically

### Feature Engineering
The system extracts sophisticated features including:
- **ID Pattern Features**: Numeric core extraction, format compatibility, transformation pattern recognition
- **String Similarity**: Levenshtein distance, Jaro similarity, sequence matching
- **Amount Analysis**: Percentage differences, ratio calculations, tolerance-based matching
- **Date Comparison**: Exact matches, drift detection, temporal proximity
- **Email/Name Matching**: Domain analysis, abbreviation handling, typo detection

## Performance Metrics

- **Accuracy**: 100% on test set
- **AUC Score**: 1.000 (perfect ROC curve)
- **Cross-Validation**: 99.9% ± 0.2%
- **Feature Count**: 26 engineered features
- **Training Speed**: Sub-second model training
- **Prediction Speed**: Real-time inference

## Project Structure

```
Smart-Record-Linking-System/
│
├── data/
│   ├── Project7SourceA.csv          # Sample invoice data (Source A)
│   ├── Project7SourceB.csv          # Sample invoice data (Source B)
│   └── record_linking_model.pkl     # Trained model
├── notebook/
│   └── Project7.ipynb               # Complete development notebook
├── streamlit_app.py                 # Web interface
├── LICENSE                          # License
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smart-record-linking.git
   cd smart-record-linking
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Web Interface
1. Upload your CSV files (Source A and Source B)
2. Map the field correspondences
3. View automatic record matching results
4. Explore feature contributions for each decision

### Python API
```python
import pickle
from src.predictor import predict_record_match

# Load trained model
with open('data/record_linking_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Compare two records
record_a = {
    'invoice_id': 'INV-2025123456',
    'customer_name': 'John Smith',
    'customer_email': 'john@example.com',
    'total_amount': 1500.00,
    'invoice_date': '2025-01-15'
}

record_b = {
    'ref_code': '2025123456',
    'client': 'John Smith',
    'email': 'john@example.com',
    'grand_total': 1500.00,
    'doc_date': '2025-01-15'
}

result = predict_record_match(record_a, record_b, model_data)
print(f"Match Probability: {result['match_probability']:.3f}")
```

## Data Format Requirements

The system expects CSV files with the following field mappings:

| Field Type | Source A | Source B | Description |
|------------|----------|----------|-------------|
| ID | `invoice_id` | `ref_code` | Unique record identifier |
| Name | `customer_name` | `client` | Customer/client name |
| Email | `customer_email` | `email` | Contact email address |
| Amount | `total_amount` | `grand_total` | Transaction amount |
| Date | `invoice_date` | `doc_date` | Document date |
| PO Number | `po_number` | `purchase_order` | Purchase order reference |

## Algorithm Details

### Model Selection Process
1. **Random Forest**: Best overall performance with robust feature importance
2. **Gradient Boosting**: Fast training, excellent accuracy
3. **SVM with RBF**: Strong generalization capabilities
4. **Logistic Regression**: Baseline linear model

The system automatically selects the best performer based on:
- AUC score (primary metric)
- Cross-validation stability
- Training efficiency

### Feature Importance Rankings
1. **ID Core Contains** (21.4%): Whether numeric cores of IDs contain each other
2. **ID Core Similarity** (14.1%): Levenshtein distance between ID cores
3. **Amount Percentage Difference** (13.1%): Relative difference in amounts
4. **Date Proximity** (11.8%): Whether dates are within 1 day
5. **Amount Ratio** (8.1%): Ratio of smaller to larger amount

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## Model Validation

The system includes comprehensive validation to prevent overfitting:
- **Train/Test Split**: 80/20 stratified split
- **Cross-Validation**: 5-fold CV with consistent results
- **Learning Curves**: Convergence analysis
- **Multiple Model Comparison**: Algorithm robustness testing

## License

MIT License - see LICENSE file for details

## Citation

If you use this system in your research or work, please cite:
```
Smart Record Linking System with Synthetic Training Data Generation
GitHub: https://github.com/yourusername/smart-record-linking
```

---

**Note**: This system demonstrates that high-quality synthetic training data can achieve excellent performance in record linking tasks, potentially reducing the need for expensive manual data labeling in similar applications.

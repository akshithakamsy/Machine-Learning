
## Machine Learning

Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.

## Core Use Cases

• **Pattern Learning** - ML systems learn patterns from historical data

• **Prediction** - Using learned patterns to make predictions on new, unseen data  

• **Automation** - Reducing the need for manual rule creation and maintenance

## ML vs Rule-Based Systems

| Aspect | Rule-Based Systems | Machine Learning |
|--------|-------------------|------------------|
| **Development** | Manual rule creation | Data-driven training |
| **Logic** | IF-THEN structures | Pattern recognition |
| **Outcomes** | Deterministic | Probabilistic |
| **Maintenance** | High (manual updates) | Low (retraining) |
| **Scalability** | Limited | High |
| **Edge Cases** | Poor handling | Better generalization |
| **Interpretability** | High | Variable |
| **Data Requirements** | None | Large datasets needed |

## Types of Machine Learning

### Supervised Learning

• Uses labeled training data

• Learns mapping from inputs to outputs

• Has known correct answers during training


#### Classification (Discrete Categories)

• **Purpose**: Predicts discrete categories or classes

• **Examples**: 
  - Email spam detection (spam/not spam)
  - Image recognition (cat/dog/bird)
  - Medical diagnosis (disease/no disease)

• **Algorithms**: Logistic Regression, Decision Trees, Random Forest, SVM

#### Regression (Continuous Values)

• **Purpose**: Predicts continuous numerical values

• **Examples**:
  - House price prediction
  - Stock price forecasting
  - Temperature prediction

• **Algorithms**: Linear Regression, Polynomial Regression, Random Forest Regressor

### Other Learning Types

#### Unsupervised Learning (No Labels)
• **Purpose**: Finding patterns in unlabeled data


• **Examples**:
  - Customer segmentation (grouping similar customers)
  - Market basket analysis (items bought together)
  - Anomaly detection (fraud detection)
  - Data compression and dimensionality reduction
    
• **Algorithms**: K-Means Clustering, Hierarchical Clustering, PCA

#### Reinforcement Learning (Learning through Trial)

• **Purpose**: Learning through interaction and feedback

• **Examples**:
  - Game playing (Chess, Go, video games)
  - Robot navigation and control
  - Trading algorithms
  - Recommendation system optimization
  - 
• **Algorithms**: Q-Learning, Policy Gradient, Actor-Critic

## CRISP-DM Methodology

| Phase | Purpose | Key Activities |
|-------|---------|----------------|
| **1. Business Understanding** | Define objectives | • Assess situation<br>• Determine goals<br>• Create project plan |
| **2. Data Understanding** | Explore available data | • Collect initial data<br>• Describe data<br>• Verify quality |
| **3. Data Preparation** | Prepare final dataset | • Clean data<br>• Engineer features<br>• Transform data |
| **4. Modeling** | Build and test models | • Select techniques<br>• Build models<br>• Assess performance |
| **5. Evaluation** | Assess business value | • Evaluate results<br>• Review process<br>• Plan next steps |
| **6. Deployment** | Put model into production | • Deploy model<br>• Monitor performance<br>• Maintain system |

## Model Selection Process
### Key Steps
• **Problem Definition** - Classification vs Regression, performance requirements

• **Data Splitting** - Divide data into Train/Validation/Test sets

• **Model Candidates** - Consider multiple algorithms and baselines

• **Evaluation** - Use appropriate metrics to compare models

• **Cross-Validation** - Ensure robust performance estimates

• **Hyperparameter Tuning** - Optimize model parameters

• **Final Selection** - Choose best model based on validation performance

### Data Splitting Strategy
| Dataset | Typical Ratio | Purpose |
|---------|---------------|---------|
| **Training** | 60-70% | Train the model |
| **Validation** | 15-20% | Select best model |
| **Test** | 15-20% | Final unbiased evaluation |

## Evaluation Metrics

### Classification Metrics
| Metric | Purpose | When to Use |
|--------|---------|-------------|
| **Accuracy** | Overall correctness | Balanced datasets |
| **Precision** | Avoid false positives | When false positives are costly |
| **Recall** | Catch all positives | When false negatives are costly |
| **F1-Score** | Balance precision/recall | Imbalanced datasets |
| **ROC AUC** | Overall discrimination | Binary classification |

### Regression Metrics
| Metric | Purpose | Characteristics |
|--------|---------|----------------|
| **MAE** | Average absolute error | Easy to interpret |
| **MSE** | Penalizes large errors | Sensitive to outliers |
| **RMSE** | Same units as target | Most common |
| **R-squared** | Explained variance | 0-1 scale, higher better |





## Key Components of Supervised Learning

| Component | Description | Example |
|-----------|-------------|---------|
| **Training Data** | Examples with known answers | Historical sales with outcomes |
| **Features (X)** | Input variables/attributes | Age, income, location |
| **Target (y)** | Output we want to predict | Purchase decision, price |
| **Algorithm** | Method to learn mapping | Linear regression, decision tree |
| **Model** | Trained algorithm ready for predictions | Trained classifier |

## Supervised Learning Workflow

1. **Data Collection** → Gather relevant historical data
2. **Data Preprocessing** → Clean and prepare data
3. **Feature Engineering** → Create meaningful input variables
4. **Model Training** → Use algorithm to learn from data
5. **Model Evaluation** → Test performance on unseen data
6. **Model Deployment** → Put model into production use

## Common Pitfalls to Avoid


• **Overfitting** - Model memorizes training data, poor generalization

• **Data Leakage** - Using future information to predict the past

• **Inadequate Validation** - Not properly testing on unseen data

• **Ignoring Business Context** - Focusing only on technical metrics

• **Poor Data Quality** - Garbage in, garbage out

• **Feature Engineering Neglect** - Using raw data without thoughtful preparation

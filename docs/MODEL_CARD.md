# Model Card: Electricity Demand Forecasting

## Model Details

### Overview
This model card describes the electricity demand forecasting models deployed in the Forecast Service. The primary production model is **LightGBM** with multi-quantile regression.

### Model Type
- **Primary**: LightGBM (Gradient Boosting Decision Trees)
- **Alternatives**: LSTM, TCN, SARIMA, Seasonal Naive

### Version
- Model Version: 1.0.0
- Last Updated: 2024-01-01
- Framework: LightGBM 4.x, PyTorch 2.x

### Developers
- Forecast Service Team

### License
- MIT License

---

## Intended Use

### Primary Use Cases
1. **Day-ahead forecasting**: Predict electricity demand for the next 24 hours
2. **Week-ahead planning**: Extended forecasts up to 168 hours
3. **Grid operations**: Support load balancing and resource scheduling
4. **Market operations**: Inform bidding strategies in electricity markets

### Intended Users
- Grid operators
- Energy traders
- Utility planners
- Researchers and analysts

### Out-of-Scope Uses
- **NOT for**: Real-time grid control (latency requirements)
- **NOT for**: Financial trading decisions without human oversight
- **NOT for**: Safety-critical systems without redundant systems
- **NOT for**: Forecasting in regions without similar load characteristics

---

## Training Data

### Data Source
- **Primary**: PJM Interconnection hourly load data
- **Fallback**: Synthetic data generator (for testing and development)

### Data Description
| Attribute | Description |
|-----------|-------------|
| Time Range | 2015-01-01 to 2018-12-31 |
| Frequency | Hourly |
| Target Variable | Load (MW) |
| Geographic Coverage | PJM Interconnection (US East) |
| Data Size | ~35,000 observations |

### Preprocessing
1. Missing value interpolation (time-based, max 24-hour gap)
2. Outlier detection using IQR method
3. Frequency validation (hourly)
4. Duplicate removal

### Synthetic Data
The synthetic data generator creates realistic load patterns including:
- Daily seasonality (morning/evening peaks)
- Weekly patterns (weekday/weekend)
- Annual seasonality (summer/winter peaks)
- Holiday effects
- Temperature effects
- Random noise

**Important**: Synthetic data is clearly labeled and should not be confused with real data.

---

## Features

### Feature Groups
| Group | Count | Description |
|-------|-------|-------------|
| Lag Features | 12 | Historical load values (1h to 168h) |
| Rolling Statistics | 20 | Mean, std, min, max over various windows |
| Calendar | 8 | Hour, day of week, month, etc. |
| Fourier | 18 | Sine/cosine for daily, weekly, annual cycles |
| Holiday | 3 | Holiday indicators |
| Temperature | 9 | Temperature and heating/cooling degrees |
| **Total** | **~70** | |

See [FEATURES.md](FEATURES.md) for detailed feature documentation.

---

## Model Architecture

### LightGBM (Primary)
```
Algorithm: Gradient Boosting Decision Trees
Objective: Quantile Regression
Number of Trees: 500 (with early stopping)
Learning Rate: 0.05
Max Depth: Unlimited (controlled by num_leaves)
Num Leaves: 63
Feature Fraction: 0.8
Bagging Fraction: 0.8
```

### LSTM (Alternative)
```
Architecture: 2-layer LSTM
Hidden Size: 64
Sequence Length: 168 (1 week)
Dropout: 0.2
Output: Multi-quantile (p10, p50, p90)
```

---

## Evaluation

### Metrics
| Metric | Description | Value (LightGBM) |
|--------|-------------|------------------|
| MAPE | Mean Absolute Percentage Error | 3.2% |
| RMSE | Root Mean Squared Error | 1,250 MW |
| MAE | Mean Absolute Error | 980 MW |
| Pinball Loss (p10) | Quantile loss at 10th percentile | 320 |
| Pinball Loss (p50) | Quantile loss at 50th percentile | 490 |
| Pinball Loss (p90) | Quantile loss at 90th percentile | 380 |
| Coverage (80%) | Actual values within p10-p90 | 82% |

### Evaluation Methodology
- **Rolling-Origin Cross-Validation**: 3-5 folds
- **Test Period**: 1 week per fold
- **Gap**: 24 hours between training and test (prevent leakage)

### Performance by Condition
| Condition | MAPE (%) | Notes |
|-----------|----------|-------|
| Normal Days | 2.8 | Typical weekday/weekend |
| Holidays | 5.5 | Higher uncertainty |
| Extreme Weather | 4.2 | Hot summers, cold winters |
| Weather Transitions | 4.8 | Rapid temperature changes |

---

## Limitations

### Known Limitations
1. **Forecast Horizon**: Accuracy degrades beyond 48 hours
2. **Extreme Events**: May underestimate load during unprecedented weather
3. **Structural Changes**: Does not adapt to long-term demand shifts without retraining
4. **Regional Specificity**: Trained on PJM data; may not transfer to other regions
5. **Holiday Calendar**: Only includes US federal holidays

### Technical Limitations
- Requires minimum 4 weeks of historical data
- Temperature forecast dependency for weather features
- Model size: ~50MB (LightGBM), ~20MB (LSTM)
- Inference time: ~50ms per forecast

---

## Ethical Considerations

### Fairness
- The model does not use demographic or socioeconomic features
- Load forecasting is aggregated at regional level
- No individual household predictions

### Privacy
- No personally identifiable information (PII) used
- All data is aggregated grid-level load
- No individual consumption patterns

### Environmental Impact
- Accurate forecasts support renewable integration
- Reduces need for spinning reserves
- Training carbon footprint: Minimal (single GPU-hours)

---

## Risks and Mitigations

### Risk: Over-reliance on Forecasts
**Description**: Users may make critical decisions based solely on forecasts.
**Mitigation**:
- Always provide uncertainty quantiles (p10/p90)
- Clear documentation of limitations
- Recommend human oversight for critical decisions

### Risk: Model Drift
**Description**: Model performance may degrade over time.
**Mitigation**:
- Automated monitoring of prediction errors
- Regular retraining schedule (monthly)
- MLflow tracking for model versioning

### Risk: Adversarial Manipulation
**Description**: Input data could be manipulated to affect predictions.
**Mitigation**:
- Input validation and anomaly detection
- Audit logging of all predictions
- Rate limiting on API endpoints

### Risk: Cascading Failures
**Description**: Forecast errors could compound in downstream systems.
**Mitigation**:
- Graceful degradation (fallback to Seasonal Naive)
- Health checks and circuit breakers
- Clear error responses with fallback recommendations

---

## Deployment

### Requirements
- Python 3.10+
- 4GB RAM minimum
- No GPU required for inference

### Monitoring
- MLflow for experiment tracking
- Structured logging with timestamps
- Health check endpoint
- Prometheus metrics (optional)

### Update Frequency
- Model retraining: Monthly or on significant drift
- Feature updates: As needed with validation
- API versioning for breaking changes

---

## References

1. PJM Interconnection: https://www.pjm.com/
2. LightGBM: https://lightgbm.readthedocs.io/
3. SHAP: https://shap.readthedocs.io/
4. MLflow: https://mlflow.org/

---

## Contact

For questions or issues:
- GitHub Issues: https://github.com/example/electricity-demand-forecasting/issues
- Email: team@example.com

# HIV Disease Prediction System

A production-ready machine learning system for predicting advanced HIV disease using XGBoost, built with FastAPI and deployed on Kubernetes.

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting advanced HIV disease (CD4 count < 200 cells/µL) using patient clinical data. The system includes:

- **XGBoost Classification Model** with comprehensive feature engineering
- **FastAPI REST API** with async support and automatic documentation
- **Kubernetes Deployment** with horizontal pod autoscaling
- **Comprehensive Monitoring** with Prometheus and Grafana
- **Complete Test Suite** including load testing
- **Docker Containerization** for consistent deployments

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │   ML Pipeline   │    │   API Service   │
│                 │    │                 │    │                 │
│ • Data Gen      │───▶│ • Feature Eng   │───▶│ • FastAPI       │
│ • Preprocessing │    │ • XGBoost       │    │ • Validation    │
│ • Validation    │    │ • Evaluation    │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Deployment    │    │   Infrastructure│
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • Docker        │    │ • Kubernetes    │
│ • Grafana       │    │ • Kubernetes    │    │ • Load Balancer │
│ • Alerting      │    │ • CI/CD         │    │ • Auto-scaling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- uv (recommended) or pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/danielmaangi/ahd-e2d.git
   cd ahd-e2d
   ```

2. **Install uv (if not already installed)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

   **Alternative with pip:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the API server:**
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload
```

4. **Test the API:**
```bash
python tests/test_api.py
```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t hiv-prediction-api .
docker run -p 5000:5000 hiv-prediction-api

# Or use Docker Compose
docker-compose -f docker/docker-compose.yml up
```

### Kubernetes Deployment

```bash
# Deploy to Minikube
./deployment/minikube/deploy.sh

# Or deploy manually
kubectl apply -f kubernetes/
```

## 📊 Model Performance

The XGBoost model achieves the following performance metrics on the test set:

- **AUC Score**: 0.85+
- **Accuracy**: 80%+
- **Precision**: 78%+
- **Recall**: 82%+
- **F1 Score**: 80%+

### Feature Importance

Top predictive features:
1. Viral load (primary indicator)
2. ART adherence
3. Time since diagnosis
4. Hemoglobin level
5. Albumin level
6. Age
7. BMI
8. Gender
9. Comorbidities

## 🔌 API Usage

### Endpoints

- `POST /predict` - Single patient prediction
- `POST /predict-batch` - Batch predictions (up to 100 patients)
- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `GET /model-info` - Model metadata
- `GET /docs` - Interactive API documentation
- `GET /metrics` - Prometheus metrics

### Example Request

```python
import requests

patient_data = {
    "age": 45,
    "viral_load": 50000,
    "time_since_diagnosis": 5,
    "art_adherence": 0.6,
    "bmi": 22.5,
    "hemoglobin": 10.5,
    "albumin": 3.0,
    "gender": "M",
    "comorbidities": "Diabetes"
}

response = requests.post(
    "http://localhost:5000/predict",
    json=patient_data
)

print(response.json())
```

### Example Response

```json
{
    "prediction": 1,
    "probability": {
        "no_advanced_hiv": 0.25,
        "advanced_hiv": 0.75
    },
    "risk_level": "High",
    "confidence": 0.50,
    "model_version": "1.0.0",
    "timestamp": 1703875200.0
}
```

## 📁 Project Structure

```
hiv-disease-prediction/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model training and evaluation
│   ├── api/               # FastAPI application
│   └── utils/             # Utility functions
├── scripts/               # Standalone scripts
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── config/                # Configuration files
├── kubernetes/            # K8s deployment manifests
├── docker/                # Docker configurations
├── monitoring/            # Monitoring setup
└── deployment/            # Deployment scripts
```

## 🧪 Testing

Run the complete test suite:

```bash
# Unit tests
pytest tests/ -v

# API tests
python tests/test_api.py

# Load testing
python tests/performance_test.py

# Integration tests
python tests/integration_test.py
```

## 📈 Monitoring

### Metrics Available

- **API Metrics**: Request count, latency, error rates
- **Model Metrics**: Prediction distribution, confidence scores
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Prediction accuracy, model drift

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin123) to view:

- API Performance Dashboard
- Model Performance Dashboard
- Infrastructure Monitoring Dashboard

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_WORKERS=4

# Model Configuration
MODEL_PATH=models/
MODEL_VERSION=1.0.0

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO

# Security
ENABLE_CORS=true
RATE_LIMIT_PER_MINUTE=100
```

### Configuration Files

- `config/app_config.yaml` - Application settings
- `config/model_config.yaml` - Model training parameters
- `config/logging_config.yaml` - Logging configuration

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
docker build -f docker/Dockerfile.prod -t hiv-prediction-api:prod .

# Deploy to production Kubernetes cluster
./deployment/production/deploy.sh
```

### Scaling

The application supports horizontal scaling:

```bash
# Scale pods
kubectl scale deployment hiv-prediction-deployment --replicas=10

# Auto-scaling is configured via HPA
kubectl get hpa
```

## 🔒 Security

- Input validation with Pydantic models
- Rate limiting to prevent abuse
- Non-root container execution
- Kubernetes security contexts
- HTTPS/TLS termination at ingress

## 📚 Documentation

- [API Documentation](docs/api_documentation.md)
- [Model Documentation](docs/model_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. It should not be used for actual medical diagnosis or treatment decisions without proper validation and regulatory approval.

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the documentation

---

**Built with ❤️ for advancing HIV care through machine learning**

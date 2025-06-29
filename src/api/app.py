"""
HIV Disease Prediction - FastAPI Application

This module implements the main FastAPI application for serving HIV disease
prediction models. It provides REST API endpoints for single and batch predictions,
health checks, model information, and monitoring capabilities.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict

# Monitoring and metrics
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

# Local imports
from models.model_utils import HIVModelPredictor, ModelValidator, ModelPerformanceTracker
from utils.config import load_config
from utils.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
model_predictor: Optional[HIVModelPredictor] = None
performance_tracker: Optional[ModelPerformanceTracker] = None
app_config: Dict[str, Any] = {}
startup_time = time.time()


# Pydantic models for API
class PatientData(BaseModel):
    """Patient data model for HIV disease prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )
    
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    viral_load: float = Field(..., ge=0, description="HIV viral load (copies/mL)")
    time_since_diagnosis: float = Field(..., ge=0, description="Years since HIV diagnosis")
    art_adherence: float = Field(..., ge=0, le=1, description="ART adherence rate (0-1)")
    bmi: float = Field(..., ge=10, le=50, description="Body Mass Index")
    hemoglobin: float = Field(..., ge=5, le=20, description="Hemoglobin level (g/dL)")
    albumin: float = Field(..., ge=1, le=6, description="Albumin level (g/dL)")
    gender: str = Field(..., description="Patient gender (M/F/Male/Female)")
    comorbidities: str = Field(..., description="Comorbid conditions")
    
    @validator('gender')
    def validate_gender(cls, v):
        valid_genders = ['M', 'F', 'Male', 'Female']
        if v not in valid_genders:
            raise ValueError(f'Gender must be one of: {valid_genders}')
        return 'M' if v in ['M', 'Male'] else 'F'
    
    @validator('comorbidities')
    def validate_comorbidities(cls, v):
        valid_options = ['None', 'Diabetes', 'Hypertension', 'Both']
        if v not in valid_options:
            raise ValueError(f'Comorbidities must be one of: {valid_options}')
        return v


class PredictionResponse(BaseModel):
    """Response model for HIV disease predictions."""
    
    prediction: int = Field(..., description="Binary prediction (0: no advanced HIV, 1: advanced HIV)")
    probability: Dict[str, float] = Field(..., description="Prediction probabilities")
    risk_level: str = Field(..., description="Risk level classification (Low/Medium/High)")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    patients: List[PatientData] = Field(..., max_items=100, description="List of patient data")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of patients processed")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Model version")
    feature_count: int = Field(..., description="Number of features")
    training_date: Optional[str] = Field(None, description="Model training date")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting HIV Disease Prediction API...")
    await startup_event()
    
    yield
    
    # Shutdown
    logger.info("Shutting down HIV Disease Prediction API...")
    await shutdown_event()


# Create FastAPI app
app = FastAPI(
    title="HIV Disease Prediction API",
    description="Machine learning API for predicting advanced HIV disease using XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


async def startup_event():
    """Initialize application on startup."""
    global model_predictor, performance_tracker, app_config
    
    try:
        # Load configuration
        config_path = os.getenv('CONFIG_PATH', 'config/app_config.yaml')
        app_config = load_config(config_path)
        
        # Setup logging
        setup_logging()
        
        # Load model
        model_path = os.getenv('MODEL_PATH', 'models/')
        if not Path(model_path).exists():
            logger.error(f"Model directory not found: {model_path}")
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        model_predictor = HIVModelPredictor(model_path)
        logger.info("Model loaded successfully")
        
        # Initialize performance tracker
        log_file = os.getenv('PREDICTION_LOG_FILE', 'logs/predictions.log')
        performance_tracker = ModelPerformanceTracker(log_file)
        logger.info("Performance tracker initialized")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Application shutdown completed")


# Middleware setup
def setup_middleware():
    """Setup application middleware."""
    
    # CORS middleware
    if app_config.get('security', {}).get('cors', {}).get('enabled', True):
        cors_config = app_config.get('security', {}).get('cors', {})
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get('allow_origins', ["*"]),
            allow_credentials=cors_config.get('allow_credentials', True),
            allow_methods=cors_config.get('allow_methods', ["*"]),
            allow_headers=cors_config.get('allow_headers', ["*"]),
        )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )


# Dependency functions
async def get_model_predictor() -> HIVModelPredictor:
    """Dependency to get model predictor."""
    if model_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return model_predictor


async def get_performance_tracker() -> ModelPerformanceTracker:
    """Dependency to get performance tracker."""
    if performance_tracker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Performance tracker not initialized"
        )
    return performance_tracker


# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HIV Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    patient_data: PatientData,
    background_tasks: BackgroundTasks,
    predictor: HIVModelPredictor = Depends(get_model_predictor),
    tracker: ModelPerformanceTracker = Depends(get_performance_tracker)
):
    """
    Make a single HIV disease prediction.
    
    This endpoint accepts patient clinical data and returns a prediction
    for advanced HIV disease along with probability scores and risk assessment.
    """
    start_time = time.time()
    
    try:
        # Validate input data
        is_valid, errors = ModelValidator.validate_patient_data(patient_data.dict())
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input data: {'; '.join(errors)}"
            )
        
        # Make prediction
        result = predictor.predict(patient_data.dict())
        
        # Validate result
        is_valid, errors = ModelValidator.validate_prediction_result(result)
        if not is_valid:
            logger.error(f"Invalid prediction result: {errors}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid prediction result"
            )
        
        # Log prediction in background
        prediction_time = time.time() - start_time
        background_tasks.add_task(
            log_prediction_async,
            tracker,
            patient_data.dict(),
            result,
            prediction_time
        )
        
        logger.info(f"Prediction completed in {prediction_time:.3f}s: {result['risk_level']} risk")
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: HIVModelPredictor = Depends(get_model_predictor),
    tracker: ModelPerformanceTracker = Depends(get_performance_tracker)
):
    """
    Make batch HIV disease predictions.
    
    This endpoint accepts multiple patient records and returns predictions
    for each patient. Maximum batch size is 100 patients.
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        max_batch_size = app_config.get('model', {}).get('prediction', {}).get('batch_size_limit', 100)
        if len(request.patients) > max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {len(request.patients)} exceeds maximum {max_batch_size}"
            )
        
        # Convert to list of dictionaries
        patient_data_list = [patient.dict() for patient in request.patients]
        
        # Make batch predictions
        results = predictor.predict_batch(patient_data_list, max_batch_size)
        
        processing_time = time.time() - start_time
        
        # Log batch prediction in background
        background_tasks.add_task(
            log_batch_prediction_async,
            tracker,
            patient_data_list,
            results,
            processing_time
        )
        
        logger.info(f"Batch prediction completed: {len(results)} patients in {processing_time:.3f}s")
        
        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(results),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API service including uptime,
    model status, and basic system information.
    """
    current_time = time.time()
    uptime = current_time - startup_time
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time,
        version="1.0.0",
        uptime_seconds=uptime,
        model_loaded=model_predictor is not None and model_predictor.is_loaded
    )


@app.get("/ready")
async def readiness_check(predictor: HIVModelPredictor = Depends(get_model_predictor)):
    """
    Readiness check endpoint.
    
    Verifies that the service is ready to handle requests by testing
    the model with a sample prediction.
    """
    try:
        # Test prediction with sample data
        from models.model_utils import create_sample_input
        sample_data = create_sample_input()
        
        # Make test prediction
        result = predictor.predict(sample_data)
        
        # Validate result
        is_valid, errors = ModelValidator.validate_prediction_result(result)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model validation failed: {errors}"
            )
        
        return {"status": "ready", "test_prediction": "successful"}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(predictor: HIVModelPredictor = Depends(get_model_predictor)):
    """
    Get model information and metadata.
    
    Returns detailed information about the loaded model including
    version, features, training date, and performance metrics.
    """
    try:
        model_info = predictor.get_model_info()
        
        return ModelInfoResponse(
            model_type=model_info.get('model_type', 'Unknown'),
            model_version=model_info.get('model_version', '1.0.0'),
            feature_count=model_info.get('feature_count', 0),
            training_date=model_info.get('training_date'),
            performance_metrics=model_info.get('evaluation_results', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/performance")
async def get_performance_summary(
    days: int = 7,
    tracker: ModelPerformanceTracker = Depends(get_performance_tracker)
):
    """
    Get model performance summary.
    
    Returns performance statistics for the specified number of days,
    including prediction counts, accuracy metrics, and distribution analysis.
    """
    try:
        summary = tracker.get_performance_summary(days)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance summary: {str(e)}"
        )


# Background task functions
async def log_prediction_async(
    tracker: ModelPerformanceTracker,
    input_data: Dict[str, Any],
    result: Dict[str, Any],
    prediction_time: float
):
    """Log prediction asynchronously."""
    try:
        tracker.log_prediction(input_data, result, prediction_time=prediction_time)
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


async def log_batch_prediction_async(
    tracker: ModelPerformanceTracker,
    input_data_list: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    processing_time: float
):
    """Log batch prediction asynchronously."""
    try:
        for input_data, result in zip(input_data_list, results):
            if 'error' not in result:
                tracker.log_prediction(
                    input_data, 
                    result, 
                    prediction_time=processing_time / len(results)
                )
    except Exception as e:
        logger.error(f"Failed to log batch prediction: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


# Setup middleware
setup_middleware()

# Setup Prometheus metrics
if app_config.get('monitoring', {}).get('metrics', {}).get('enabled', True):
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="HIV Disease Prediction API",
        version="1.0.0",
        description="""
        ## HIV Disease Prediction API
        
        This API provides machine learning-based predictions for advanced HIV disease
        using patient clinical data. The model uses XGBoost to predict the likelihood
        of advanced HIV disease (CD4 count < 200 cells/μL).
        
        ### Features
        - Single patient predictions
        - Batch processing (up to 100 patients)
        - Risk level assessment (Low/Medium/High)
        - Model performance monitoring
        - Comprehensive health checks
        
        ### Usage
        1. Use `/predict` for single patient predictions
        2. Use `/predict-batch` for multiple patients
        3. Check `/health` for service status
        4. View `/model-info` for model details
        
        ### Authentication
        Currently no authentication is required for this demo API.
        In production, implement appropriate authentication mechanisms.
        """,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "predictions",
            "description": "HIV disease prediction endpoints"
        },
        {
            "name": "health",
            "description": "Health and status check endpoints"
        },
        {
            "name": "model",
            "description": "Model information and performance endpoints"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Main function for running the app
def main():
    """Main function to run the FastAPI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run HIV Disease Prediction API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Log level')
    
    args = parser.parse_args()
    
    # Configure uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()

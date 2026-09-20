"""Pydantic request/response models for the /predict endpoint."""

from pydantic import BaseModel, Field


class PolicyFeatures(BaseModel):
    Exposure: float = Field(..., gt=0, le=1, description="Fraction of the year at risk")
    VehPower: int = Field(..., gt=0)
    VehAge: int = Field(..., ge=0)
    DrivAge: int = Field(..., ge=18)
    BonusMalus: int = Field(..., ge=50, le=350)
    VehBrand: str
    VehGas: str
    Density: float = Field(..., gt=0)
    Region: str
    Area: str


class ShapContribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    probability_of_claim: float
    expected_severity: float
    expected_loss: float
    top_shap_frequency: list[ShapContribution]
    top_shap_severity: list[ShapContribution]

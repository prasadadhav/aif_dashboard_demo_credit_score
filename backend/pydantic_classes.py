from datetime import datetime, date, time
from typing import List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, field_validator

from abc import ABC, abstractmethod

############################################
# Enumerations are defined here
############################################

class DatasetType(Enum):
    Validation = "Validation"
    Training = "Training"
    Test = "Test"

class EvaluationStatus(Enum):
    Processing = "Processing"
    Done = "Done"
    Pending = "Pending"
    Archived = "Archived"
    Custom = "Custom"

class ProjectStatus(Enum):
    Ready = "Ready"
    Archived = "Archived"
    Closed = "Closed"
    Pending = "Pending"
    Created = "Created"

class LicensingType(Enum):
    Proprietary = "Proprietary"
    Open_Source = "Open_Source"

############################################
# Classes are defined here
############################################
class DatashapeCreate(BaseModel):
    accepted_target_values: str
    f_features: Optional[List[int]] = None  # 1:N Relationship
    f_date: Optional[List[int]] = None  # 1:N Relationship
    dataset_1: Optional[List[int]] = None  # 1:N Relationship

class ProjectCreate(BaseModel):
    status: ProjectStatus
    name: str
    legal_requirements: Optional[List[int]] = None  # 1:N Relationship
    involves: Optional[List[int]] = None  # 1:N Relationship
    eval: Optional[List[int]] = None  # 1:N Relationship

class EvaluationCreate(BaseModel):
    status: EvaluationStatus
    config: int  # N:1 Relationship (mandatory)
    ref: List[int]  # N:M Relationship
    project: int  # N:1 Relationship (mandatory)
    observations: Optional[List[int]] = None  # 1:N Relationship
    evaluates: List[int]  # N:M Relationship

class MeasureCreate(BaseModel):
    error: str
    unit: str
    value: float
    uncertainty: float
    observation: int  # N:1 Relationship (mandatory)
    metric: int  # N:1 Relationship (mandatory)
    measurand: int  # N:1 Relationship (mandatory)

class AssessmentElementCreate(ABC, BaseModel):
    description: str
    name: str

class ObservationCreate(AssessmentElementCreate):
    observer: str
    whenObserved: datetime
    eval: int  # N:1 Relationship (mandatory)
    measures: Optional[List[int]] = None  # 1:N Relationship
    tool: int  # N:1 Relationship (mandatory)
    dataset: int  # N:1 Relationship (mandatory)

class ElementCreate(AssessmentElementCreate):
    evalu: List[int]  # N:M Relationship
    project: Optional[int] = None  # N:1 Relationship (optional)
    measure: Optional[List[int]] = None  # 1:N Relationship
    eval: List[int]  # N:M Relationship

class ModelCreate(ElementCreate):
    data: str
    source: str
    licensing: LicensingType
    pid: str
    dataset: int  # N:1 Relationship (mandatory)

class DatasetCreate(ElementCreate):
    licensing: LicensingType
    version: str
    source: str
    dataset_type: DatasetType
    models: Optional[List[int]] = None  # 1:N Relationship
    observation_2: Optional[List[int]] = None  # 1:N Relationship
    datashape: int  # N:1 Relationship (mandatory)

class FeatureCreate(ElementCreate):
    min_value: float
    feature_type: str
    max_value: float
    features: int  # N:1 Relationship (mandatory)
    date: int  # N:1 Relationship (mandatory)

class MetricCreate(AssessmentElementCreate):
    measures: Optional[List[int]] = None  # 1:N Relationship
    derivedBy: List[int]  # N:M Relationship
    category: List[int]  # N:M Relationship

class DerivedCreate(MetricCreate):
    expression: str
    baseMetric: List[int]  # N:M Relationship

class DirectCreate(MetricCreate):
    pass

class CommentsCreate(BaseModel):
    Comments: str
    TimeStamp: datetime
    Name: str

class MetricCategoryCreate(AssessmentElementCreate):
    metrics: List[int]  # N:M Relationship

class LegalRequirementCreate(BaseModel):
    standard: str
    principle: str
    legal_ref: str
    project_1: int  # N:1 Relationship (mandatory)

class ToolCreate(BaseModel):
    source: str
    licensing: LicensingType
    version: str
    name: str
    observation_1: Optional[List[int]] = None  # 1:N Relationship

class ConfParamCreate(AssessmentElementCreate):
    param_type: str
    value: str
    conf: int  # N:1 Relationship (mandatory)

class ConfigurationCreate(AssessmentElementCreate):
    eval: Optional[List[int]] = None  # 1:N Relationship
    params: Optional[List[int]] = None  # 1:N Relationship


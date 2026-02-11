from pydantic import BaseModel, Field, computed_field
from enum import Enum

class AgeGroup(str, Enum):
    CHILD = 'child'
    TEEN = 'teen'
    ADULT = 'adult'
    ELDERLY = 'elderly'

class Disability(str, Enum):
    NONE = 'none'
    WHEELCHAIR = 'wheelchair'
    CANE = 'cane'
    BLIND = 'blind'

class SocialRole(str, Enum):
    CIVILIAN = 'civilian'
    EMERGENCY = 'emergency'
    HEALTHCARE = 'healthcare'
    
class EthicalAttributeSchema(BaseModel):
    age_group: AgeGroup
    disability: Disability
    pregnancy: bool
    group_size: int = Field(ge=1, le=5)
    social_role: SocialRole

    @computed_field
    @property
    def vulnerability_score(self) -> float:
        base = 0.0
        if self.age_group == AgeGroup.CHILD: base += 0.3
        if self.age_group == AgeGroup.ELDERLY: base += 0.2
        if self.disability != Disability.NONE: base += 0.25
        if self.pregnancy: base += 0.25
        return min(1.0, base)

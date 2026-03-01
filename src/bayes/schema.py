from enum import Enum
from typing import Literal

from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.base import Parameter
from baybe.targets import NumericalTarget
from baybe.utils.random import set_random_seed
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VariableType(str, Enum):
    """变量类型枚举"""

    CONTINUOUS = "Continuous"
    DISCRETE = "Discrete"
    CATEGORICAL = "Categorical"
    # SUBSTANCE = "Substance"


class ObjectiveType(str, Enum):
    """优化目标类型"""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class SurrogateModel(str, Enum):
    """代理模型类型"""

    GP = "GP"
    RF = "RF"


class Variable(BaseModel):
    """优化变量定义"""

    name: str = Field(..., description="变量名称")
    param_type: VariableType = Field(..., description="变量类型")
    min: float | None = Field(None, description="最小值")
    max: float | None = Field(None, description="最大值")
    chooses: list[str | int | float] = Field(
        description="分类选项", default_factory=list
    )

    @model_validator(mode="after")
    def validate_variable(self):
        """综合验证变量定义"""
        if self.param_type == VariableType.CONTINUOUS:
            if self.min is None or self.max is None:
                raise ValueError(f"连续变量 {self.name} 必须有最小值和最大值")
            if self.min >= self.max:
                raise ValueError(f"连续变量 {self.name} 的最小值必须小于最大值")

        elif self.param_type in [VariableType.DISCRETE, VariableType.CATEGORICAL]:
            if not self.chooses or len(self.chooses) < 2:
                raise ValueError(f"离散变量 {self.name} 至少需要2个选项")

        return self

    model_config = ConfigDict(use_enum_values=True)

    def to_baybe_param(self) -> Parameter:
        if self.param_type == VariableType.CONTINUOUS:
            return NumericalContinuousParameter(
                name=self.name, bounds=(self.min, self.max)
            )
        elif self.param_type == VariableType.DISCRETE:
            return NumericalDiscreteParameter(
                name=self.name,
                values=tuple(self.chooses),  # type: ignore
            )
        elif self.param_type == VariableType.CATEGORICAL:
            return CategoricalParameter(
                name=self.name,
                values=tuple(self.chooses),  # type: ignore
            )
        else:
            raise NotImplementedError(f"暂不支持的变量类型: {self.param_type}")


class Objective(BaseModel):
    """优化目标定义"""

    name: str = Field(..., description="目标名称（如产率、选择性）")
    target_type: ObjectiveType = Field(..., description="优化类型")
    target_value: float | None = Field(None, description="目标值")
    weight: float = Field(1.0, description="多目标优化中的权重", ge=0, le=1)
    unit: str | None = Field(None, description="单位")

    @model_validator(mode="after")
    def validate_target(self):
        """验证目标值"""
        if self.target_type == ObjectiveType.TARGET and self.target_value is None:
            raise ValueError("目标值优化必须指定目标值")

        return self

    model_config = ConfigDict(use_enum_values=True)

    def to_baybe_target(self) -> NumericalTarget:
        if self.target_type == ObjectiveType.MAXIMIZE:
            return NumericalTarget(name=self.name, minimize=False)
        elif self.target_type == ObjectiveType.MINIMIZE:
            return NumericalTarget(name=self.name, minimize=True)
        elif self.target_type == ObjectiveType.TARGET:
            return NumericalTarget.match_absolute(
                name=self.name,
                match_value=self.target_value,  # type: ignore
            )
        else:
            raise NotImplementedError(f"暂不支持的优化目标类型: {self.target_type}")


class BOConfig(BaseModel):
    """贝叶斯优化配置"""

    n_iterations: int = Field(50, description="优化迭代次数", ge=1)
    n_initial_points: int = Field(5, description="初始随机探索点数", ge=1)
    surrogate_model: SurrogateModel = Field(SurrogateModel.GP, description="代理模型")
    n_parallel: int = Field(1, description="并行实验数", ge=1)
    random_state: int | None = Field(None, description="随机种子")

    def set_random_seed(self):
        randomseed = self.random_state
        if randomseed is not None:
            set_random_seed(randomseed)


class ProjectModel(BaseModel):
    """实验设置"""

    name: str = Field(..., description="实验名称")
    description: str | None = Field(None, description="实验描述")
    variables: list[Variable] = Field(..., description="优化变量列表", min_length=1)
    objectives: list[Objective] = Field(..., description="优化目标列表", min_length=1)
    bo_config: BOConfig = Field(..., description="贝叶斯优化配置")


class ExperimentModel(BaseModel):
    """单次实验结果"""

    id: int | None = Field(None, description="实验ID")
    experiment_id: int = Field(..., description="所属实验设置ID")
    iteration: int = Field(..., description="迭代轮数")
    variables: dict[str, int | float | str] = Field(..., description="变量实际值")
    metrics: dict[str, float] = Field(..., description="实验结果指标（产率、选择性等）")

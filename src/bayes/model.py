from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Mapped
from sqlmodel import JSON, Column, Field, Relationship, SQLModel, create_engine

from .schema import BOConfig, ExperimentModel, Objective, ProjectModel, Variable


class Project(SQLModel, table=True):
    """实验设置表"""

    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str | None = Field(default=None)
    variables: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
    objectives: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
    bo_config: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=datetime.now)

    experiments: Mapped[list["Experiment"]] = Relationship(back_populates="project")

    class Config:
        from_attributes = True


class Experiment(SQLModel, table=True):
    """实验结果表"""

    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="project.id", index=True)
    variables: dict[str, int | float | str] = Field(
        sa_column=Column(JSON, nullable=False)
    )
    metrics: dict[str, float] = Field(sa_column=Column(JSON, nullable=False))
    iteration: int | None = Field(default=None, index=True)
    timestamp: datetime = Field(default_factory=datetime.now)

    project: Project = Relationship(back_populates="experiments")

    class Config:
        from_attributes = True


def project_from_pydantic(exp_setup: ProjectModel) -> Project:
    """将Pydantic ProjectModel转换为数据库模型"""
    return Project(
        name=exp_setup.name,
        description=exp_setup.description,
        variables=[v.model_dump() for v in exp_setup.variables],
        objectives=[o.model_dump() for o in exp_setup.objectives],
        bo_config=exp_setup.bo_config.model_dump(),
    )


def project_to_pydantic(db_exp: Project) -> ProjectModel:
    """将数据库模型转换回Pydantic模型"""

    return ProjectModel(
        name=db_exp.name,
        description=db_exp.description,
        variables=[Variable.model_validate(v) for v in db_exp.variables],
        objectives=[Objective.model_validate(o) for o in db_exp.objectives],
        bo_config=BOConfig.model_validate(db_exp.bo_config),
    )


def experiment_from_pydantic(result: ExperimentModel) -> Experiment:
    """将Pydantic ExperimentModel转换为数据库模型"""
    return Experiment(
        experiment_id=result.experiment_id,
        iteration=result.iteration,
        variables=result.variables,
        metrics=result.metrics,
    )


def experiment_to_pydantic(db_result: Experiment) -> ExperimentModel:
    """将数据库模型转换回Pydantic ExperimentModel"""
    return ExperimentModel(
        id=db_result.id,
        experiment_id=db_result.experiment_id,
        iteration=db_result.iteration,  # type: ignore
        variables=db_result.variables,
        metrics=db_result.metrics,
    )


DATABASE = Path(__file__).parent.parent.parent / "data" / "experiments.db"
DATABASE_URL = f"sqlite:///{DATABASE}"
engine = create_engine(DATABASE_URL, echo=False)


def creat_db_and_tables(engine):
    """初始化数据库表"""
    SQLModel.metadata.create_all(engine)


creat_db_and_tables(engine)

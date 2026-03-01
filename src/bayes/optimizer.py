from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from sqlmodel import Session

from bayes.model import Project, engine

from .model import (
    experiment_from_pydantic,
    experiment_to_pydantic,
    project_from_pydantic,
    project_to_pydantic,
)
from .schema import BOConfig, ExperimentModel, Objective, ProjectModel, Variable


class ExperimentOptimizer:
    @staticmethod
    def setup(
        name: str,
        description: str,
        variables: list[dict[str, Any]],
        objectives: list[dict[str, Any]],
        bo_config: dict[str, Any],
    ) -> int | None:
        """创建实验项目"""
        expt_setup = ProjectModel(
            name=name,
            description=description,
            variables=[Variable.model_validate(v) for v in variables],
            objectives=[Objective.model_validate(o) for o in objectives],
            bo_config=BOConfig.model_validate(bo_config),
        )
        with Session(engine) as session:
            expt = project_from_pydantic(expt_setup)
            session.add(expt)
            session.commit()
            return expt.id

    @staticmethod
    def read_project(id: int) -> ProjectModel:
        """读取项目设置"""
        with Session(engine) as session:
            project = session.get(Project, id)
            if project is None:
                raise ValueError("项目不存在")

            return project_to_pydantic(project)

    @staticmethod
    def read_experiments(id: int) -> pd.DataFrame:
        """读取项目实验结果"""
        with Session(engine) as session:
            project = session.get(Project, id)
            if project is None:
                raise ValueError("项目不存在")

            experiments = [
                experiment_to_pydantic(result) for result in project.experiments
            ]

            condition_df = pd.DataFrame([e.variables for e in experiments])
            result_df = pd.DataFrame([e.metrics for e in experiments])

            return pd.concat([condition_df, result_df], axis=1)

    @staticmethod
    def tell(
        id: int,
        iteration: int,
        conditions: list[dict[str, int | float | str]],
        results: list[dict[str, int | float]],
    ):
        """写入项目实验结果"""
        with Session(engine) as session:
            project = session.get(Project, id)
            if project is None:
                raise ValueError("项目不存在")
            else:
                for condition, result in zip(conditions, results):
                    experiment = experiment_from_pydantic(
                        ExperimentModel(
                            id=id,
                            experiment_id=id,
                            iteration=iteration,
                            variables=condition,
                            metrics=result,
                        )
                    )
                    project.experiments.append(experiment)
                    session.add(experiment)

            session.add(project)
            session.commit()

    @staticmethod
    def ask(id: int, batch_size: int) -> list[dict[str, int | float | str]]:
        """推荐实验条件"""
        project = ExperimentOptimizer.read_project(id)
        project.bo_config.set_random_seed()

        params = [param.to_baybe_param() for param in project.variables]
        targets = [
            target.to_baybe_target().to_objective() for target in project.objectives
        ]
        if len(targets) > 1:
            raise NotImplementedError("目前仅支持单目标优化")

        searchspace = SearchSpace.from_product(parameters=params)

        recommender = TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(),
            switch_after=project.bo_config.n_initial_points,
        )
        recommendation = recommender.recommend(
            batch_size,
            searchspace,
            targets[0],
            ExperimentOptimizer.read_experiments(id),
        ).to_dict(orient="records")

        return recommendation  # type: ignore


def current_max(values: list[float]) -> list[float]:
    if len(values) < 2:
        return values

    max_value = values[0]
    current_max_values = [max_value]
    for v in values[1:]:
        if v > max_value:
            max_value = v

        current_max_values.append(max_value)

    return current_max_values


def plot(expt_id: int):
    with Session(engine) as session:
        project = session.get(Project, expt_id)
        if project is None:
            raise ValueError(f"Experiment {expt_id} not exists")

        expts = project.experiments
        results = pd.DataFrame([e.metrics for e in expts])
        batch_ids = [e.iteration for e in expts]

        fig, ax = plt.subplots()
        x = list(range(1, len(expts) + 1))
        for col in results.columns:
            y = current_max(results[col].to_list())
            ax.plot(x, y)

            y = np.random.rand(10)
            colors = np.arange(max(batch_ids))  # type: ignore
            ax.scatter(
                x,
                results[col].to_list(),
                c=[colors[i - 1] for i in batch_ids],  # type: ignore
                cmap="viridis",
                label=col,
            )

        plt.xlabel("Experiment ID")
        plt.ylabel("Targets")
        plt.legend()

    return fig

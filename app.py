import pandas as pd
import streamlit as st

from bayes.component import init_variavles, render_variables_section
from bayes.optimizer import ExperimentOptimizer, plot


def init_session_session_state():
    init_variavles()

    if "expt_id" not in st.session_state:
        st.session_state.expt_id = None

    if "iteration" not in st.session_state:
        st.session_state.iteration = 1

    if "conditions" not in st.session_state:
        st.session_state.conditions = pd.DataFrame()

    if "results" not in st.session_state:
        st.session_state.results = []

    if "bo_config" not in st.session_state:
        st.session_state.bo_config = {
            "n_iterations": 10,
            "n_initial_points": 6,
            "surrogate_model": "GP",
        }


def start():
    if st.session_state.variables.values() and st.session_state.targets.values():
        st.session_state.expt_id = ExperimentOptimizer.setup(
            name=st.session_state.name,
            description=st.session_state.description,
            variables=list(st.session_state.variables.values()),
            objectives=list(st.session_state.targets.values()),
            bo_config=st.session_state.bo_config,
        )

        st.session_state.iteration = 1
        return st.session_state.expt_id
    else:
        st.error("请设置优化变量和目标")


def step(name, description, variables, targets):
    pass


init_session_session_state()

with st.sidebar:
    st.header("参数设定")
    st.session_state.name = st.text_input("实验名称", value="化学反应1")
    st.session_state.description = st.text_area("实验描述", value="化学反应1条件优化")
    st.session_state.batch_size = st.slider(
        "每轮推荐反应条件数目", min_value=1, max_value=10, value=3
    )
    st.session_state.epoch_num = st.slider(
        "优化轮数", min_value=5, max_value=100, value=10
    )
    st.session_state.model_option = st.selectbox("代理模型", ["高斯过程", "随机森林"])
    st.session_state.acq_func = st.selectbox("采集函数", ["EI", "LCB", "PI"])


st.title("工艺条件贝叶斯优化")
render_variables_section()

st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("开始", type="primary"):
        expt_id = start()
        st.info(f"新建实验 #{expt_id}: {st.session_state.name}")

with col2:
    if st.button("推荐"):
        conditions = ExperimentOptimizer.ask(
            st.session_state.expt_id, st.session_state.batch_size
        )
        condition_df = pd.DataFrame(conditions)
        for k in st.session_state.targets.keys():
            condition_df[k] = None

        st.session_state.conditions = condition_df

st.subheader(f"第{st.session_state.iteration}轮实验条件推荐")
table_edit = st.data_editor(st.session_state.conditions, width="stretch")


if st.button("提交") and not table_edit.empty:
    result_df = table_edit[list(st.session_state.targets.keys())].astype(float)
    if result_df.isna().sum().sum() > 0:
        st.error("请填写所有实验结果")
    else:
        results = result_df.to_dict(orient="records")
        variables = table_edit[list(st.session_state.variables.keys())].to_dict(
            orient="records"
        )

        ExperimentOptimizer.tell(
            st.session_state.expt_id,
            st.session_state.iteration,
            variables,  # type: ignore
            results,  # type: ignore
        )
        st.session_state.iteration += 1
        st.info(f"新增{len(variables)}条实验数据")

        st.subheader("实验条件优化结果")
        fig = plot(st.session_state.expt_id)
        st.pyplot(fig)  # type: ignore

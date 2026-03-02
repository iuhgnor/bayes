import pandas as pd
import streamlit as st

from bayes.demo import carry_experiments, objectives, toc_img, variables
from bayes.optimizer import ExperimentOptimizer, plot


def init_session_session_state():
    st.session_state.name = "直接芳基化反应条件优化"
    st.session_state.description = (
        "从12种催化剂配体、4种碱、4种溶剂、3个底物浓度、3个温度"
        "组成的参数空间种搜索产率最高的反应条件组合。"
    )

    if "expt_id" not in st.session_state:
        st.session_state.expt_id = None

    if "iteration" not in st.session_state:
        st.session_state.iteration = 1

    if "conditions" not in st.session_state:
        st.session_state.conditions = []

    if "results" not in st.session_state:
        st.session_state.results = []

    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame()

    if "bo_config" not in st.session_state:
        st.session_state.bo_config = {
            "n_iterations": 10,
            "n_initial_points": 6,
            "surrogate_model": "GP",
        }


def start():
    st.session_state.expt_id = ExperimentOptimizer.setup(
        name=st.session_state.name,
        description=st.session_state.description,
        variables=variables,  # type: ignore
        objectives=objectives,
        bo_config=st.session_state.bo_config,
    )

    st.session_state.iteration = 1
    return st.session_state.expt_id


init_session_session_state()

with st.sidebar:
    st.header("参数设定")
    st.session_state.name = st.text_input(
        "实验名称", value=st.session_state.name, disabled=True
    )
    st.session_state.description = st.text_area(
        "实验描述", value=st.session_state.description, disabled=True
    )
    st.session_state.batch_size = st.slider(
        "每轮推荐反应条件数目", min_value=1, max_value=10, value=3
    )
    st.session_state.epoch_num = st.slider(
        "优化轮数", min_value=5, max_value=100, value=10
    )
    st.session_state.model_option = st.selectbox("代理模型", ["高斯过程", "随机森林"])
    st.session_state.acq_func = st.selectbox("采集函数", ["EI", "LCB", "PI"])


st.title("工艺条件贝叶斯优化演示案例")
st.image(
    toc_img,
    caption="Nature, 2021, 590, 89–96, https://doi.org/10.1038/s41586-021-03213-y",
)

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("开始", type="primary"):
        expt_id = start()
        st.info(f"新建实验 #{expt_id}: {st.session_state.name}")

with col2:
    if st.button("推荐"):
        st.session_state.conditions = ExperimentOptimizer.ask(
            st.session_state.expt_id, st.session_state.batch_size
        )
        st.session_state.results = carry_experiments(st.session_state.conditions)

st.subheader(f"第{st.session_state.iteration}轮实验条件推荐和实验结果")
current_result_df = pd.concat(
    [
        pd.DataFrame(st.session_state.conditions),
        pd.DataFrame(st.session_state.results),
    ],
    axis=1,
)
st.session_state.history = pd.concat([st.session_state.history, current_result_df])
if not st.session_state.history.empty:
    st.session_state.history = st.session_state.history.sort_values(
        by="Yield", ascending=False
    ).reset_index(drop=True)

table_edit = st.data_editor(current_result_df, width="stretch")

with col3:
    submit = st.button("提交")


if submit and not table_edit.empty:
    ExperimentOptimizer.tell(
        st.session_state.expt_id,
        st.session_state.iteration,
        st.session_state.conditions,
        st.session_state.results,
    )
    st.session_state.iteration += 1
    st.info(f"新增{st.session_state.batch_size}条实验数据")

st.markdown("---")
st.subheader("实验条件优化结果")
if st.session_state.expt_id:
    fig = plot(st.session_state.expt_id)
    st.pyplot(fig)  # type: ignore

    st.subheader("Top-10 实验条件")
    st.data_editor(
        st.session_state.history, width="stretch", key="history_results"
    ).head(10)

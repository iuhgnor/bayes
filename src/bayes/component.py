import pandas as pd
import streamlit as st


def data_editor(data, key: str, editable: bool = True, **kwargs) -> pd.DataFrame:
    return st.data_editor(pd.DataFrame(data), key=key, disabled=not editable, **kwargs)


def init_variavles():
    if "variables" not in st.session_state:
        st.session_state.variables = {}

    if "targets" not in st.session_state:
        st.session_state.targets = {}


def show_variables():
    variables = []
    for k, v in st.session_state.variables.items():
        if v["param_type"] == "Continuous":
            variables.append(f"{k}: {v['min']} - {v['max']}")
        else:
            chooses = ", ".join([str(i) for i in v["chooses"]])
            variables.append(f"{k}: {chooses}")
    return "\n".join(variables)


def show_targets():
    variables = []
    for k, v in st.session_state.targets.items():
        if v["target_type"] == "target":
            variables.append(f"{k}: {v['target_value']}")
        else:
            variables.append(
                f"{k}: {'最大化' if v['target_type'] == 'maximize' else '最小化'}"
            )
    return "\n".join(variables)


def render_variables_section() -> None:
    st.subheader("设置实验条件")

    col1, _ = st.columns([1, 3])
    with col1:
        st.session_state.var_type = st.selectbox(
            "选择实验条件变量类型", ["类别", "连续", "离散"], key="var_type_select"
        )
    with st.form("manual_var_form"):
        col1, col2 = st.columns([1, 3])
        with col1:
            var_name = st.text_input("名称")
        with col2:
            if st.session_state.var_type == "连续":
                col21, col22 = st.columns([1, 1])
                with col21:
                    lower = st.number_input("下限", value=0.0)
                with col22:
                    upper = st.number_input("上限", value=1.0)
            elif st.session_state.var_type == "离散":
                categories = st.text_input("选项 (逗号分隔)", value="1.0, 1.1, 1.2")
                values = [float(x.strip()) for x in categories.split(",") if x.strip()]
            else:
                categories = st.text_input("选项 (逗号分隔)", value="A, B, C, D")
                values = [x.strip() for x in categories.split(",") if x.strip()]

        st.text_area(label="当前实验条件变量", value=show_variables())

        add_var = st.form_submit_button("增加变量")
        if add_var and var_name:
            if st.session_state.var_type == "连续" and lower < upper:  # type: ignore
                var = {
                    "name": var_name,
                    "param_type": "Continuous",
                    "min": lower,
                    "max": upper,
                }

            elif st.session_state.var_type == "离散" and categories:
                var = {
                    "name": var_name,
                    "param_type": "Discrete",
                    "chooses": list(set(values)),
                }

            else:
                var = {
                    "name": var_name,
                    "param_type": "Categorical",
                    "chooses": list(set(values)),
                }

            if var_name not in st.session_state.variables:
                st.info(f"添加实验条件变量: {var_name}")
            else:
                st.info(f"修改实验条件变量: {var_name}")
            st.session_state.variables[var_name] = var
            st.rerun()

    st.subheader("设置优化目标")
    col1, _ = st.columns([2, 3])
    with col1:
        st.session_state.var_type = st.selectbox(
            "选择优化目标类型", ["最大化", "最小化", "目标值"], key="target_type_select"
        )
    with st.form("manual_target_form"):
        col1, col2 = st.columns([2, 3])
        with col1:
            var_name = st.text_input("名称 (不能与条件变量重名)")
        if st.session_state.var_type == "目标值":
            with col2:
                target_value = st.number_input("目标值", value=1)
                target = {
                    "name": var_name,
                    "target_type": "target",
                    "target_value": target_value,
                }
        else:
            target = {
                "name": var_name,
                "target_type": "maximize"
                if st.session_state.var_type == "最大化"
                else "minimize",
            }

        st.text_area(label="当前优化目标", value=show_targets())

        add_target = st.form_submit_button("增加变量")
        if add_target and var_name:
            if var_name not in st.session_state.targets:
                st.info(f"添加实验优化目标: {var_name}")
            else:
                st.info(f"修改实验优化目标: {var_name}")
            st.session_state.targets[var_name] = target
            st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler 

# =================================================================
# 0. 초기 설정 및 상수
# =================================================================
st.set_page_config(layout="wide", page_title="Weld Line 통합 진단 시스템")

# 기본 경계 조건
GLOBAL_BOUNDS = {
    'T_Melt': (200, 300), 'V_Inj': (1, 10), 'P_Pack': (50, 100), 
    'T_Mold': (30, 80), 'Meter': (180, 200), 'VP_Switch_Pos': (10, 20)
}

TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5

# 세션 상태 초기화 (한 번만 실행)
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['model'] = None
    st.session_state['df_weld'] = pd.DataFrame()
    st.session_state['global_process_vars'] = []
    st.session_state['ui_display_vars'] = []
    st.session_state['global_bounds'] = GLOBAL_BOUNDS.copy()
    st.session_state['conf_level'] = 75.0
    st.session_state['influence_factor_display_val'] = 0.75
    st.session_state['current_risk_display'] = None
    st.session_state['optimization_result'] = None
    st.session_state['knowhow_settings'] = {}
    st.session_state['selected_knowhow_vars'] = []
    st.session_state['default_init_values'] = {}

# =================================================================
# 1. 주요 함수 (데이터 및 모델링)
# =================================================================
def process_weld_data(df_virtual, df_real):
    valid_dfs = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    if not valid_dfs: return pd.DataFrame()
    df_combined = pd.concat(valid_dfs, ignore_index=True)
    df_combined.dropna(subset=[TARGET_VAR], inplace=True)
    all_vars = [col for col in df_combined.columns if col != TARGET_VAR]
    st.session_state['global_process_vars'] = all_vars
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    return df_combined

def train_model(df):
    vars = st.session_state['global_process_vars']
    if df.empty or not vars: return None, None
    X, Y = df[vars], df[TARGET_VAR]
    if Y.nunique() < 2: return None, None
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42).fit(X_scaled, Y)
    return model, scaler

def predict_weld_risk(model, scaler, input_series):
    if model is None: return 0.5
    df_in = pd.DataFrame([input_series.to_dict()], columns=st.session_state['global_process_vars']).fillna(0)
    return model.predict_proba(scaler.transform(df_in))[:, 1][0]

# =================================================================
# 2. 사이드바 (파일 로드 및 학습)
# =================================================================
with st.sidebar:
    st.header("📂 데이터 로드")
    f_init = st.file_uploader("1. UI 초기 조건", type=['xlsx', 'csv'])
    f_real = st.file_uploader("2. 학습 데이터", type=['xlsx', 'csv'])

    if st.button("🚀 모델 학습 시작"):
        if f_init and f_real:
            df_init = pd.read_excel(f_init) if f_init.name.endswith('xlsx') else pd.read_csv(f_init)
            df_real = pd.read_excel(f_real) if f_real.name.endswith('xlsx') else pd.read_csv(f_real)
            
            df_p = process_weld_data(None, df_real)
            model, scaler = train_model(df_p)
            
            if model:
                st.session_state['model'], st.session_state['scaler'] = model, scaler
                st.session_state['df_weld'] = df_p
                ui_cols = [c for c in df_init.columns if c != TARGET_VAR]
                st.session_state['ui_display_vars'] = ui_cols
                st.session_state['selected_knowhow_vars'] = ui_cols.copy()
                
                # 초기값 세팅
                init_row = df_init.iloc[0]
                for v in ui_cols:
                    st.session_state[f'input_{v}'] = int(init_row.get(v, 0))
                st.success("학습 완료!")
                st.rerun()

# =================================================================
# 3. 메인 UI (진단 및 최적화)
# =================================================================
st.title("Weld Line AI 통합 진단 시스템")
tab1, tab2 = st.tabs(["진단 및 최적화", "데이터 확인"])

with tab1:
    # A. 현재 공정 조건 입력
    st.header("A. 현재 공정 조건 입력")
    ui_vars = st.session_state['ui_display_vars']
    current_inputs = {}

    if ui_vars:
        cols = st.columns(3)
        for i, var in enumerate(ui_vars):
            b_min, b_max = st.session_state['global_bounds'].get(var, (0, 300))
            # 🌟 해결 포인트 1: value는 세션에서 가져오고, key는 충돌 방지를 위해 다르게 설정
            with cols[i % 3]:
                val = st.slider(
                    f"{var}", int(b_min), int(b_max),
                    value=st.session_state.get(f'input_{var}', int((b_min+b_max)/2)),
                    key=f"widget_{var}"
                )
                st.session_state[f'input_{var}'] = val
                current_inputs[var] = val

    st.markdown("---")

    # B. 전문가 노하우 입력
    st.header("B. 전문가 노하우 입력")
    conf = st.slider("노하우 반영 강도 (%)", 0, 100, value=int(st.session_state['conf_level']))
    st.session_state['influence_factor_display_val'] = conf / 100.0
    
    sel_vars = st.multiselect("대상 변수 선택", options=ui_vars, default=st.session_state['selected_knowhow_vars'])
    st.session_state['selected_knowhow_vars'] = sel_vars

    if sel_vars:
        k_cols = st.columns(3)
        for i, var in enumerate(sel_vars):
            with k_cols[i % 3]:
                st.selectbox(f"{var} 조절 의도", ["Keep_Constant", "Increase", "Decrease"], key=f"intent_{var}")
                with st.expander("🛠 경계 설정"):
                    # 현재 설정된 경계값 가져오기
                    low, high = st.session_state['global_bounds'].get(var, (0, 300))
                    # 전체 가능한 범위 (GLOBAL_BOUNDS 기준)
                    g_low, g_high = GLOBAL_BOUNDS.get(var, (0, 500))
                    
                    new_range = st.slider(f"{var} 범위 설정", int(g_low), int(g_high), (int(low), int(high)), key=f"range_{var}")
                    
                    if st.button(f"적용 및 UI 반영", key=f"btn_{var}"):
                        # 🌟 해결 포인트 2: 세션 데이터 수정 후 즉시 rerun 호출
                        st.session_state['global_bounds'][var] = (new_range[0], new_range[1])
                        st.session_state[f'input_{var}'] = int((new_range[0] + new_range[1]) / 2)
                        st.success(f"{var} 적용 완료")
                        st.rerun()

    st.markdown("---")

    # C. 진단 및 결과
    if st.button("진단 실행", type="primary"):
        if st.session_state['model']:
            risk = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], pd.Series(current_inputs))
            st.session_state['current_risk_display'] = risk
        else:
            st.error("모델을 먼저 학습시켜주세요.")

    if st.session_state['current_risk_display'] is not None:
        st.metric("불량 위험도", f"{st.session_state['current_risk_display']*100:.2f}%")

with tab2:
    if not st.session_state['df_weld'].empty:
        st.dataframe(st.session_state['df_weld'].head(20))
    else:
        st.write("데이터가 없습니다.")

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

GLOBAL_BOUNDS = {
    'T_Melt': (200, 300), 'V_Inj': (1, 10), 'P_Pack': (50, 100), 
    'T_Mold': (30, 80), 'Meter': (180, 200), 'VP_Switch_Pos': (10, 20)
}

TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5

# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['model'] = None
    st.session_state['df_weld'] = pd.DataFrame()
    st.session_state['global_process_vars'] = []
    st.session_state['ui_display_vars'] = []
    st.session_state['global_bounds'] = GLOBAL_BOUNDS.copy()
    st.session_state['conf_level'] = 75.0
    st.session_state['current_risk_display'] = None
    st.session_state['knowhow_settings'] = {}
    st.session_state['selected_knowhow_vars'] = []

# =================================================================
# 1. 주요 함수
# =================================================================
def predict_weld_risk(model, scaler, input_series):
    if model is None: return 0.5
    df_in = pd.DataFrame([input_series.to_dict()], columns=st.session_state['global_process_vars']).fillna(0)
    return model.predict_proba(scaler.transform(df_in))[:, 1][0]

# =================================================================
# 2. 사이드바 (데이터 로드)
# =================================================================
with st.sidebar:
    st.header("📂 데이터 로드")
    f_init = st.file_uploader("1. UI 초기 조건", type=['xlsx', 'csv'])
    f_real = st.file_uploader("2. 학습 데이터", type=['xlsx', 'csv'])

    if st.button("🚀 모델 학습 시작"):
        if f_init and f_real:
            df_init = pd.read_excel(f_init) if f_init.name.endswith('xlsx') else pd.read_csv(f_init)
            df_real = pd.read_excel(f_real) if f_real.name.endswith('xlsx') else pd.read_csv(f_real)
            
            # 전처리 및 학습
            df_real.dropna(subset=[TARGET_VAR], inplace=True)
            vars = [c for c in df_real.columns if c != TARGET_VAR]
            st.session_state['global_process_vars'] = vars
            df_real[TARGET_VAR] = np.where(df_real[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df_real[vars])
            model = LogisticRegression().fit(X_scaled, df_real[TARGET_VAR])
            
            st.session_state['model'], st.session_state['scaler'] = model, scaler
            st.session_state['df_weld'] = df_real
            ui_cols = [c for c in df_init.columns if c != TARGET_VAR]
            st.session_state['ui_display_vars'] = ui_cols
            st.session_state['selected_knowhow_vars'] = ui_cols.copy()
            
            for v in ui_cols:
                st.session_state[f'input_{v}'] = int(df_init.iloc[0].get(v, 0))
            st.success("✅ 학습 완료!")
            st.rerun()

# =================================================================
# 3. 메인 UI
# =================================================================
st.title("Weld Line AI 통합 진단 및 최적화 시스템")
tab1, tab2 = st.tabs(["진단 및 최적화", "데이터 확인"])

with tab1:
    # A. 현재 공정 조건 입력
    st.header("A. 현재 공정 조건 입력")
    ui_vars = st.session_state['ui_display_vars']
    current_inputs = {}

    if ui_vars:
        cols = st.columns(3)
        for i, var in enumerate(ui_vars):
            # 최신 경계값 가져오기
            b_min, b_max = st.session_state['global_bounds'].get(var, (0, 300))
            
            # 현재 세션값 가져오기 및 범위 이탈 방지
            curr_val = st.session_state.get(f'input_{var}', int((b_min+b_max)/2))
            curr_val = max(b_min, min(curr_val, b_max))

            with cols[i % 3]:
                # 🌟 핵심: key에 b_min, b_max를 포함하여 범위가 바뀔 때 슬라이더를 강제 재생성
                val = st.slider(
                    f"{var}", int(b_min), int(b_max),
                    value=int(curr_val),
                    key=f"slider_{var}_{b_min}_{b_max}" 
                )
                st.session_state[f'input_{var}'] = val
                current_inputs[var] = val

    st.markdown("---")

    # B. 전문가 노하우 입력
    st.header("B. 전문가 노하우 입력")
    conf = st.slider("노하우 반영 강도 (%)", 0, 100, value=int(st.session_state['conf_level']), key="conf_slider")
    st.session_state['conf_level'] = conf
    
    sel_vars = st.multiselect("대상 변수 선택", options=ui_vars, default=st.session_state['selected_knowhow_vars'])
    st.session_state['selected_knowhow_vars'] = sel_vars

    if sel_vars:
        k_cols = st.columns(3)
        for i, var in enumerate(sel_vars):
            with k_cols[i % 3]:
                st.selectbox(f"{var} 조절 의도", ["Keep_Constant", "Increase", "Decrease"], key=f"intent_{var}")
                with st.expander("🛠 경계 범위 수정"):
                    # 현재 설정된 경계
                    low, high = st.session_state['global_bounds'].get(var, (0, 300))
                    # 전체 가능한 절대 범위
                    abs_low, abs_high = GLOBAL_BOUNDS.get(var, (0, 500))
                    
                    new_range = st.slider(
                        f"{var} 신규 범위", 
                        int(abs_low), int(abs_high), (int(low), int(high)), 
                        key=f"range_setter_{var}"
                    )
                    
                    # 🌟 적용 버튼 클릭 시 로직
                    if st.button(f"적용 및 UI 반영", key=f"btn_{var}", use_container_width=True):
                        # 1. 경계값 저장
                        st.session_state['global_bounds'][var] = (new_range[0], new_range[1])
                        # 2. 핸들 위치를 새 범위의 중앙으로 이동
                        st.session_state[f'input_{var}'] = int((new_range[0] + new_range[1]) / 2)
                        # 3. 알림 및 즉시 리런
                        st.toast(f"✅ {var} 범위 변경 완료!")
                        st.rerun()

    st.markdown("---")

    # C. 진단 실행
    if st.button("🔍 진단 실행", type="primary", use_container_width=True):
        if st.session_state['model']:
            risk = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], pd.Series(current_inputs))
            st.session_state['current_risk_display'] = risk
        else:
            st.error("먼저 사이드바에서 모델을 학습시켜주세요.")

    if st.session_state['current_risk_display'] is not None:
        risk_val = st.session_state['current_risk_display'] * 100
        color = "red" if risk_val > 50 else "orange" if risk_val > 20 else "green"
        st.markdown(f"### 진단 결과: <span style='color:{color}'>{risk_val:.2f}%</span>", unsafe_allow_html=True)

with tab2:
    st.dataframe(st.session_state['df_weld'])

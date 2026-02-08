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

PROCESS_VARS = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos']
TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5

# 절대 경계값 (시스템 기준)
ABS_BOUNDS = {
    'T_Melt': (200.0, 300.0), 'V_Inj': (1.0, 10.0), 'P_Pack': (50.0, 100.0), 
    'T_Mold': (30.0, 80.0), 'Meter': (180.0, 200.0), 'VP_Switch_Pos': (10.0, 20.0)
}

# 기본 입력값
DEFAULT_INPUT_VALS = {
    'T_Melt': 230.0, 'V_Inj': 3.0, 'P_Pack': 70.0, 
    'T_Mold': 50.0, 'Meter': 195.0, 'VP_Switch_Pos': 14.0
}

# 세션 상태 초기화
if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['df_weld'] = pd.DataFrame()
    st.session_state['scaler'] = None
    st.session_state['current_risk_display'] = None
    st.session_state['optimization_result'] = None
    st.session_state['conf_level'] = 75.0
    st.session_state['influence_factor_display_val'] = 0.75
    # ⭐ 가변적 경계값 저장소
    st.session_state['current_bounds'] = ABS_BOUNDS.copy()
    # 초기 입력값 설정
    for var, val in DEFAULT_INPUT_VALS.items():
        if f'input_{var}' not in st.session_state:
            st.session_state[f'input_{var}'] = float(val)

# -------------------------------------------------------------
# 콜백 함수
# -------------------------------------------------------------
def update_influence_factor():
    if 'expert_confidence_slider' in st.session_state:
        conf = st.session_state['expert_confidence_slider']
        st.session_state['conf_level'] = conf
        st.session_state['influence_factor_display_val'] = conf / 100.0
    st.session_state['current_risk_display'] = None 
    st.session_state['optimization_result'] = None 

# =================================================================
# 1. 데이터 처리 함수 (기존 로직 유지)
# =================================================================
def load_df_from_uploader(uploaded_file):
    if uploaded_file is not None:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'csv': df = pd.read_csv(uploaded_file)
        else: df = pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = df.columns.str.strip()
        return df
    return None

def process_weld_data(df_virtual, df_real):
    valid_dfs = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    if not valid_dfs: return pd.DataFrame()
    df_combined = pd.concat(valid_dfs, ignore_index=True)
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    return df_combined

def train_model(df):
    if df.empty: return None, None
    X, Y = df[PROCESS_VARS], df[TARGET_VAR]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42).fit(X_scaled, Y)
    return model, scaler

def predict_weld_risk(model, scaler, input_data):
    if model is None: return 0.5
    input_df = pd.DataFrame([input_data], columns=PROCESS_VARS)
    return model.predict_proba(scaler.transform(input_df))[:, 1][0]

# =================================================================
# 4. Streamlit UI
# =================================================================
with st.sidebar:
    st.header("📂 데이터 로드")
    up_init = st.file_uploader("1. UI 초기 조건", type=['xlsx', 'csv'])
    up_virtual = st.file_uploader("2. 가상 학습 데이터", type=['xlsx', 'csv'])
    up_real = st.file_uploader("3. 해석 학습 데이터", type=['xlsx', 'csv'])

    if st.button("🚀 모델 학습 시작"):
        df_real = load_df_from_uploader(up_real)
        if df_real is not None:
            df_init = load_df_from_uploader(up_init)
            df_virt = load_df_from_uploader(up_virtual)
            df_p = process_weld_data(df_virt, df_real)
            st.session_state['df_weld'] = df_p
            model, scaler = train_model(df_p)
            st.session_state['model'], st.session_state['scaler'] = model, scaler
            if df_init is not None:
                for var in PROCESS_VARS:
                    if var in df_init.columns:
                        st.session_state[f'input_{var}'] = float(df_init.iloc[0][var])
            st.success("학습 완료!")
            st.rerun()

st.title("Weld Line AI 통합 진단 및 최적화 시스템")
tab1, tab2 = st.tabs(["탭 1. 진단 및 최적화", "탭 2. 데이터 확인"])

with tab1:
    st.header("A. 현재 공정 조건 입력")
    # 화면 그리드 구성
    cols = st.columns(3)
    input_vars = {}
    
    for i, var in enumerate(PROCESS_VARS):
        # ⭐ 핵심: 세션에 저장된 현재 경계값 가져오기
        b_min, b_max = st.session_state['current_bounds'][var]
        curr_val = st.session_state[f'input_{var}']
        
        # 범위 이탈 보정
        curr_val = max(b_min, min(curr_val, b_max))
        
        with cols[i % 3]:
            # ⭐ 핵심: key에 경계값을 포함하여 범위 변경 시 슬라이더 강제 재생성
            input_vars[var] = st.slider(
                f'{var}', b_min, b_max, 
                value=float(curr_val),
                key=f"main_slider_{var}_{b_min}_{b_max}",
                on_change=lambda: st.session_state.update({'current_risk_display': None, 'optimization_result': None})
            )
            st.session_state[f'input_{var}'] = input_vars[var]

    st.markdown("---")
    st.header("B. 전문가의 정성적/정량적 노하우 입력")
    
    # 1. 확신 수준
    st.subheader("1. 전문가 확신 수준")
    st.slider('노하우 반영도 (%)', 0.0, 100.0, value=st.session_state['conf_level'], 
              key='expert_confidence_slider', on_change=update_influence_factor)

    # 2. 사출 속도 & 3. 금형 온도 (대표 예시)
    for var, label in zip(['V_Inj', 'T_Mold'], ['2. 사출 속도 (V_Inj)', '3. 금형 온도 (T_Mold)']):
        st.subheader(label)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.checkbox('정성적 적용', key=f'{var}_qual_apply')
        with c2:
            st.selectbox('조절 의도', ['Keep_Constant', 'Increase', 'Decrease'], key=f'{var}_intent')
        with c3:
            # ⭐ 핵심: 여기서 범위를 수정하고 상단에 반영하는 기능
            with st.expander("🛠 범위 직접 수정"):
                abs_low, abs_high = ABS_BOUNDS[var]
                curr_low, curr_high = st.session_state['current_bounds'][var]
                
                new_range = st.slider(f"{var} 설정 범위", abs_low, abs_high, (curr_low, curr_high), key=f"setter_{var}")
                
                if st.button(f"{var} 범위 적용 및 상단 UI 반영", key=f"btn_{var}"):
                    # 1. 새로운 경계값 세션 저장
                    st.session_state['current_bounds'][var] = (new_range[0], new_range[1])
                    # 2. 상단 슬라이더 핸들을 새 범위의 중앙으로 이동
                    st.session_state[f'input_{var}'] = (new_range[0] + new_range[1]) / 2
                    # 3. 화면 갱신 (상단 슬라이더가 Key 변화를 감지하고 새로 그려짐)
                    st.toast(f"✅ {var} UI가 업데이트되었습니다.")
                    st.rerun()

    st.markdown("---")
    st.header("C. 진단 실행 및 결과")
    st.write(f"현재 노하우 영향 계수: **{st.session_state['influence_factor_display_val']:.2f}**")
    
    if st.button("🔴 Weld Line 통합 진단 실행", use_container_width=True):
        if st.session_state['model']:
            st.session_state['current_risk_display'] = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], input_vars)
        else: st.error("모델 학습이 필요합니다.")

    if st.session_state['current_risk_display'] is not None:
        risk = st.session_state['current_risk_display'] * 100
        st.metric("현재 조건 불량 위험도", f"{risk:.2f}%")

with tab2:
    st.subheader("데이터 및 모델 정보")
    if not st.session_state['df_weld'].empty:
        st.dataframe(st.session_state['df_weld'].head(50))
    else:
        st.info("데이터가 로드되지 않았습니다.")

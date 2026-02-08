import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler 
import json 

# =================================================================
# 0. 초기 설정 및 상수
# =================================================================
st.set_page_config(layout="wide", page_title="Weld Line 통합 진단 시스템")

GLOBAL_BOUNDS = {
    'T_Melt': (200, 300), 'V_Inj': (1, 10), 'P_Pack': (50, 100), 
    'T_Mold': (30, 80), 'Meter': (180, 200), 'VP_Switch_Pos': (10, 20)
}

if 'ui_display_vars' not in st.session_state:
    st.session_state['ui_display_vars'] = [] 

TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5

# 시스템 상태 초기화
states = {
    'model': None, 'df_weld': pd.DataFrame(), 'df_init': None, 
    'df_virtual': None, 'df_real': None, 'scaler': None,
    'global_process_vars': [], 'default_init_values': {},
    'current_risk_display': None, 'optimization_result': None,
    'conf_level': 75.0, 'influence_factor_display_val': 0.75,
    'knowhow_settings': {}, 'knowhow_temp_storage': {},
    'global_bounds': GLOBAL_BOUNDS.copy(),
    'selected_knowhow_vars': []
}

for key, value in states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------------------------------------------------
# 콜백 함수
# -------------------------------------------------------------
def update_influence_factor():
    new_confidence_level = st.session_state.get('expert_confidence_slider', 75.0)
    st.session_state['conf_level'] = new_confidence_level
    st.session_state['influence_factor_display_val'] = new_confidence_level / 100.0
    st.session_state['current_risk_display'] = None 
    st.session_state['optimization_result'] = None 

# =================================================================
# 1. 데이터 로드 및 전처리
# =================================================================
@st.cache_data(show_spinner=False)
def load_df_from_uploader(uploaded_file):
    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"⚠️ 파일 로드 중 오류: {e}")
    return None

def process_weld_data(df_virtual, df_real):
    valid_dfs = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    if not valid_dfs: return pd.DataFrame()
    df_combined = pd.concat(valid_dfs, ignore_index=True)
    if TARGET_VAR not in df_combined.columns: return pd.DataFrame()
    
    all_vars = [col for col in df_combined.columns if col != TARGET_VAR]
    st.session_state['global_process_vars'] = all_vars
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    return df_combined[all_vars + [TARGET_VAR]].fillna(0)

def train_model(df):
    if df.empty or not st.session_state['global_process_vars']: return None, None
    X = df[st.session_state['global_process_vars']]
    Y = df[TARGET_VAR]
    if Y.nunique() < 2: return None, None
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42).fit(X_scaled, Y)
    return model, scaler

def predict_weld_risk(model, scaler, input_data_series):
    if model is None or scaler is None: return 0.5
    input_df = pd.DataFrame([input_data_series.to_dict()], columns=st.session_state['global_process_vars']).fillna(0)
    input_scaled = scaler.transform(input_df)
    return model.predict_proba(input_scaled)[:, 1][0]

# =================================================================
# 4. Streamlit UI
# =================================================================
with st.sidebar:
    st.header("📂 데이터 및 모델 학습")
    up_init = st.file_uploader("1. UI 초기 조건", type=['xlsx', 'csv'])
    up_virtual = st.file_uploader("2. 가상 학습 데이터", type=['xlsx', 'csv'])
    up_real = st.file_uploader("3. 해석 학습 데이터", type=['xlsx', 'csv'])

    def load_and_train_model():
        st.session_state['df_init'] = load_df_from_uploader(up_init)
        st.session_state['df_virtual'] = load_df_from_uploader(up_virtual)
        st.session_state['df_real'] = load_df_from_uploader(up_real)
        
        if st.session_state['df_init'] is not None:
            df_p = process_weld_data(st.session_state['df_virtual'], st.session_state['df_real'])
            st.session_state['df_weld'] = df_p
            model, scaler = train_model(df_p)
            st.session_state['model'], st.session_state['scaler'] = model, scaler
            
            if model:
                cols = [c for c in st.session_state['df_init'].columns if c != TARGET_VAR]
                st.session_state['ui_display_vars'] = cols
                init_row = st.session_state['df_init'].iloc[0]
                for v in cols:
                    st.session_state[f'input_{v}'] = int(init_row.get(v, (GLOBAL_BOUNDS.get(v, (0,300))[0]+GLOBAL_BOUNDS.get(v, (0,300))[1])/2))
                st.session_state['selected_knowhow_vars'] = cols.copy()

    st.button("🚀 모델 학습 시작", on_click=load_and_train_model)

st.title("Weld Line AI 통합 진단 및 최적화 시스템")
tab1, tab2 = st.tabs(["탭 1. 진단 및 최적화", "탭 2. 데이터 확인"])

with tab1:
    st.header("A. 현재 공정 조건 입력")
    ui_vars = st.session_state['ui_display_vars']
    input_vars = {}
    
    if ui_vars:
        cols = st.columns(3)
        for i, var in enumerate(ui_vars):
            b_min, b_max = st.session_state['global_bounds'].get(var, (0, 300))
            # 위젯의 value를 session_state에서 관리
            with cols[i % 3]:
                input_vars[var] = st.slider(
                    f'{var}', int(b_min), int(b_max), 
                    key=f'input_{var}', # key를 통해 직접 관리
                    on_change=lambda: st.session_state.update({'optimization_result': None})
                )

    st.markdown("---")
    st.header("B. 전문가 노하우 입력")
    st.slider('노하우 반영 강도 (%)', 0.0, 100.0, key='expert_confidence_slider', on_change=update_influence_factor)
    
    st.session_state['selected_knowhow_vars'] = st.multiselect('노하우 적용 변수 선택', options=ui_vars, default=st.session_state['selected_knowhow_vars'])
    
    if st.session_state['selected_knowhow_vars']:
        k_cols = st.columns(3)
        for i, var in enumerate(st.session_state['selected_knowhow_vars']):
            with k_cols[i % 3]:
                intent = st.selectbox(f'{var} 의도', ['Keep_Constant', 'Increase', 'Decrease'], key=f'intent_{var}')
                with st.expander("🛠️ 경계 설정"):
                    curr_b = st.session_state['global_bounds'].get(var, (0, 300))
                    # ⚠️ 여기서 value를 직접 수정하지 않고 UI로만 받음
                    new_b = st.slider(f'{var} 범위', float(GLOBAL_BOUNDS.get(var, (0,300))[0]), float(GLOBAL_BOUNDS.get(var, (0,300))[1]), value=(float(curr_b[0]), float(curr_b[1])), key=f'b_slider_{var}')
                    
                    if st.button(f'적용', key=f'btn_{var}'):
                        # 🌟 에러 해결 핵심: Bounds만 바꾸고 슬라이더 값은 다음 렌더링 때 계산되게 함
                        st.session_state['global_bounds'][var] = (int(new_b[0]), int(new_b[1]))
                        st.session_state[f'input_{var}'] = int((new_b[0] + new_b[1]) / 2)
                        st.rerun() # 🌟 새로고침하여 위젯들에 변경된 값 반영

                st.session_state['knowhow_settings'][var] = {'qual_intent': intent, 'qual_apply': intent != 'Keep_Constant'}

    st.markdown("---")
    st.header("C. 진단 실행 및 결과")
    
    def run_diag():
        full_data = {v: st.session_state.get(f'input_{v}', 0) for v in st.session_state['global_process_vars']}
        st.session_state['current_risk_display'] = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], pd.Series(full_data))

    if st.button("진단 실행", type="primary"): run_diag()
    
    if st.session_state['current_risk_display'] is not None:
        risk = st.session_state['current_risk_display'] * 100
        st.metric("불량 위험도", f"{risk:.2f}%")

# ... (이후 최적화 로직 및 Tab2는 기존과 동일하게 유지)

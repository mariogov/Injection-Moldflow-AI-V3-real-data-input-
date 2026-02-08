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

# 중요: 이 이름들이 엑셀의 컬럼명과 일치해야 합니다.
PROCESS_VARS = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos']
TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5

# 시스템 절대 경계 (범위 수정 시 가이드 라인)
ABS_BOUNDS = {
    'T_Melt': (200.0, 300.0), 'V_Inj': (1.0, 10.0), 'P_Pack': (50.0, 100.0), 
    'T_Mold': (30.0, 80.0), 'Meter': (180.0, 200.0), 'VP_Switch_Pos': (10.0, 20.0)
}

# 기본 입력값
DEFAULT_INPUT_VALS = {
    'T_Melt': 230.0, 'V_Inj': 3.0, 'P_Pack': 70.0, 
    'T_Mold': 50.0, 'Meter': 195.0, 'VP_Switch_Pos': 14.0
}

# 시스템 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': True,
        'model': None,
        'scaler': None,
        'df_weld': pd.DataFrame(),
        'current_bounds': ABS_BOUNDS.copy(), # 가변 경계 상태
        'current_risk_display': None,
        'optimization_result': None,
        'conf_level': 75.0,
        'influence_factor_display_val': 0.75
    })
    # 각 변수별 초기 입력값 세션 저장
    for var, val in DEFAULT_INPUT_VALS.items():
        st.session_state[f'input_{var}'] = float(val)

# -------------------------------------------------------------
# 콜백 함수: 확신 수준 변경 시
# -------------------------------------------------------------
def update_influence_factor():
    if 'expert_confidence_slider' in st.session_state:
        conf = st.session_state['expert_confidence_slider']
        st.session_state['conf_level'] = conf
        st.session_state['influence_factor_display_val'] = conf / 100.0
    st.session_state['current_risk_display'] = None 
    st.session_state['optimization_result'] = None 

# =================================================================
# 1. 데이터 로드 및 학습 함수 (KeyError 방지 강화)
# =================================================================
def load_df_from_uploader(uploaded_file):
    if uploaded_file is not None:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            # ⭐ KeyError 방지: 컬럼명 앞뒤 공백 제거
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            st.error(f"⚠️ 파일 로드 중 오류 발생: {e}")
    return None

def process_weld_data(df_virtual, df_real):
    valid_dfs = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    if not valid_dfs: return pd.DataFrame()
    
    df_combined = pd.concat(valid_dfs, ignore_index=True)
    
    # ⭐ KeyError 방지: 필수 컬럼 존재 확인
    missing_cols = [c for c in PROCESS_VARS + [TARGET_VAR] if c not in df_combined.columns]
    if missing_cols:
        st.error(f"🚨 데이터에 다음 컬럼이 누락되었습니다: {missing_cols}")
        return pd.DataFrame()
        
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    return df_combined[PROCESS_VARS + [TARGET_VAR]].dropna()

def train_model(df):
    if df.empty: return None, None
    try:
        X = df[PROCESS_VARS]
        Y = df[TARGET_VAR]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42).fit(X_scaled, Y)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ 모델 학습 오류: {e}")
        return None, None

def predict_weld_risk(model, scaler, input_dict):
    if model is None or scaler is None: return 0.5
    input_df = pd.DataFrame([input_dict], columns=PROCESS_VARS)
    input_scaled = scaler.transform(input_df)
    return model.predict_proba(input_scaled)[:, 1][0]

# =================================================================
# 4. Streamlit UI 및 로직
# =================================================================

# -----------------
# 사이드바 (데이터 로드)
# -----------------
with st.sidebar:
    st.header("📂 데이터 및 모델 학습")
    up_init = st.file_uploader("1. UI 초기 조건 [선택]", type=['xlsx', 'csv'])
    up_real = st.file_uploader("2. 학습 데이터 [필수]", type=['xlsx', 'csv'])

    if st.button("🚀 AI 모델 학습 시작", type="primary"):
        df_real_raw = load_df_from_uploader(up_real)
        if df_real_raw is not None:
            df_p = process_weld_data(None, df_real_raw)
            if not df_p.empty:
                model, scaler = train_model(df_p)
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['df_weld'] = df_p
                
                # 초기 조건 파일 처리
                df_init = load_df_from_uploader(up_init)
                if df_init is not None and not df_init.empty:
                    for var in PROCESS_VARS:
                        if var in df_init.columns:
                            st.session_state[f'input_{var}'] = float(df_init.iloc[0][var])
                st.success("✅ 학습 완료 및 UI 업데이트!")
                st.rerun()

# -----------------
# 메인 페이지
# -----------------
st.title("Weld Line AI 통합 진단 및 최적화 시스템")
tab1, tab2 = st.tabs(["탭 1. 진단 및 최적화", "탭 2. 데이터 확인"])

with tab1:
    # A. 현재 공정 조건 입력
    st.header("A. 현재 공정 조건 입력")
    cols = st.columns(3)
    current_inputs = {}
    
    for i, var in enumerate(PROCESS_VARS):
        # 세션에서 현재 유효한 경계값과 입력값 가져오기
        b_min, b_max = st.session_state['current_bounds'][var]
        curr_val = st.session_state[f'input_{var}']
        
        # 범위 이탈 보정
        curr_val = max(b_min, min(curr_val, b_max))
        
        with cols[i % 3]:
            # ⭐ 핵심: key에 경계값을 포함하여 범위 변경 시 슬라이더 강제 재생성
            val = st.slider(
                f"{var}", float(b_min), float(b_max), float(curr_val),
                key=f"slider_{var}_{b_min}_{b_max}",
                on_change=lambda: st.session_state.update({'current_risk_display': None, 'optimization_result': None})
            )
            st.session_state[f'input_{var}'] = val
            current_inputs[var] = val

    st.markdown("---")

    # B. 전문가 노하우 입력 (범위 수정 기능 포함)
    st.header("B. 전문가의 정성적/정량적 노하우 입력")
    
    # 1. 확신 수준
    st.subheader("1. 전문가 확신 수준")
    st.slider('노하우 반영도 (%)', 0.0, 100.0, 
              value=st.session_state['conf_level'], 
              key='expert_confidence_slider', 
              on_change=update_influence_factor)

    # 2. 정성적/정량적 범위 수정 (대표 변수 예시)
    for var in ['V_Inj', 'T_Mold']:
        st.subheader(f"⚙️ {var} 노하우 및 범위 제약")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.checkbox('정성적 적용', key=f'{var}_qual_apply')
        with c2:
            st.selectbox('조절 의도', ['Keep_Constant', 'Increase', 'Decrease'], key=f'{var}_intent')
        with c3:
            with st.expander("🛠 UI 슬라이더 범위 수정"):
                abs_l, abs_h = ABS_BOUNDS[var]
                curr_l, curr_h = st.session_state['current_bounds'][var]
                
                new_range = st.slider(f"{var} 설정 범위", abs_l, abs_h, (curr_l, curr_h), key=f"set_{var}")
                
                if st.button(f"{var} 범위 적용", key=f"btn_{var}", use_container_width=True):
                    # 1. 경계값 업데이트
                    st.session_state['current_bounds'][var] = (new_range[0], new_range[1])
                    # 2. 핸들을 중앙으로 강제 이동
                    st.session_state[f'input_{var}'] = (new_range[0] + new_range[1]) / 2
                    # 3. 즉시 리런 (상단 슬라이더의 Key가 바뀌면서 새로 그려짐)
                    st.rerun()

    st.markdown("---")

    # C. 진단 결과
    st.header("C. 진단 실행 및 결과")
    if st.button("🔴 Weld Line 통합 진단 실행", use_container_width=True):
        if st.session_state['model']:
            risk = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], current_inputs)
            st.session_state['current_risk_display'] = risk
        else:
            st.error("먼저 사이드바에서 학습 데이터를 로드하고 모델을 학습시켜주세요.")

    if st.session_state['current_risk_display'] is not None:
        risk_pct = st.session_state['current_risk_display'] * 100
        st.metric("현재 불량 위험 확률", f"{risk_pct:.2f}%")
        if risk_pct >= 50:
            st.error("⚠️ 위험도가 높습니다. 공정 조건 최적화가 필요합니다.")
        else:
            st.success("✅ 현재 조건이 안정적입니다.")

with tab2:
    st.header("모델 및 데이터 확인")
    if not st.session_state['df_weld'].empty:
        st.write("학습 데이터 미리보기")
        st.dataframe(st.session_state['df_weld'].head(100))
    else:
        st.info("데이터가 아직 로드되지 않았습니다.")

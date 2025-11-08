import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler 
import json # 최적화 결과 저장을 위해 추가

# =================================================================
# 0. 초기 설정 및 상수
# =================================================================
st.set_page_config(layout="wide", page_title="Weld Line 통합 진단 시스템")

# 🌟 UI 슬라이더에 사용할 6가지 변수와 그 속성 정의 (컬럼명이 이와 일치해야 함)
# 사용자가 업로드하는 데이터의 컬럼명이 아래 key와 일치해야 합니다.
UI_VARS_SPEC = {
    'T_Melt': {'label': '용융 온도', 'unit': '°C', 'min': 200.0, 'max': 300.0, 'step': 5.0, 'default': 230.0},
    'V_Inj': {'label': '사출 속도', 'unit': 'mm/s', 'min': 1.0, 'max': 10.0, 'step': 1.0, 'default': 3.0},
    'P_Pack': {'label': '보압', 'unit': 'MPa', 'min': 50.0, 'max': 100.0, 'step': 5.0, 'default': 70.0},
    'T_Mold': {'label': '금형 온도', 'unit': '°C', 'min': 30.0, 'max': 80.0, 'step': 5.0, 'default': 50.0},
    'Meter': {'label': '계량 위치', 'unit': 'mm', 'min': 180.0, 'max': 200.0, 'step': 1.0, 'default': 195.0},
    'VP_Switch_Pos': {'label': 'VP 전환 위치', 'unit': 'mm', 'min': 10.0, 'max': 20.0, 'step': 1.0, 'default': 14.0}
}
UI_INPUT_VARS = list(UI_VARS_SPEC.keys()) # UI_INPUT_VARS는 6개 변수의 키 리스트

# 🌟 전체 공정 변수 리스트는 학습 데이터에서 동적으로 결정됩니다.
GLOBAL_PROCESS_VARS = [] 

# 종속 변수 정의 (Y 변수)
TARGET_VAR = 'Y_Weld'
# 불량 기준 (0.5 이상이면 1, 미만이면 0)
DEFECT_THRESHOLD = 0.5

# BOUNDS 정의: UI_INPUT_VARS에 대한 전역 경계
GLOBAL_BOUNDS = {var: (spec['min'], spec['max']) for var, spec in UI_VARS_SPEC.items()}


# 시스템 상태 초기화 (세션 상태)
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'df_weld' not in st.session_state:
    st.session_state['df_weld'] = pd.DataFrame()
if 'df_init' not in st.session_state:
    st.session_state['df_init'] = None
if 'df_virtual' not in st.session_state:
    st.session_state['df_virtual'] = None
if 'df_real' not in st.session_state:
    st.session_state['df_real'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
# 🌟 전체 공정 변수 저장을 위한 세션 상태
if 'global_process_vars' not in st.session_state:
    st.session_state['global_process_vars'] = GLOBAL_PROCESS_VARS
if 'default_init_values' not in st.session_state:
    st.session_state['default_init_values'] = {} # 나머지 변수의 초기값을 저장할 딕셔너리
    
# 진단 결과 저장을 위한 세션 상태 추가
if 'current_risk_display' not in st.session_state:
    st.session_state['current_risk_display'] = None
if 'optimization_result' not in st.session_state:
    st.session_state['optimization_result'] = None

# 슬라이더 오류 방지 로직: 초기값을 무조건 float으로 설정
for var in UI_INPUT_VARS:
    if f'input_{var}' not in st.session_state:
        st.session_state[f'input_{var}'] = float(UI_VARS_SPEC[var]['default'])

# UI 상태를 위한 세션 상태 추가
if 'conf_level' not in st.session_state:
    st.session_state['conf_level'] = 75.0
if 'influence_factor_display_val' not in st.session_state:
    st.session_state['influence_factor_display_val'] = st.session_state['conf_level'] / 100.0

# V_Inj, T_Mold 노하우 관련 세션 상태 초기화 (V_Inj, T_Mold만 적용)
for var in ['v_inj', 't_mold']:
    if f'{var}_qual_apply' not in st.session_state:
        st.session_state[f'{var}_qual_apply'] = False
    if f'{var}_quant_apply' not in st.session_state:
        st.session_state[f'{var}_quant_apply'] = False
    if f'{var}_qual_intent' not in st.session_state:
        st.session_state[f'{var}_qual_intent'] = 'Keep_Constant'
    if f'{var}_quant_percent' not in st.session_state:
        st.session_state[f'{var}_quant_percent'] = 0.0


# -------------------------------------------------------------
# 🌟 콜백 함수: 전문가 확신 수준 변경 시 영향 계수 업데이트 (동일)
# -------------------------------------------------------------
def update_influence_factor():
    """전문가 확신 수준 슬라이더 변경 시 영향 계수를 업데이트하고 진단 결과를 초기화합니다."""
    
    if 'expert_confidence_slider' in st.session_state:
        new_confidence_level = st.session_state['expert_confidence_slider']
    else:
        new_confidence_level = st.session_state['conf_level'] 
        
    new_influence_factor = new_confidence_level / 100.0
    
    st.session_state['conf_level'] = new_confidence_level
    st.session_state['influence_factor_display_val'] = new_influence_factor
    
    st.session_state['current_risk_display'] = None 
    st.session_state['optimization_result'] = None 


# =================================================================
# 1. 데이터 로드 및 전처리 함수 (확장)
# =================================================================

@st.cache_data(show_spinner=False)
def load_df_from_uploader(uploaded_file):
    """업로드된 파일(xlsx, csv)을 Pandas DataFrame으로 로드합니다."""
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                uploaded_file.seek(0)
                try:
                    # 쉼표 구분자 시도
                    df = pd.read_csv(uploaded_file, sep=',')
                except Exception:
                    uploaded_file.seek(0)
                    # 탭 구분자 시도
                    df = pd.read_csv(uploaded_file, sep='\t')
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error(f"⚠️ 지원하지 않는 파일 형식입니다: .{file_extension}")
                return None
            
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"⚠️ 파일 로드 중 오류 발생: {e}")
            return None
    return None

def process_weld_data(df_virtual, df_real):
    """실제 데이터와 가상 데이터를 결합하고 전처리합니다. 전체 변수를 GLOBAL_PROCESS_VARS로 설정합니다."""
    
    valid_dataframes = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    
    if not valid_dataframes:
        st.session_state['global_process_vars'] = []
        return pd.DataFrame() 

    # 🌟 수정: concat 전에 공통 컬럼만 남기는 대신, 모든 컬럼을 유지하고 나중에 처리
    df_combined = pd.concat(valid_dataframes, ignore_index=True)
    df_combined.dropna(subset=[TARGET_VAR], inplace=True) # 타겟 변수 NaN 제거
    
    if TARGET_VAR not in df_combined.columns:
        st.error(f"⚠️ 데이터에 필수 타겟 컬럼('{TARGET_VAR}')이 누락되었습니다. 컬럼 이름을 확인해 주세요.")
        st.session_state['global_process_vars'] = []
        return pd.DataFrame()
        
    # 🌟 핵심 수정: TARGET_VAR를 제외한 모든 컬럼을 GLOBAL_PROCESS_VARS로 사용
    all_vars = [col for col in df_combined.columns if col != TARGET_VAR]
    st.session_state['global_process_vars'] = all_vars # 세션 상태에 전체 변수 저장
    
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    
    required_cols = all_vars + [TARGET_VAR]
    
    # 전체 변수만 남기고 처리
    df_processed = df_combined[required_cols].copy()
    
    # 🌟 주의: 학습 데이터의 NaN 값은 0으로 임시 처리. 실제 모델링에서는 평균/중앙값 대체 또는 결측치 제거 필요
    df_processed.fillna(0, inplace=True)
    
    return df_processed

def train_model(df):
    """데이터를 사용하여 로지스틱 회귀 모델을 학습하고 스케일러를 저장합니다."""
    if df.empty or not st.session_state['global_process_vars']:
        return None, None
        
    # 🌟 저장된 전체 변수를 사용
    global_vars = st.session_state['global_process_vars']
    X = df[global_vars]
    Y = df[TARGET_VAR]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, Y)
    
    return model, scaler

def predict_weld_risk(model, scaler, input_data_series):
    """입력 데이터(pd.Series)에 대한 불량 확률을 예측합니다."""
    if model is None or scaler is None:
        return 0.5 
    
    global_vars = st.session_state['global_process_vars']

    if isinstance(input_data_series, pd.Series):
        # 전체 변수 순서에 맞게 데이터를 재구성
        input_df = pd.DataFrame([input_data_series.to_dict()], columns=global_vars)
    elif isinstance(input_data_series, pd.DataFrame) and len(input_data_series) == 1:
        input_df = input_data_series[global_vars]
    else:
        st.error("⚠️ 예측 입력 데이터 형식이 올바르지 않습니다.")
        return 0.5
    
    # NaN 값은 0으로 대체 (학습 데이터 전처리 방식과 일치)
    input_df.fillna(0, inplace=True)
    
    input_scaled = scaler.transform(input_df)
    
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    
    return prediction_proba


# =================================================================
# 4. Streamlit UI 및 로직
# =================================================================

# -----------------
# 사이드바 (데이터 로드)
# -----------------
with st.sidebar:
    st.header("📂 데이터 및 모델 학습")
    
    uploaded_file_init = st.file_uploader(
        "1. UI 초기 조건 (initial_condition.xlsx) [선택]", type=['xlsx', 'csv'], key="init_file"
    )
    uploaded_file_virtual = st.file_uploader(
        "2. 가상 학습 데이터 (test_condition.xlsx) [선택]", type=['xlsx', 'csv'], key="virtual_file"
    )
    uploaded_file_real = st.file_uploader(
        "3. 해석 학습 데이터 (moldflow_condition.xlsx) [필수]", type=['xlsx', 'csv'], key="real_file"
    )

    st.session_state['df_init'] = load_df_from_uploader(uploaded_file_init)
    st.session_state['df_virtual'] = load_df_from_uploader(uploaded_file_virtual)
    st.session_state['df_real'] = load_df_from_uploader(uploaded_file_real)


    def load_and_train_model():
        st.session_state['current_risk_display'] = None
        st.session_state['optimization_result'] = None
        
        # 🌟 데이터 전처리 및 전체 변수 목록 생성
        df_weld_processed = process_weld_data(st.session_state['df_virtual'], st.session_state['df_real'])
        st.session_state['df_weld'] = df_weld_processed
        
        global_vars = st.session_state['global_process_vars']
        
        if st.session_state['df_weld'].empty or not global_vars:
            st.error("🚨 모델 학습 실패: 필수 데이터(3번 파일)가 로드되지 않았습니다. 또는 유효한 컬럼이 없습니다.")
            st.session_state['model'] = None
            st.session_state['scaler'] = None
            return

        model, scaler = train_model(st.session_state['df_weld'])
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler

        if model is not None:
            st.success(f"✅ AI 모델 학습 및 로드 완료! (총 **{len(global_vars)}개** 변수 사용)")
            
            # 🌟 UI 초기값 및 나머지 변수의 기본값 설정 로직
            if st.session_state['df_init'] is not None and not st.session_state['df_init'].empty:
                init_row = st.session_state['df_init'].iloc[0].to_dict()
            else:
                # 초기 조건 파일이 없으면 학습 데이터의 첫 번째 행을 초기값으로 사용
                if not st.session_state['df_weld'].empty:
                     init_row = st.session_state['df_weld'][global_vars].iloc[0].to_dict()
                else:
                    init_row = {}
            
            # 🌟 핵심 수정: UI_INPUT_VARS 컬럼이 데이터에 있는지 확인 및 슬라이더 초기값 설정
            missing_ui_vars = []
            for var in UI_INPUT_VARS:
                if var in global_vars:
                    # 데이터에 컬럼이 있는 경우: 초기 조건으로 슬라이더 값 업데이트
                    if var in init_row:
                        try:
                            st.session_state[f'input_{var}'] = float(init_row[var])
                        except (ValueError, TypeError):
                            st.warning(f"⚠️ 초기 조건/학습 데이터의 '{var}' 값이 유효한 숫자가 아닙니다. 기본값을 유지합니다.")
                else:
                    missing_ui_vars.append(var)
                    
            if missing_ui_vars:
                st.warning(f"⚠️ 경고: 다음 UI 필수 변수가 데이터 컬럼에 없습니다: **{', '.join(missing_ui_vars)}**. 해당 슬라이더는 표시되지 않거나 기본값으로 고정됩니다.")

            # 🌟 GLOBAL_PROCESS_VARS 중 UI_INPUT_VARS에 없는 변수의 초기값을 저장
            default_init_values = {}
            for var in global_vars:
                if var not in UI_INPUT_VARS:
                    # 초기 조건 파일/학습 데이터의 값 사용, 없으면 0.0 (nan 처리)
                    default_init_values[var] = init_row.get(var, 0.0) 
                
            st.session_state['default_init_values'] = default_init_values
            st.success("✅ UI 초기 조건 및 전체 변수 기본값 설정 완료")


    st.button("🚀 파일 로드 및 AI 모델 학습 시작", on_click=load_and_train_model)

    st.markdown("---")
    st.header("ℹ️ 시스템 상태 확인")

    if st.session_state['model'] is not None:
        st.success("모델 상태: 학습 완료")
        st.write(f"사용된 총 공정 변수: **{len(st.session_state['global_process_vars'])}개**")
        
        total_count = len(st.session_state['df_weld'])
        defect_count = st.session_state['df_weld'][TARGET_VAR].sum()
        defect_rate = (defect_count / total_count) * 100 if total_count > 0 else 0
        
        st.write(f"총 데이터 개수: **{total_count}개**")
        st.write(f"불량 비율(Y=1): **{defect_rate:.1f}%**")
        
        if defect_rate == 0:
            st.warning("⚠️ 경고: 학습 데이터에 불량(1) 샘플이 0개입니다. 정확한 진단이 어려울 수 있습니다.")
    else:
        st.warning("모델 상태: 학습 필요")
        

# -----------------
# 메인 페이지 (진단 UI)
# -----------------
st.title("Weld Line AI 통합 진단 및 최적화 시스템")

tab1, tab2 = st.tabs(["탭 1. 진단 및 최적 공정 조건 제시", "탭 2. 모델 및 데이터 확인"])

with tab1:
    st.header("A. 현재 공정 조건 입력")
    
    # 🌟 수정: UI_INPUT_VARS를 기반으로 슬라이더 동적 생성
    cols = st.columns(3)
    input_vars = {} # UI_INPUT_VARS (6개) 값만 저장
    
    # UI_INPUT_VARS를 3개씩 2줄로 배치
    for i, var in enumerate(UI_INPUT_VARS):
        if var in st.session_state['global_process_vars']: # 데이터에 해당 컬럼이 있는 경우에만 슬라이더 표시
            spec = UI_VARS_SPEC[var]
            
            with cols[i % 3]:
                input_vars[var] = st.slider(
                    f'{spec["label"]} (**{var}**) [{spec["unit"]}]', 
                    spec['min'], 
                    spec['max'], 
                    value=st.session_state.get(f'input_{var}', spec['default']), # 세션 상태 값 사용
                    step=spec['step'], 
                    key=f'slider_{var}',
                    on_change=lambda: st.session_state.update({'current_risk_display': None, 'optimization_result': None})
                )
        elif var in UI_INPUT_VARS:
             # 데이터에 없는 필수 변수는 메시지로 대체 (선택적으로)
             with cols[i % 3]:
                 st.info(f'**{var}** 컬럼이 데이터에 없어 슬라이더를 표시할 수 없습니다.')
                 input_vars[var] = st.session_state.get(f'input_{var}', UI_VARS_SPEC[var]['default']) # 기본값은 유지

    st.markdown("---")
    
    # B. 전문가의 정성적/정량적 노하우 입력 (동일)
    st.header("B. 전문가의 정성적/정량적 노하우 입력")
    
    # 1. 전문가 확신 수준
    st.subheader("1. 전문가 확신 수준")
    st.write("노하우 반영도 (%)") 
    expert_confidence = st.slider(
        '노하우 반영도 (%)', 
        0.0, 
        100.0, 
        value=st.session_state['conf_level'], 
        step=5.0, 
        label_visibility="collapsed",
        key='expert_confidence_slider',
        on_change=update_influence_factor 
    )
    st.markdown('<div style="margin-top: -20px; font-size: 12px; color: grey;">(0%는 노하우 미반영, 100%는 노하우를 제약 조건으로 강력히 적용)</div>', unsafe_allow_html=True)
    
    # V_Inj 변수명 검증
    V_INJ_VAR = 'V_Inj'
    T_MOLD_VAR = 'T_Mold'
    V_INJ_VAR_EXISTS = V_INJ_VAR in st.session_state['global_process_vars']
    T_MOLD_VAR_EXISTS = T_MOLD_VAR in st.session_state['global_process_vars']
    
    
    # 2. 사출 속도 (V_Inj)
    st.subheader(f"2. 사출 속도 ({V_INJ_VAR})")
    if V_INJ_VAR_EXISTS:
        col_v_qual, col_v_intent, col_v_quant, col_v_delta = st.columns(4)
        with col_v_qual:
             v_inj_qual_apply = st.checkbox('정성적 노하우 적용', value=st.session_state['v_inj_qual_apply'], key='v_inj_qual_apply_chk', on_change=lambda: st.session_state.update({'optimization_result': None}))
             st.session_state['v_inj_qual_apply'] = v_inj_qual_apply
        with col_v_intent:
            v_inj_intent = st.selectbox('V_Inj 조절 의도', ['Keep_Constant', 'Increase', 'Decrease'], index=['Keep_Constant', 'Increase', 'Decrease'].index(st.session_state['v_inj_qual_intent']), disabled=not v_inj_qual_apply, key='intent_v_inj_selectbox', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['v_inj_qual_intent'] = v_inj_intent
        with col_v_quant:
            v_inj_quant_apply = st.checkbox('정량적 노하우 적용', value=st.session_state['v_inj_quant_apply'], key='v_inj_quant_apply_chk', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['v_inj_quant_apply'] = v_inj_quant_apply
        with col_v_delta:
            st.write('V_Inj 노하우 변화율 (%)')
            v_inj_quant_percent = st.slider('V_Inj 변화율', 0.0, 100.0, value=st.session_state['v_inj_quant_percent'], step=1.0, label_visibility="collapsed", disabled=not v_inj_quant_apply, key='v_inj_quant_percent_slider', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['v_inj_quant_percent'] = v_inj_quant_percent
    else:
        st.info(f"데이터에 **{V_INJ_VAR}** 컬럼이 없어 노하우를 적용할 수 없습니다.")
        v_inj_intent = 'Keep_Constant'
        v_inj_quant_percent = 0.0
        v_inj_quant_apply = False

    
    # 3. 금형 온도 (T_Mold)
    st.subheader(f"3. 금형 온도 ({T_MOLD_VAR})")
    if T_MOLD_VAR_EXISTS:
        col_t_qual, col_t_intent, col_t_quant, col_t_delta = st.columns(4)
        with col_t_qual:
            t_mold_qual_apply = st.checkbox('정성적 노하우 적용', value=st.session_state['t_mold_qual_apply'], key='t_mold_qual_apply_chk', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['t_mold_qual_apply'] = t_mold_qual_apply
        with col_t_intent:
            t_mold_intent = st.selectbox('T_Mold 조절 의도', ['Keep_Constant', 'Increase', 'Decrease'], index=['Keep_Constant', 'Increase', 'Decrease'].index(st.session_state['t_mold_qual_intent']), disabled=not t_mold_qual_apply, key='intent_t_mold_selectbox', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['t_mold_qual_intent'] = t_mold_intent
        with col_t_quant:
            t_mold_quant_apply = st.checkbox('정량적 노하우 적용', value=st.session_state['t_mold_quant_apply'], key='t_mold_quant_apply_chk', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['t_mold_quant_apply'] = t_mold_quant_apply
        with col_t_delta:
            st.write('T_Mold 노하우 변화율 (%)')
            t_mold_quant_percent = st.slider('T_Mold 변화율', 0.0, 100.0, value=st.session_state['t_mold_quant_percent'], step=1.0, label_visibility="collapsed", disabled=not t_mold_quant_apply, key='t_mold_quant_percent_slider', on_change=lambda: st.session_state.update({'optimization_result': None}))
            st.session_state['t_mold_quant_percent'] = t_mold_quant_percent
    else:
        st.info(f"데이터에 **{T_MOLD_VAR}** 컬럼이 없어 노하우를 적용할 수 없습니다.")
        t_mold_intent = 'Keep_Constant'
        t_mold_quant_percent = 0.0
        t_mold_quant_apply = False


    st.markdown("---")

    # C. 진단 실행 및 결과
    st.header("C. 진단 실행 및 결과")

    st.write("노하우 영향 계수")
    st.slider(
        '노하우 영향 계수 (0.0~1.0)', 
        0.0, 
        1.0, 
        value=st.session_state['influence_factor_display_val'], 
        step=0.01, 
        label_visibility="collapsed",
        disabled=True
    )
    
    st.markdown("---")


    # -----------------
    # 진단 실행 및 최적화 함수 (핵심 수정)
    # -----------------
    
    def run_diagnosis_callback(input_vars):
        """진단 버튼 클릭 시 현재 조건 진단 실행"""
        model = st.session_state['model']
        global_vars = st.session_state['global_process_vars']
        
        if model is None:
            st.session_state['current_risk_display'] = "🚨 모델이 학습되지 않았습니다."
            return

        # 🌟 수정: 전체 변수 딕셔너리를 구성 (UI 입력값 + 초기 조건 파일의 나머지 값)
        full_input_data = {}
        
        # 1. UI 입력값 (6개) - 실제 슬라이더 값이 있는 변수만 사용
        for var in input_vars:
            full_input_data[var] = input_vars[var]

        # 2. 나머지 변수 (UI 입력창이 없는 변수)
        for var in global_vars:
            if var not in full_input_data:
                # 초기 조건 파일에서 미리 저장해둔 기본값 사용 (load_and_train_model에서 저장됨)
                full_input_data[var] = st.session_state['default_init_values'].get(var, 0.0) 
        
        # 전체 변수 순서에 맞게 Series로 변환하여 예측 함수에 전달
        full_input_series = pd.Series(full_input_data, index=global_vars)

        current_risk = predict_weld_risk(model, st.session_state['scaler'], full_input_series)
        st.session_state['current_risk_display'] = current_risk
        st.session_state['optimization_result'] = None 

    
    def run_optimization_callback(input_vars, v_inj_intent, v_inj_quant_percent, v_inj_quant_apply, t_mold_intent, t_mold_quant_percent, t_mold_quant_apply):
        """최적 공정 조건 제시 버튼 클릭 시 실행 (다변수 처리)"""
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        global_vars = st.session_state['global_process_vars']
        
        if model is None:
            st.session_state['optimization_result'] = {"success": False, "message": "모델이 학습되지 않았습니다."}
            return
            
        # V_Inj, T_Mold의 존재 여부 다시 확인
        if V_INJ_VAR not in global_vars:
            v_inj_quant_apply = False
        if T_MOLD_VAR not in global_vars:
            t_mold_quant_apply = False
        
        # 🌟 1. 초기 조건 (X0): 전체 변수를 포함
        x0_dict = {}
        for var in global_vars:
             if var in input_vars:
                 x0_dict[var] = input_vars[var] # UI 슬라이더 값
             else:
                 x0_dict[var] = st.session_state['default_init_values'].get(var, 0.0) # 나머지 변수의 초기값
        
        # 최적화 시작점 (x0)을 전체 변수 순서에 맞게 배열로 변환
        x0 = np.array([x0_dict[var] for var in global_vars])
        
        # 🌟 2. 경계 조건 (Bounds): 전체 변수를 포함
        bounds_list = []
        for var in global_vars:
            if var in GLOBAL_BOUNDS:
                # UI 입력 변수는 정의된 범위 사용
                bounds_list.append(GLOBAL_BOUNDS[var])
            else:
                # 🌟 나머지 변수는 현재 초기값으로 고정 (제약 조건)
                # min=max 이므로 최적화 중 변하지 않음
                init_val = x0_dict[var]
                bounds_list.append((init_val, init_val))

        
        # 🌟 3. 목적 함수 (Objective Function)
        def objective(x):
            # x는 numpy array (global_vars 순서)
            # 예측을 위해 pandas Series로 변환
            x_series = pd.Series(x, index=global_vars)
            
            # 예측 함수는 확률을 반환 (목표: 확률을 최소화 = 1-확률을 최대화)
            risk = predict_weld_risk(model, scaler, x_series)
            
            # 최소화 함수이므로, Risk를 최소화하는 방향으로 설정 (목표)
            return risk

        # 🌟 4. 제약 조건 (Constraints): 노하우 반영
        constraints = []
        influence_factor = st.session_state['influence_factor_display_val']
        
        # V_Inj 인덱스 및 현재 값
        v_inj_index = global_vars.index(V_INJ_VAR) if V_INJ_VAR_EXISTS else -1
        v_inj_current = x0_dict.get(V_INJ_VAR, 0.0)

        # T_Mold 인덱스 및 현재 값
        t_mold_index = global_vars.index(T_MOLD_VAR) if T_MOLD_VAR_EXISTS else -1
        t_mold_current = x0_dict.get(T_MOLD_VAR, 0.0)

        
        # 4-1. V_Inj 정성적 노하우 (Keep_Constant, Increase, Decrease)
        if v_inj_index != -1 and st.session_state['v_inj_qual_apply']:
            tol = 1e-6 # float 비교 허용 오차
            
            if v_inj_intent == 'Keep_Constant':
                # 노하우 반영 계수만큼 현재 값 주변으로 제약
                delta = v_inj_current * (1 - influence_factor) * 0.1 # 예: 0% 시 10% 허용, 100% 시 0% 허용
                lower = v_inj_current - delta
                upper = v_inj_current + delta
                
                # 제약 조건 함수: g(x) >= 0 형태 (x[v_inj_index] - lower >= 0, upper - x[v_inj_index] >= 0)
                constraints.append({'type': 'ineq', 'fun': lambda x: x[v_inj_index] - lower})
                constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[v_inj_index]})
                
            elif v_inj_intent == 'Increase':
                # 현재 값보다 커야 함 (노하우 반영 계수가 클수록 더 커져야 함)
                lower = v_inj_current + influence_factor * 0.1 # 최소 증가량
                
                # 제약 조건 함수: x[v_inj_index] - lower >= 0
                constraints.append({'type': 'ineq', 'fun': lambda x: x[v_inj_index] - lower})

            elif v_inj_intent == 'Decrease':
                # 현재 값보다 작아야 함 (노하우 반영 계수가 클수록 더 작아져야 함)
                upper = v_inj_current - influence_factor * 0.1 # 최대 증가량
                
                # 제약 조건 함수: upper - x[v_inj_index] >= 0
                constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[v_inj_index]})
                
        # 4-2. T_Mold 정성적 노하우 (Keep_Constant, Increase, Decrease)
        if t_mold_index != -1 and st.session_state['t_mold_qual_apply']:
            # V_Inj와 동일한 정성적 노하우 적용 로직 (변수만 T_Mold로 변경)
            tol = 1e-6 
            
            if t_mold_intent == 'Keep_Constant':
                delta = t_mold_current * (1 - influence_factor) * 0.1 
                lower = t_mold_current - delta
                upper = t_mold_current + delta
                constraints.append({'type': 'ineq', 'fun': lambda x: x[t_mold_index] - lower})
                constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[t_mold_index]})
                
            elif t_mold_intent == 'Increase':
                lower = t_mold_current + influence_factor * 0.1
                constraints.append({'type': 'ineq', 'fun': lambda x: x[t_mold_index] - lower})

            elif t_mold_intent == 'Decrease':
                upper = t_mold_current - influence_factor * 0.1
                constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[t_mold_index]})


        # 4-3. V_Inj 정량적 노하우 (특정 퍼센트 범위)
        if v_inj_index != -1 and v_inj_quant_apply and v_inj_quant_percent > 0:
            percent_factor = v_inj_quant_percent / 100.0
            
            # 노하우 반영 계수가 클수록 제약의 폭이 좁아짐
            current_range = GLOBAL_BOUNDS[V_INJ_VAR][1] - GLOBAL_BOUNDS[V_INJ_VAR][0]
            
            # 허용 폭 계산: 노하우 변화율(%)과 전체 변동 폭, 그리고 노하우 반영 계수 고려
            # 반영 계수가 100%이면, 노하우 변화율(%) 만큼만 움직일 수 있음
            max_delta = v_inj_current * percent_factor
            
            lower = v_inj_current - max_delta * (1 - (1 - influence_factor)) # 100% 반영 시 max_delta만큼만 움직임
            upper = v_inj_current + max_delta * (1 - (1 - influence_factor))
            
            # 최종 경계는 GLOBAL_BOUNDS를 벗어나지 않도록 클리핑
            lower = max(lower, GLOBAL_BOUNDS[V_INJ_VAR][0])
            upper = min(upper, GLOBAL_BOUNDS[V_INJ_VAR][1])
            
            # 제약 조건 추가
            constraints.append({'type': 'ineq', 'fun': lambda x: x[v_inj_index] - lower})
            constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[v_inj_index]})


        # 4-4. T_Mold 정량적 노하우 (특정 퍼센트 범위)
        if t_mold_index != -1 and t_mold_quant_apply and t_mold_quant_percent > 0:
            percent_factor = t_mold_quant_percent / 100.0
            
            max_delta = t_mold_current * percent_factor
            
            lower = t_mold_current - max_delta * (1 - (1 - influence_factor)) 
            upper = t_mold_current + max_delta * (1 - (1 - influence_factor))
            
            lower = max(lower, GLOBAL_BOUNDS[T_MOLD_VAR][0])
            upper = min(upper, GLOBAL_BOUNDS[T_MOLD_VAR][1])
            
            constraints.append({'type': 'ineq', 'fun': lambda x: x[t_mold_index] - lower})
            constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[t_mold_index]})


        # 🌟 5. 최적화 실행
        try:
            # SLSQP는 bounded constraint와 inequality constraint를 모두 지원
            result = minimize(objective, x0, method='SLSQP', bounds=bounds_list, constraints=constraints)
            
            # 6. 결과 처리
            if result.success:
                optimized_vars = pd.Series(result.x, index=global_vars).to_dict()
                optimized_risk = objective(result.x) # 최종 최적화된 리스크 값
                
                st.session_state['optimization_result'] = {
                    "success": True,
                    "optimized_vars": {var: f"{val:.4f}" for var, val in optimized_vars.items()},
                    "optimized_risk": optimized_risk,
                    "message": "최적 공정 조건 제시 성공"
                }
            else:
                st.session_state['optimization_result'] = {
                    "success": False,
                    "message": f"최적화 실패: {result.message}"
                }
                
        except Exception as e:
            st.session_state['optimization_result'] = {
                "success": False,
                "message": f"최적화 중 예외 발생: {e}"
            }


    # 진단 및 최적화 버튼
    col_diag, col_opt = st.columns(2)
    with col_diag:
        if st.button("진단 실행", type="primary", use_container_width=True, disabled=st.session_state['model'] is None):
            run_diagnosis_callback(input_vars)
            
    with col_opt:
        if st.button("최적 공정 조건 제시", type="secondary", use_container_width=True, disabled=st.session_state['model'] is None):
            run_optimization_callback(
                input_vars, 
                st.session_state['v_inj_qual_intent'], 
                st.session_state['v_inj_quant_percent'], 
                st.session_state['v_inj_quant_apply'],
                st.session_state['t_mold_qual_intent'], 
                st.session_state['t_mold_quant_percent'], 
                st.session_state['t_mold_quant_apply']
            )

    
    # 결과 표시
    st.subheader("진단 및 최적화 결과")
    
    if st.session_state['current_risk_display'] is not None:
        risk = st.session_state['current_risk_display'] * 100
        
        risk_color = "red" if risk >= 50 else ("orange" if risk >= 20 else "green")
        risk_text = "높음 (불량 가능성 높음)" if risk >= 50 else ("보통 (주의 필요)" if risk >= 20 else "낮음 (양호)")
        
        st.metric(
            label="현재 조건 Weld Line 불량 위험도", 
            value=f"{risk:.2f}%", 
            delta_color="off"
        )
        st.markdown(f"**진단 결과**: <span style='color:{risk_color}; font-weight:bold;'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown("---")

    if st.session_state['optimization_result']:
        result = st.session_state['optimization_result']
        if result['success']:
            st.success(f"✅ 최적 공정 조건 제시 완료! (예상 위험도: **{result['optimized_risk'] * 100:.2f}%**)")
            
            optimized_df = pd.DataFrame(
                {
                    'Variable': list(result['optimized_vars'].keys()),
                    'Optimized Value': [f"{float(v):.2f}" for v in result['optimized_vars'].values()],
                    'Initial Value (Input)': [
                        f"{input_vars.get(var, st.session_state['default_init_values'].get(var, 0.0)):.2f}" 
                        for var in result['optimized_vars'].keys()
                    ]
                }
            )
            
            # UI 입력 변수(6개)만 먼저 표시
            ui_optimized_df = optimized_df[optimized_df['Variable'].isin(UI_INPUT_VARS)]
            other_optimized_df = optimized_df[~optimized_df['Variable'].isin(UI_INPUT_VARS)]
            
            st.write("**최적화된 UI 입력 변수 (슬라이더 변수)**")
            st.dataframe(ui_optimized_df.reset_index(drop=True), hide_index=True, use_container_width=True)

            with st.expander("숨겨진 나머지 공정 변수 보기 (고정값)"):
                st.dataframe(other_optimized_df.reset_index(drop=True), hide_index=True, use_container_width=True)
            
        else:
            st.error(f"❌ 최적화 실패: {result['message']}")


with tab2:
    st.header("모델 및 데이터 상세 정보")

    st.subheader("1. 데이터 컬럼 상세")
    if not st.session_state['df_weld'].empty:
        global_vars_df = pd.DataFrame({
            'Index': range(len(st.session_state['global_process_vars'])),
            'Column Name (Variable)': st.session_state['global_process_vars'],
            'Role': ['UI Input' if var in UI_INPUT_VARS else 'Fixed Process Variable' for var in st.session_state['global_process_vars']]
        })
        st.dataframe(global_vars_df, hide_index=True)
    else:
        st.info("데이터를 로드하고 모델을 학습시켜야 컬럼 정보가 표시됩니다.")
        
    st.subheader("2. 학습 데이터 미리보기 (10행)")
    if not st.session_state['df_weld'].empty:
        st.dataframe(st.session_state['df_weld'].head(10))
    else:
        st.info("학습 데이터가 로드되지 않았습니다.")
        
    st.subheader("3. 모델 계수 (회귀 분석)")
    model = st.session_state['model']
    if model:
        coeffs = pd.DataFrame({
            'Variable': st.session_state['global_process_vars'],
            'Coefficient (Scaled)': model.coef_[0]
        }).sort_values(by='Coefficient (Scaled)', key=abs, ascending=False)
        st.dataframe(coeffs, hide_index=True)
    else:
        st.info("모델이 학습되지 않았습니다.")

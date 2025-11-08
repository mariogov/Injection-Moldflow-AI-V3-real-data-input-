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

# 🌟 UI 입력 변수 정의 (초기 조건 파일에 따라 동적으로 설정됨)
# 임시 기본값 및 경계: (최적화에 필요)
GLOBAL_BOUNDS = {
    'T_Melt': (200.0, 300.0), 'V_Inj': (1.0, 10.0), 'P_Pack': (50.0, 100.0), 
    'T_Mold': (30.0, 80.0), 'Meter': (180.0, 200.0), 'VP_Switch_Pos': (10.0, 20.0)
}
# 실제 UI에 표시될 변수 목록
if 'ui_display_vars' not in st.session_state:
    st.session_state['ui_display_vars'] = [] 

# 🌟 전체 공정 변수 리스트는 학습 데이터에서 동적으로 결정됩니다.
GLOBAL_PROCESS_VARS = [] 

# 종속 변수 정의 (Y 변수)
TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5


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
if 'global_process_vars' not in st.session_state:
    st.session_state['global_process_vars'] = GLOBAL_PROCESS_VARS
if 'default_init_values' not in st.session_state:
    st.session_state['default_init_values'] = {}
    
if 'current_risk_display' not in st.session_state:
    st.session_state['current_risk_display'] = None
if 'optimization_result' not in st.session_state:
    st.session_state['optimization_result'] = None

if 'conf_level' not in st.session_state:
    st.session_state['conf_level'] = 75.0
if 'influence_factor_display_val' not in st.session_state:
    st.session_state['influence_factor_display_val'] = st.session_state['conf_level'] / 100.0

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
# 콜백 함수: 전문가 확신 수준 변경 시 영향 계수 업데이트 (동일)
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
# 1. 데이터 로드 및 전처리 함수
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
                    df = pd.read_csv(uploaded_file, sep=',')
                except Exception:
                    uploaded_file.seek(0)
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

    df_combined = pd.concat(valid_dataframes, ignore_index=True)
    df_combined.dropna(subset=[TARGET_VAR], inplace=True) 
    
    if TARGET_VAR not in df_combined.columns:
        st.error(f"⚠️ 데이터에 필수 타겟 컬럼('{TARGET_VAR}')이 누락되었습니다. 컬럼 이름을 확인해 주세요.")
        st.session_state['global_process_vars'] = []
        return pd.DataFrame()
        
    all_vars = [col for col in df_combined.columns if col != TARGET_VAR]
    st.session_state['global_process_vars'] = all_vars
    
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    
    required_cols = all_vars + [TARGET_VAR]
    df_processed = df_combined[required_cols].copy()
    
    df_processed.fillna(0, inplace=True)
    
    return df_processed

def train_model(df):
    """데이터를 사용하여 로지스틱 회귀 모델을 학습하고 스케일러를 저장합니다."""
    if df.empty or not st.session_state['global_process_vars']:
        return None, None
        
    global_vars = st.session_state['global_process_vars']
    
    # 🌟🌟🌟 수정된 로직: 타겟 변수 및 독립 변수 추출 🌟🌟🌟
    X = df[global_vars]
    Y = df[TARGET_VAR]

    # 🚨 ValueError 방지를 위한 핵심 체크
    if Y.nunique() < 2:
        st.error(f"🚨 모델 학습 실패: 타겟 변수('{TARGET_VAR}')에 **두 개 이상의 클래스** (0과 1)가 존재해야 합니다. 현재 데이터셋에는 클래스가 {Y.nunique()}개만 존재합니다. 학습 데이터 파일을 확인해 주세요.")
        return None, None
    # -----------------------------------------------------
    
    # NaN/Inf 값 체크는 MinMaxScaler/fit_transform에서 처리되지만, 명시적으로 방어 코드를 남기지 않습니다.
    # df.fillna(0)에서 대부분의 NaN은 처리되었음.

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
        input_df = pd.DataFrame([input_data_series.to_dict()], columns=global_vars)
    elif isinstance(input_data_series, pd.DataFrame) and len(input_data_series) == 1:
        input_df = input_data_series[global_vars]
    else:
        st.error("⚠️ 예측 입력 데이터 형식이 올바르지 않습니다.")
        return 0.5
    
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
        "1. UI 초기 조건 (initial_condition.xlsx) [필수]", type=['xlsx', 'csv'], key="init_file"
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
        
        if st.session_state['df_init'] is None or st.session_state['df_init'].empty:
            st.error("🚨 UI 초기 조건 파일(1번)이 로드되지 않았습니다. UI 구성을 할 수 없습니다.")
            return
            
        df_weld_processed = process_weld_data(st.session_state['df_virtual'], st.session_state['df_real'])
        st.session_state['df_weld'] = df_weld_processed
        
        global_vars = st.session_state['global_process_vars']
        
        if st.session_state['df_weld'].empty or not global_vars:
            st.error("🚨 모델 학습 실패: 필수 데이터(3번 파일)가 로드되지 않았습니다. 또는 유효한 컬럼이 없습니다.")
            st.session_state['model'] = None
            st.session_state['scaler'] = None
            return

        # 수정된 train_model 호출
        model, scaler = train_model(st.session_state['df_weld'])
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler

        if model is not None:
            st.success(f"✅ AI 모델 학습 및 로드 완료! (총 **{len(global_vars)}개** 변수 사용)")
            
            all_init_cols = list(st.session_state['df_init'].columns)
            ui_vars_to_display = [col for col in all_init_cols if col != TARGET_VAR]
            st.session_state['ui_display_vars'] = ui_vars_to_display
            
            if TARGET_VAR in all_init_cols:
                st.info(f"💡 UI 초기 조건 파일에 포함된 **{TARGET_VAR}** 컬럼은 종속 변수이므로 UI 입력창에서 **자동으로 제외**되었습니다.")
            
            init_row = st.session_state['df_init'].iloc[0].to_dict()
            
            for var in st.session_state['ui_display_vars']:
                if var in init_row:
                    try:
                        st.session_state[f'input_{var}'] = float(init_row[var])
                    except (ValueError, TypeError):
                        st.session_state[f'input_{var}'] = 0.0 
                        st.warning(f"⚠️ 초기 조건 파일의 '{var}' 값이 유효한 숫자가 아닙니다. 0으로 설정합니다.")
                else:
                    st.session_state[f'input_{var}'] = 0.0

            default_init_values = {}
            for var in global_vars:
                if var not in st.session_state['ui_display_vars']:
                    default_init_values[var] = df_weld_processed.iloc[0].get(var, 0.0)
                
            st.session_state['default_init_values'] = default_init_values
            st.success("✅ UI 입력 변수 및 전체 변수 기본값 설정 완료")
        else:
             # train_model에서 오류가 발생했으므로 여기서 추가 메시지를 표시할 필요 없음
             pass


    st.button("🚀 파일 로드 및 AI 모델 학습 시작", on_click=load_and_train_model)

    st.markdown("---")
    st.header("ℹ️ 시스템 상태 확인")

    if st.session_state['model'] is not None:
        st.success("모델 상태: 학습 완료")
        st.write(f"사용된 총 공정 변수: **{len(st.session_state['global_process_vars'])}개**")
        st.write(f"UI 입력 변수: **{len(st.session_state['ui_display_vars'])}개**")
        
        total_count = len(st.session_state['df_weld'])
        defect_count = st.session_state['df_weld'][TARGET_VAR].sum()
        defect_rate = (defect_count / total_count) * 100 if total_count > 0 else 0
        
        st.write(f"총 학습 데이터 개수: **{total_count}개**")
        st.write(f"불량 비율(Y=1): **{defect_rate:.1f}%**")
    else:
        st.warning("모델 상태: 학습 필요")
        

# -----------------
# 메인 페이지 (진단 UI)
# -----------------
st.title("Weld Line AI 통합 진단 및 최적화 시스템")

tab1, tab2 = st.tabs(["탭 1. 진단 및 최적 공정 조건 제시", "탭 2. 모델 및 데이터 확인"])

with tab1:
    st.header("A. 현재 공정 조건 입력")
    
    ui_vars = st.session_state['ui_display_vars']
    
    if not ui_vars and st.session_state['df_init'] is not None:
        st.info("💡 UI 입력 변수 목록은 로드된 초기 조건 파일에서 **Y_Weld를 제외한** 컬럼으로 구성됩니다.")
    elif not ui_vars:
        st.warning("⚠️ UI 입력 변수가 설정되지 않았습니다. **UI 초기 조건 파일(1번)**을 로드하고 **모델 학습**을 먼저 실행해 주세요.")
        input_vars = {}
    else:
        cols = st.columns(3)
        input_vars = {}
        
        for i, var in enumerate(ui_vars):
            min_val = GLOBAL_BOUNDS.get(var, (0.0, 300.0))[0] 
            max_val = GLOBAL_BOUNDS.get(var, (0.0, 300.0))[1] 
            default_val = st.session_state.get(f'input_{var}', (min_val + max_val) / 2)
            
            if min_val == max_val:
                 min_val -= 1.0
                 max_val += 1.0

            with cols[i % 3]:
                input_vars[var] = st.slider(
                    f'{var}', 
                    min_val, 
                    max_val, 
                    value=default_val, 
                    step=(max_val - min_val) / 100.0,
                    key=f'slider_{var}',
                    on_change=lambda: st.session_state.update({'current_risk_display': None, 'optimization_result': None})
                )
                
    st.markdown("---")
    
    # B. 전문가의 정성적/정량적 노하우 입력
    st.header("B. 전문가의 정성적/정량적 노하우 입력")
    
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
    
    V_INJ_VAR = 'V_Inj'
    T_MOLD_VAR = 'T_Mold'
    V_INJ_VAR_EXISTS_IN_UI = V_INJ_VAR in st.session_state['ui_display_vars']
    T_MOLD_VAR_EXISTS_IN_UI = T_MOLD_VAR in st.session_state['ui_display_vars']
    
    
    # 2. 사출 속도 (V_Inj)
    st.subheader(f"2. 사출 속도 ({V_INJ_VAR})")
    if V_INJ_VAR_EXISTS_IN_UI:
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
        st.info(f"UI 입력 변수 목록에 **{V_INJ_VAR}** 컬럼이 없어 노하우를 적용할 수 없습니다.")
        v_inj_intent = 'Keep_Constant'
        v_inj_quant_percent = 0.0
        v_inj_quant_apply = False

    
    # 3. 금형 온도 (T_Mold)
    st.subheader(f"3. 금형 온도 ({T_MOLD_VAR})")
    if T_MOLD_VAR_EXISTS_IN_UI:
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
        st.info(f"UI 입력 변수 목록에 **{T_MOLD_VAR}** 컬럼이 없어 노하우를 적용할 수 없습니다.")
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
    # 진단 실행 및 최적화 함수
    # -----------------
    
    def run_diagnosis_callback(input_vars):
        """진단 버튼 클릭 시 현재 조건 진단 실행"""
        model = st.session_state['model']
        global_vars = st.session_state['global_process_vars']
        
        if model is None:
            st.session_state['current_risk_display'] = "🚨 모델이 학습되지 않았습니다."
            return

        full_input_data = {}
        
        for var in input_vars:
            full_input_data[var] = input_vars[var]

        for var in global_vars:
            if var not in full_input_data:
                full_input_data[var] = st.session_state['default_init_values'].get(var, 0.0) 
        
        full_input_series = pd.Series(full_input_data, index=global_vars)

        current_risk = predict_weld_risk(model, st.session_state['scaler'], full_input_series)
        st.session_state['current_risk_display'] = current_risk
        st.session_state['optimization_result'] = None 

    
    def run_optimization_callback(input_vars, v_inj_intent, v_inj_quant_percent, v_inj_qual_apply, t_mold_intent, t_mold_quant_percent, t_mold_qual_apply):
        """최적 공정 조건 제시 버튼 클릭 시 실행 (다변수 처리)"""
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        global_vars = st.session_state['global_process_vars']
        
        if model is None:
            st.session_state['optimization_result'] = {"success": False, "message": "모델이 학습되지 않았습니다."}
            return
            
        v_inj_exists_in_global = V_INJ_VAR in global_vars
        t_mold_exists_in_global = T_MOLD_VAR in global_vars
        
        # 🌟 1. 초기 조건 (X0): 전체 변수를 포함
        x0_dict = {}
        for var in global_vars:
             if var in input_vars:
                 x0_dict[var] = input_vars[var] 
             else:
                 x0_dict[var] = st.session_state['default_init_values'].get(var, 0.0) 
        
        x0 = np.array([x0_dict[var] for var in global_vars])
        
        # 🌟 2. 경계 조건 (Bounds): 전체 변수를 포함
        bounds_list = []
        for var in global_vars:
            if var in input_vars:
                bounds_list.append(GLOBAL_BOUNDS.get(var, (0.0, 300.0)))
            else:
                init_val = x0_dict[var]
                bounds_list.append((init_val, init_val))

        
        # 3. 목적 함수 (Objective Function)
        def objective(x):
            x_series = pd.Series(x, index=global_vars)
            risk = predict_weld_risk(model, scaler, x_series)
            return risk

        # 4. 제약 조건 (Constraints): 노하우 반영
        constraints = []
        influence_factor = st.session_state['influence_factor_display_val']
        
        v_inj_index = global_vars.index(V_INJ_VAR) if v_inj_exists_in_global else -1
        v_inj_current = x0_dict.get(V_INJ_VAR, 0.0)

        t_mold_index = global_vars.index(T_MOLD_VAR) if t_mold_exists_in_global else -1
        t_mold_current = x0_dict.get(T_MOLD_VAR, 0.0)

        
        # 4-1. V_Inj 정성적 노하우
        if v_inj_index != -1 and v_inj_qual_apply and V_INJ_VAR in input_vars: 
            if v_inj_intent == 'Keep_Constant':
                delta = v_inj_current * (1 - influence_factor) * 0.1
                lower = v_inj_current - delta
                upper = v_inj_current + delta
                constraints.append({'type': 'ineq', 'fun': lambda x: x[v_inj_index] - lower})
                constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[v_inj_index]})
                
            elif v_inj_intent == 'Increase':
                lower = v_inj_current + influence_factor * 0.1 
                constraints.append({'type': 'ineq', 'fun': lambda x: x[v_inj_index] - lower})

            elif v_inj_intent == 'Decrease':
                upper = v_inj_current - influence_factor * 0.1
                constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[v_inj_index]})

        # 4-2. T_Mold 정성적 노하우
        if t_mold_index != -1 and t_mold_qual_apply and T_MOLD_VAR in input_vars:
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


        # 4-3. V_Inj 정량적 노하우
        if v_inj_index != -1 and v_inj_quant_apply and v_inj_quant_percent > 0 and V_INJ_VAR in input_vars:
            percent_factor = v_inj_quant_percent / 100.0
            
            bounds_v_inj = GLOBAL_BOUNDS.get(V_INJ_VAR, (0.0, 300.0))
            max_delta = v_inj_current * percent_factor
            
            lower = v_inj_current - max_delta * (1 - (1 - influence_factor)) 
            upper = v_inj_current + max_delta * (1 - (1 - influence_factor))
            
            lower = max(lower, bounds_v_inj[0])
            upper = min(upper, bounds_v_inj[1])
            
            constraints.append({'type': 'ineq', 'fun': lambda x: x[v_inj_index] - lower})
            constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[v_inj_index]})


        # 4-4. T_Mold 정량적 노하우
        if t_mold_index != -1 and t_mold_quant_apply and t_mold_quant_percent > 0 and T_MOLD_VAR in input_vars:
            percent_factor = t_mold_quant_percent / 100.0
            
            bounds_t_mold = GLOBAL_BOUNDS.get(T_MOLD_VAR, (0.0, 300.0))
            max_delta = t_mold_current * percent_factor
            
            lower = t_mold_current - max_delta * (1 - (1 - influence_factor)) 
            upper = t_mold_current + max_delta * (1 - (1 - influence_factor))
            
            lower = max(lower, bounds_t_mold[0])
            upper = min(upper, bounds_t_mold[1])
            
            constraints.append({'type': 'ineq', 'fun': lambda x: x[t_mold_index] - lower})
            constraints.append({'type': 'ineq', 'fun': lambda x: upper - x[t_mold_index]})


        # 5. 최적화 실행
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds_list, constraints=constraints)
            
            # 6. 결과 처리
            if result.success:
                optimized_vars = pd.Series(result.x, index=global_vars).to_dict()
                optimized_risk = objective(result.x)
                
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
        if st.button("진단 실행", type="primary", use_container_width=True, disabled=st.session_state['model'] is None or not ui_vars):
            run_diagnosis_callback(input_vars)
            
    with col_opt:
        if st.button("최적 공정 조건 제시", type="secondary", use_container_width=True, disabled=st.session_state['model'] is None or not ui_vars):
            run_optimization_callback(
                input_vars, 
                st.session_state['v_inj_qual_intent'], 
                st.session_state['v_inj_quant_percent'], 
                st.session_state['v_inj_qual_apply'],
                st.session_state['t_mold_qual_intent'], 
                st.session_state['t_mold_quant_percent'], 
                st.session_state['t_mold_qual_apply']
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
            
            optimized_vars_data = []
            for var, opt_val_str in result['optimized_vars'].items():
                opt_val = float(opt_val_str)
                init_val = input_vars.get(var) if var in input_vars else st.session_state['default_init_values'].get(var, 0.0)
                
                optimized_vars_data.append({
                    'Variable': var,
                    'Optimized Value': f"{opt_val:.2f}",
                    'Initial Value (Input)': f"{init_val:.2f}"
                })
                
            optimized_df = pd.DataFrame(optimized_vars_data)
            
            ui_optimized_df = optimized_df[optimized_df['Variable'].isin(st.session_state['ui_display_vars'])]
            other_optimized_df = optimized_df[~optimized_df['Variable'].isin(st.session_state['ui_display_vars'])]
            
            st.write("**최적화된 UI 입력 변수**")
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
            'Role': ['UI Input' if var in st.session_state['ui_display_vars'] else 'Fixed Process Variable' for var in st.session_state['global_process_vars']]
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

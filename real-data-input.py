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
# 기본 경계 조건 (사용자가 수정할 수 있는 초기값)
GLOBAL_BOUNDS = {
    'T_Melt': (200.0, 300.0), 'V_Inj': (1.0, 10.0), 'P_Pack': (50.0, 100.0), 
    'T_Mold': (30.0, 80.0), 'Meter': (180.0, 200.0), 'VP_Switch_Pos': (10.0, 20.0)
}
if 'ui_display_vars' not in st.session_state:
    st.session_state['ui_display_vars'] = [] 

GLOBAL_PROCESS_VARS = [] 
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

# 모든 변수의 노하우 설정 및 경계값 저장
if 'knowhow_settings' not in st.session_state:
    st.session_state['knowhow_settings'] = {}
    
if 'knowhow_temp_storage' not in st.session_state:
    st.session_state['knowhow_temp_storage'] = {}
    
# UI 입력 변수의 현재 경계값을 저장 (사용자 수정 가능)
if 'global_bounds' not in st.session_state:
    st.session_state['global_bounds'] = GLOBAL_BOUNDS.copy() 


# -------------------------------------------------------------
# 콜백 함수: 전문가 확신 수준 변경 시 영향 계수 업데이트
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
    
    # 0.5 이상이면 불량(1), 미만이면 양호(0)로 이진 분류
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
    
    X = df[global_vars]
    Y = df[TARGET_VAR]

    # 🚨 ValueError 방지를 위한 핵심 체크
    if Y.nunique() < 2:
        st.error(f"🚨 모델 학습 실패: 타겟 변수('{TARGET_VAR}')에 **두 개 이상의 클래스** (0과 1)가 존재해야 합니다. 현재 데이터셋에는 클래스가 {Y.nunique()}개만 존재합니다. 학습 데이터 파일을 확인해 주세요.")
        return None, None
    
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
            
            # UI 입력 변수 초기값 설정
            for var in st.session_state['ui_display_vars']:
                # 경계값 초기화 시 global_bounds도 업데이트
                min_val = GLOBAL_BOUNDS.get(var, (0.0, 300.0))[0] 
                max_val = GLOBAL_BOUNDS.get(var, (0.0, 300.0))[1] 
                st.session_state['global_bounds'][var] = (min_val, max_val)
                
                if var in init_row:
                    try:
                        # 초기 조건 파일의 값 사용
                        st.session_state[f'input_{var}'] = float(init_row[var])
                    except (ValueError, TypeError):
                        # 값 오류 시 경계 중앙값 사용
                        st.session_state[f'input_{var}'] = (min_val + max_val) / 2
                        st.warning(f"⚠️ 초기 조건 파일의 '{var}' 값이 유효한 숫자가 아닙니다. 중앙값으로 설정합니다.")
                else:
                    st.session_state[f'input_{var}'] = (min_val + max_val) / 2

            # 나머지 변수 초기값 저장
            default_init_values = {}
            for var in global_vars:
                if var not in st.session_state['ui_display_vars']:
                    default_init_values[var] = df_weld_processed.iloc[0].get(var, 0.0)
                
            st.session_state['default_init_values'] = default_init_values
            st.success("✅ UI 입력 변수 및 전체 변수 기본값 설정 완료")
            
            # 모든 UI 변수에 대한 노하우 설정 초기화/업데이트
            knowhow_settings = st.session_state.get('knowhow_settings', {})
            for var in st.session_state['ui_display_vars']:
                if var not in knowhow_settings:
                    # 기본 설정
                    knowhow_settings[var] = {
                        'qual_apply': False, 
                        'qual_intent': 'Keep_Constant', 
                        'quant_apply': False, 
                        'quant_percent': 0.0
                    }
            st.session_state['knowhow_settings'] = knowhow_settings
            st.session_state['knowhow_temp_storage'] = {} # 임시 저장소 초기화


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
    st.header("A. 현재 공정 조건 입력 (숫자 직접 입력)")
    
    ui_vars = st.session_state['ui_display_vars']
    
    if not ui_vars and st.session_state['df_init'] is not None:
        st.info("💡 UI 입력 변수 목록은 로드된 초기 조건 파일에서 **Y_Weld를 제외한** 컬럼으로 구성됩니다.")
    elif not ui_vars:
        st.warning("⚠️ UI 입력 변수가 설정되지 않았습니다. **UI 초기 조건 파일(1번)**을 로드하고 **모델 학습**을 먼저 실행해 주세요.")
        input_vars = {}
    else:
        cols = st.columns(3)
        input_vars = {}
        
        # st.number_input 사용
        for i, var in enumerate(ui_vars):
            
            # 세션 상태의 global_bounds에서 경계값을 가져옴
            min_val, max_val = st.session_state['global_bounds'].get(var, GLOBAL_BOUNDS.get(var, (0.0, 300.0)))
            
            default_val = st.session_state.get(f'input_{var}', (min_val + max_val) / 2)
            
            # min_val == max_val 인 경우를 위한 안전 장치
            if min_val >= max_val:
                 min_val_safe = min_val - 1.0
                 max_val_safe = max_val + 1.0
            else:
                 min_val_safe = min_val
                 max_val_safe = max_val

            with cols[i % 3]:
                # st.number_input: 직접 숫자 입력 창
                input_vars[var] = st.number_input(
                    f'{var}', 
                    min_value=min_val_safe, 
                    max_value=max_val_safe, 
                    value=default_val, 
                    step=(max_val_safe - min_val_safe) / 100.0, 
                    format="%.2f", # 소수점 둘째 자리까지 표시
                    key=f'input_{var}', # 세션 상태 키를 그대로 사용하여 값 저장
                    on_change=lambda: st.session_state.update({'current_risk_display': None, 'optimization_result': None})
                )
                
    st.markdown("---")
    
    # B. 전문가 노하우 입력 (점진적 노출 적용)
    st.header("B. 전문가 노하우 입력")
    
    # 1. 전문가 확신 수준 (노하우 반영 강도)
    st.subheader("1. 전문가 확신 수준 (강도)")
    st.write("노하우 반영 강도 (%)") 
    st.slider(
        '노하우 반영도 (%)', 
        0.0, 
        100.0, 
        value=st.session_state['conf_level'], 
        step=5.0, 
        label_visibility="collapsed",
        key='expert_confidence_slider',
        on_change=update_influence_factor 
    )
    st.markdown('<div style="margin-top: -20px; font-size: 12px; color: grey;">(확신 수준이 높을수록 노하우 방향(Increase/Decrease)을 강력하게 따릅니다.)</div>', unsafe_allow_html=True)
    
    # 변수 조절 의도 및 조건값 선택 버튼 (점진적 노출)
    st.subheader("2. 공정 변수별 조절 의도 및 조건 (방향 및 경계)")
    
    if not ui_vars:
        st.info("UI 입력 변수가 없습니다. **모델 학습**을 먼저 실행해 주세요.")
    else:
        
        cols = st.columns(3)
        
        for i, var in enumerate(ui_vars):
            
            settings = st.session_state['knowhow_temp_storage'].get(var) or \
                       st.session_state['knowhow_settings'].get(var, {'qual_intent': 'Keep_Constant'}) 
            
            current_intent = settings.get('qual_intent', 'Keep_Constant')
            
            # 현재 경계값 (input의 min/max 값)
            current_bounds = st.session_state['global_bounds'].get(var, GLOBAL_BOUNDS.get(var, (0.0, 300.0)))
            
            with cols[i % 3]:
                # 1. 조절 의도 Selectbox (메인 UI)
                intent = st.selectbox(
                    f'{var} 조절 의도', 
                    ['Keep_Constant', 'Increase', 'Decrease'], 
                    index=['Keep_Constant', 'Increase', 'Decrease'].index(current_intent),
                    key=f'intent_{var}_selectbox',
                    on_change=lambda: st.session_state.update({'optimization_result': None})
                )
                
                # 2. 조건값 선택 버튼 및 Expander (점진적 노출)
                with st.expander("🛠️ **조건값 상/하한 설정**"):
                    
                    # 슬라이더의 전체 범위를 설정 (안전을 위해 GLOBAL_BOUNDS를 기준으로)
                    default_min_safe = GLOBAL_BOUNDS.get(var, (0.0, 300.0))[0]
                    default_max_safe = GLOBAL_BOUNDS.get(var, (0.0, 300.0))[1]
                    
                    # 현재 설정된 경계값을 슬라이더의 기본값으로 사용
                    new_min, new_max = st.slider(
                        f'{var} 경계 범위',
                        default_min_safe, 
                        default_max_safe, 
                        value=current_bounds,
                        step=(default_max_safe - default_min_safe) / 100.0,
                        key=f'bounds_slider_{var}',
                        on_change=lambda: st.session_state.update({'optimization_result': None})
                    )
                    
                    # 버튼 클릭 시 세션 상태 업데이트 함수
                    def update_bounds(v, min_v, max_v):
                        # 슬라이더에서 설정된 min/max 값을 global_bounds에 반영
                        st.session_state['global_bounds'][v] = (min_v, max_v)
                        # UI 입력창의 중앙값으로 재설정하여 즉시 반영
                        st.session_state[f'input_{v}'] = (min_v + max_v) / 2 
                        st.session_state['optimization_result'] = None
                        
                    # 경계 변경 확정 버튼
                    if st.button(f'**{var}** 경계 조건 적용 및 UI 반영', key=f'apply_bounds_{var}', use_container_width=True):
                        update_bounds(var, new_min, new_max)
                        st.success(f"✅ **{var}**의 조건 범위가 [{new_min:.2f}, {new_max:.2f}]로 업데이트되었고, UI 입력창에 반영되었습니다.")


                # 3. 노하우 설정값 임시 저장
                st.session_state['knowhow_temp_storage'][var] = {
                    'qual_intent': intent,
                    'qual_apply': (intent != 'Keep_Constant'), 
                    'quant_apply': False, 
                    'quant_percent': 0.0 
                }
        
        # 루프 종료 후 최종적으로 knowhow_settings 업데이트
        st.session_state['knowhow_settings'].update(st.session_state['knowhow_temp_storage'])


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
        
        # 1. UI 입력값 (number_input 값)
        for var in input_vars:
            full_input_data[var] = input_vars[var]

        # 2. 나머지 변수 (UI 입력창이 없는 변수 - 고정값)
        for var in global_vars:
            if var not in full_input_data:
                full_input_data[var] = st.session_state['default_init_values'].get(var, 0.0) 
        
        full_input_series = pd.Series(full_input_data, index=global_vars)

        current_risk = predict_weld_risk(model, st.session_state['scaler'], full_input_series)
        st.session_state['current_risk_display'] = current_risk
        st.session_state['optimization_result'] = None 

    
    def run_optimization_callback(input_vars):
        """최적 공정 조건 제시 버튼 클릭 시 실행 (다변수 처리)"""
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        global_vars = st.session_state['global_process_vars']
        
        if model is None:
            st.session_state['optimization_result'] = {"success": False, "message": "모델이 학습되지 않았습니다."}
            return
            
        # 1. 초기 조건 (X0): 전체 변수를 포함
        x0_dict = {}
        for var in global_vars:
             if var in input_vars:
                 x0_dict[var] = input_vars[var] 
             else:
                 x0_dict[var] = st.session_state['default_init_values'].get(var, 0.0) 
        
        x0 = np.array([x0_dict[var] for var in global_vars])
        
        # 2. 경계 조건 (Bounds): 전체 변수를 포함
        bounds_list = []
        for var in global_vars:
            if var in input_vars:
                # UI 입력 변수는 세션 상태의 global_bounds 사용
                bounds_list.append(st.session_state['global_bounds'].get(var, GLOBAL_BOUNDS.get(var, (0.0, 300.0))))
            else:
                # 나머지 변수는 현재 초기값으로 고정 (제약 조건)
                init_val = x0_dict[var]
                bounds_list.append((init_val, init_val))

        
        # 3. 목적 함수 (Objective Function)
        def objective(x):
            x_series = pd.Series(x, index=global_vars)
            risk = predict_weld_risk(model, scaler, x_series)
            return risk

        # 4. 제약 조건 (Constraints): 노하우 반영 (단순화된 정성적 노하우만)
        constraints = []
        influence_factor = st.session_state['influence_factor_display_val']
        
        # UI 입력 변수 목록을 순회하며 노하우 제약 조건 생성
        for var in st.session_state['ui_display_vars']:
            
            if var not in global_vars: continue 
                 
            index = global_vars.index(var)
            current_value = x0_dict.get(var, 0.0)
            settings = st.session_state['knowhow_settings'].get(var, {})
            
            intent = settings.get('qual_intent')
            
            # 경계값은 세션 상태의 global_bounds에서 가져옵니다.
            bounds = st.session_state['global_bounds'].get(var, GLOBAL_BOUNDS.get(var, (0.0, 300.0)))

            # 4-1. 정성적 노하우 (Keep_Constant, Increase, Decrease)
            
            if intent == 'Keep_Constant':
                # Keep_Constant: 현재 값 근처에 제약. (노하우 확신 수준이 1에 가까울수록 더 좁게 제약)
                range_span = bounds[1] - bounds[0] 
                max_delta = range_span * 0.01 
                
                # 확신 수준이 100%(1.0)이면 변동 폭 0으로 고정
                delta = max_delta * (1 - influence_factor) 
                
                lower = max(current_value - delta, bounds[0])
                upper = min(current_value + delta, bounds[1])
                
                constraints.append({'type': 'ineq', 'fun': lambda x, i=index, l=lower: x[i] - l})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=index, u=upper: u - x[i]})
                
            elif intent == 'Increase':
                # Increase: 현재 값보다 일정 수준 이상으로 제약 (영향 계수만큼 강하게)
                increase_base = (bounds[1] - bounds[0]) * 0.05
                lower_limit = current_value + influence_factor * increase_base
                
                lower = min(lower_limit, bounds[1]) # 상한 경계를 넘지 않도록
                
                constraints.append({'type': 'ineq', 'fun': lambda x, i=index, l=lower: x[i] - l})

            elif intent == 'Decrease':
                # Decrease: 현재 값보다 일정 수준 이하로 제약 (영향 계수만큼 강하게)
                decrease_base = (bounds[1] - bounds[0]) * 0.05
                upper_limit = current_value - influence_factor * decrease_base
                
                upper = max(upper_limit, bounds[0]) # 하한 경계를 넘지 않도록
                
                constraints.append({'type': 'ineq', 'fun': lambda x, i=index, u=upper: u - x[i]})

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
                    "initial_vars": x0_dict, # 초기값 저장
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
    # ------------------------------------------------------------------


    # 진단 및 최적화 버튼
    col_diag, col_opt = st.columns(2)
    with col_diag:
        if st.button("진단 실행", type="primary", use_container_width=True, disabled=st.session_state['model'] is None or not ui_vars):
            run_diagnosis_callback(input_vars)
            
    with col_opt:
        if st.button("최적 공정 조건 제시", type="secondary", use_container_width=True, disabled=st.session_state['model'] is None or not ui_vars):
            run_optimization_callback(input_vars)

    
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
            
            # 🌟🌟🌟 추가된 부분: 최소 불량 위험 확률 🌟🌟🌟
            st.markdown("#### ✅ 최적화 결과 요약")
            col_risk, col_message = st.columns([1, 2])
            with col_risk:
                st.metric(
                    label="최소 불량 위험 확률",
                    value=f"{result['optimized_risk'] * 100:.2f}%",
                    delta_color="off"
                )
            with col_message:
                st.info(f"💡 최적화 성공 메시지: {result['message']}")
            st.markdown("---")
            
            # 🌟🌟🌟 추가된 부분: 최적화 요약 테이블 🌟🌟🌟
            optimized_vars_data = []
            for var, opt_val_str in result['optimized_vars'].items():
                opt_val = float(opt_val_str)
                init_val = result['initial_vars'].get(var, 0.0) # run_optimization_callback에서 저장된 초기값 사용
                
                # 변화량 계산
                if init_val != 0.0:
                    percent_change = ((opt_val - init_val) / init_val) * 100
                else:
                    percent_change = 0.0 if opt_val == 0.0 else np.nan # 초기값이 0일 경우 예외 처리
                    
                # 최적화 방향 결정
                direction = ""
                if np.isnan(percent_change):
                    direction = "N/A"
                elif opt_val > init_val:
                    direction = "Increase ▲"
                elif opt_val < init_val:
                    direction = "Decrease ▼"
                else:
                    direction = "Keep"

                optimized_vars_data.append({
                    'Variable': var,
                    'Initial Value (Input)': f"{init_val:.2f}",
                    'Optimized Value': f"{opt_val:.2f}",
                    'Optimization Direction': direction,
                    'Change (%)': f"{percent_change:.2f}%" if not np.isnan(percent_change) else 'N/A'
                })
                
            optimized_df = pd.DataFrame(optimized_vars_data)
            
            # UI 입력 변수와 나머지 변수 분리
            ui_optimized_df = optimized_df[optimized_df['Variable'].isin(st.session_state['ui_display_vars'])].reset_index(drop=True)
            other_optimized_df = optimized_df[~optimized_df['Variable'].isin(st.session_state['ui_display_vars'])].reset_index(drop=True)
            
            st.markdown("#### 🚀 최적 공정 조건 상세 (UI 입력 변수)")
            st.dataframe(ui_optimized_df, hide_index=True, use_container_width=True)

            with st.expander("숨겨진 나머지 공정 변수 보기 (고정값)"):
                st.dataframe(other_optimized_df, hide_index=True, use_container_width=True)
            
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

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# [중요] st.set_page_config은 항상 최상단에 위치해야 합니다.
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

# == [UI 개선] CSS: 지표 카드 및 테이블 가독성 강화 ==
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] * { color: red !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"] * { color: blue !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"] * { color: green !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 2.2rem !important; }
thead tr th { font-size: 18px !important; color: black !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("## Test-Constrained-RL-ColdStart: S&P 500 Performance")

# == 🛠 사이드바: 파라미터 제어 ==
st.sidebar.markdown("### ⚙️ System Parameters")
env = SP500Environment()
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, 100)
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.03, st)

# 시드 및 자동 반복 설정
base_seed = st.sidebar.number_input("Base Random Seed", value=2026, step=1)
auto_runs = st.sidebar.number_input("Auto Run Count", min_value=1, value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 RL Hyperparameters (Logic: STATIC)")
lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.5, 0.01, step=0.001)
# 박사님 요청 반영: 감마 기본값 0.98
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.98, step=0.01)
eps = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.1, step=0.05)

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

agent_raw = RecommendationAgent(env, use_constraints=False, lr=lr, gamma=gamma, eps=eps)
agent_static = RecommendationAgent(env, use_constraints=True, lr=lr, gamma=gamma, eps=eps)

# == 📈 메인 차트 플레이스홀더 ==
fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL</b>', line=dict(color='red', width=2)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC</b>', line=dict(color='blue', width=2)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>S&P 500 (SPY)</b>', line=dict(color='green', dash='dot')))

chart_view = st.empty()
chart_view.plotly_chart(fig_main, use_container_width=True)

col1, col2, col3 = st.columns(3)
m_u, m_s, m_b = col1.empty(), col2.empty(), col3.empty()

st.markdown("---")
# [에러 해결 포인트] 분석 영역의 구조를 미리 정의합니다.
analysis_header = st.empty()
col_tbl, col_bar = st.columns([1.2, 1])
tbl_view = col_tbl.empty()
bar_view = col_bar.empty()

def style_df(val):
    if isinstance(val, (int, float)):
        return f'color: {"red" if val < 0 else "black"}; font-weight: bold;'
    return 'color: black; font-weight: bold;'

if st.button("Run Evaluation"):
    for run in range(auto_runs):
        trial_idx = len(st.session_state.trial_history)
        current_seed = base_seed + trial_idx
        np.random.seed(current_seed)
        st.toast(f"Trial {trial_idx + 1} (Seed: {current_seed}) - Progress {run + 1}/{auto_runs}")

        h_u, h_s, h_b, steps = [0], [0], [0], [0]
        log_data = []

        for i in range(20, 20 + episodes):
            ticker_u, _, r_u = agent_raw.select_action(current_step=i)
            ticker_s, _, r_s = agent_static.select_action(current_step=i)
            
            sc, sn = float(env.data['SPY'].iloc[i]), float(env.data['SPY'].iloc[i+1])
            r_b = ((sn - sc) / sc) * 100 if sc > 0 else 0.0
                
            h_u.append(h_u[-1] + r_u); h_s.append(h_s[-1] + r_s); h_b.append(h_b[-1] + r_b)
            current_day = i - 19; steps.append(current_day) 
            log_data.append({"Day": current_day, "Vanilla Pick": ticker_u, "Vanilla Return(%)": r_u, "STATIC Pick (Ours)": ticker_s, "STATIC Return(%)": r_s})
            
            # 실시간 차트 업데이트 (안전한 업데이트를 위해 고유 키 부여 생략 가능하나 placeholder 이용)
            fig_main.data[0].x = steps; fig_main.data[0].y = h_u
            fig_main.data[1].x = steps; fig_main.data[1].y = h_s
            fig_main.data[2].x = steps; fig_main.data[2].y = h_b
            chart_view.plotly_chart(fig_main, use_container_width=True)
            
            m_u.metric(label="Unconstrained Return", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
            m_s.metric(label=f"STATIC Return - Bought: {ticker_s}", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
            m_b.metric(label="S&P 500 Index (SPY)", value=f"{h_b[-1]:.2f}%", delta=f"{r_b:.2f}%")
            
            if speed > 0: time.sleep(speed)

        st.session_state.trial_history.append({
            "Trial": trial_idx + 1, "Seed": current_seed, 
            "Vanilla Final (%)": h_u[-1], "STATIC Final (%)": h_s[-1], "SPY Final (%)": h_b[-1]
        })

        # [에러 해결 포인트] 미리 만들어둔 placeholder에 데이터만 갈아끼웁니다.
        analysis_header.markdown("#### Agent Decision Analysis")
        df_log = pd.DataFrame(log_data).set_index("Day")
        tbl_view.dataframe(df_log.style.map(style_df).format("{:.2f}"), height=350, use_container_width=True)
        
        fig_bar = px.bar(df_log['STATIC Pick (Ours)'].value_counts().reset_index(), x='STATIC Pick (Ours)', y='count', 
                         title="<b>Safe-Asset Selection Frequency</b>", color='count', color_continuous_scale='Blues')
        bar_view.plotly_chart(fig_bar.update_layout(plot_bgcolor='white', height=350), use_container_width=True)

# == 📊 하단 통계 섹션 ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### Trial History: Statistical Analysis (Alpha Performance)")
    df_h = pd.DataFrame(st.session_state.trial_history)
    
    v_mean, v_max, v_min = df_h['Vanilla Final (%)'].mean(), df_h['Vanilla Final (%)'].max(), df_h['Vanilla Final (%)'].min()
    s_mean, s_max, s_min = df_h['STATIC Final (%)'].mean(), df_h['STATIC Final (%)'].max(), df_h['STATIC Final (%)'].min()
    v_std = df_h['Vanilla Final (%)'].std() if len(df_h) > 1 else 0.0
    s_std = df_h['STATIC Final (%)'].std() if len(df_h) > 1 else 0.0
    avg_spy = df_h['SPY Final (%)'].mean()
    
    st.success(f"시장 평균 대비 **Alpha 기대치(Expected Value)**: STATIC **{s_mean - avg_spy:.2f}%p** | Vanilla **{v_mean - avg_spy:.2f}%p**")

    # 추이 그래프
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Vanilla Final (%)'], mode='lines+markers', name='Vanilla', line=dict(color='red')))
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['STATIC Final (%)'], mode='lines+markers', name='STATIC', line=dict(color='blue')))
    fig_trend.update_layout(title="<b>Trial-by-Trial Return Progression & Stability</b>", plot_bgcolor='white', height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    col_box, col_tbl_h = st.columns([2, 1])
    with col_box:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df_h['Vanilla Final (%)'], x0=1.0, name='Vanilla RL', line=dict(color='red', width=3), boxmean=True, width=0.5))
        fig_box.add_trace(go.Box(y=df_h['STATIC Final (%)'], x0=2.25, name='STATIC RL (Ours)', line=dict(color='blue', width=3), boxmean=True, width=0.5))
        fig_box.add_hline(y=avg_spy, line_dash="dot", line_color="green", annotation_text=f"S&P 500: {avg_spy:.2f}%")
        fig_box.update_layout(title="<b>Return Distribution across Trials</b>", plot_bgcolor='white', height=550)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col_tbl_h:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;'>
            <h4 style='color: black; font-weight: 900;'>📊 통계 요약</h4>
            <ul style='font-size: 15px;'>
                <li><b style='color:red;'>Vanilla 평균:</b> {v_mean:.2f}% (σ={v_std:.2f}%)</li>
                <li><b style='color:blue;'>STATIC 평균:</b> {s_mean:.2f}% (σ={s_std:.2f}%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_h.set_index("Trial").style.format("{:.2f}"), height=320, use_container_width=True)
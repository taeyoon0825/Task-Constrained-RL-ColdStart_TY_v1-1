import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# == [UI 개선] 메트릭 라벨 색상 CSS ==
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetricLabel"] { color: red !important; font-weight: bold !important; font-size: 1.1rem !important; }
div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetricLabel"] { color: blue !important; font-weight: bold !important; font-size: 1.1rem !important; }
div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetricLabel"] { color: green !important; font-weight: bold !important; font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Test-Constrained-RL", layout="wide")
st.markdown("## >> Test-Constrained-RL-ColdStart: S&P 500 Performance")

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

with st.spinner('>> 실시간 S&P 500 데이터를 분석 중입니다...'):
    env = SP500Environment()
    agent_raw = RecommendationAgent(env, use_constraints=False)
    agent_static = RecommendationAgent(env, use_constraints=True)

st.sidebar.markdown("### >> Test Parameters")
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.05)

# == Plotly 차트: S&P 500 벤치마크 추가 ==
fig = go.Figure()

fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL (Unconstrained)</b>', line=dict(color='red', width=3), marker=dict(symbol='circle-open', size=8, line_width=2)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC (Ours)</b>', line=dict(color='blue', width=3), marker=dict(symbol='square-open', size=8, line_width=2)))
# !! [개선] S&P 500 초록색 점선 추가
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>S&P 500 Index (SPY)</b>', line=dict(color='green', width=3, dash='dash')))

fig.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=28, color='black')),
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=22, color='black'), showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=22, color='black'), showgrid=True, gridcolor='lightgray'),
    legend=dict(font=dict(size=20, color='black'), x=0.01, y=0.99, borderwidth=1, bgcolor='rgba(0,0,0,0)'),
    plot_bgcolor='white', height=500
)
fig.add_hline(y=0, line_width=3, line_color="black", opacity=0.8)

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

# 지표 카드 (SPY 추가)
col1, col2, col3 = st.columns(3)
m_u = col1.empty()
m_s = col2.empty()
m_b = col3.empty()

st.markdown("---")
analysis_view = st.empty()

if st.button(">> Run Evaluation"):
    h_u, h_s, h_b, steps = [0], [0], [0], [0]
    log_data = []

    for i in range(20, 20 + episodes):
        ticker_u, _, r_u = agent_raw.select_action(current_step=i)
        ticker_s, _, r_s = agent_static.select_action(current_step=i)
        
        # SPY 벤치마크 일일 수익률 계산
        spy_curr = float(env.data['SPY'].iloc[i])
        spy_next = float(env.data['SPY'].iloc[i+1])
        r_b = ((spy_next - spy_curr) / spy_curr) * 100 if spy_curr > 0 else 0.0
        
        h_u.append(h_u[-1] + r_u)
        h_s.append(h_s[-1] + r_s)
        h_b.append(h_b[-1] + r_b)
        current_day = i - 19
        steps.append(current_day) 
        
        log_data.append({"Day": current_day, "Vanilla Pick": ticker_u, "Vanilla Return(%)": round(r_u, 2), "STATIC Pick (Ours)": ticker_s, "STATIC Return(%)": round(r_s, 2)})
        
        fig.data[0].x = steps; fig.data[0].y = h_u
        fig.data[1].x = steps; fig.data[1].y = h_s
        fig.data[2].x = steps; fig.data[2].y = h_b
        chart_view.plotly_chart(fig, use_container_width=True)
        
        m_u.metric(label="Unconstrained Return", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
        m_s.metric(label=f"STATIC Return - Bought: {ticker_s}", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
        m_b.metric(label="S&P 500 Index (SPY)", value=f"{h_b[-1]:.2f}%", delta=f"{r_b:.2f}%")
        
        time.sleep(speed)

    # Trial History 기록
    st.session_state.trial_history.append({
        "Trial": len(st.session_state.trial_history) + 1,
        "Vanilla Final (%)": round(h_u[-1], 2),
        "STATIC Final (%)": round(h_s[-1], 2),
        "SPY Final (%)": round(h_b[-1], 2)
    })

    df_log = pd.DataFrame(log_data)
    with analysis_view.container():
        st.markdown("#### >> Agent Decision Analysis")
        col_tbl, col_bar = st.columns([1.2, 1])
        with col_tbl:
            st.dataframe(df_log.set_index("Day"), height=300, use_container_width=True)
        with col_bar:
            dist_counts = df_log['STATIC Pick (Ours)'].value_counts().reset_index()
            dist_counts.columns = ['Ticker', 'Buy Count']
            fig_bar = px.bar(dist_counts, x='Ticker', y='Buy Count', title="Frequency of Safe-Asset Selection", color='Buy Count', color_continuous_scale='Blues')
            fig_bar.update_layout(plot_bgcolor='white', height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

# == 하단: 통계적 박스 플롯 및 승률 분석 (개선됨) ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### 🏆 Trial History: Statistical Superiority Analysis")
    
    history_df = pd.DataFrame(st.session_state.trial_history)
    
    # 승률 및 평균 계산
    win_count = (history_df['STATIC Final (%)'] > history_df['Vanilla Final (%)']).sum()
    win_rate = (win_count / len(history_df)) * 100
    avg_static = history_df['STATIC Final (%)'].mean()
    avg_vanilla = history_df['Vanilla Final (%)'].mean()
    
    # 강력한 통계 요약 텍스트
    st.success(f"**🔥 누적 승률 (Win Rate):** {win_rate:.1f}% (STATIC 모델이 Vanilla를 이긴 비율) | **평균 수익률:** STATIC **{avg_static:.2f}%** vs Vanilla **{avg_vanilla:.2f}%**")
    
    col_box, col_hist_table = st.columns([2, 1])
    
    with col_box:
        # 지그재그 선 대신 Box Plot으로 분포 시각화
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=history_df['Vanilla Final (%)'], name='Vanilla RL', marker_color='red', boxmean=True))
        fig_box.add_trace(go.Box(y=history_df['STATIC Final (%)'], name='STATIC RL (Ours)', marker_color='blue', boxmean=True))
        
        fig_box.update_layout(title="Return Distribution across Trials", yaxis_title="Final Cumulative Return (%)", plot_bgcolor='white', height=350)
        fig_box.add_hline(y=0, line_width=2, line_color="black", opacity=0.8)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_hist_table:
        st.dataframe(history_df.set_index("Trial"), height=350, use_container_width=True)
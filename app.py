import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# == 대시보드 초기 설정 ==
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

st.markdown("## >> Test-Constrained-RL-ColdStart: S&P 500 Performance")
st.markdown("이 테스트는 STATIC 제약 조건이 하락장과 변동성을 어떻게 방어하는지 측정합니다.")

# == 데이터 환경 및 에이전트 초기화 ==
with st.spinner('>> 실시간 S&P 500 데이터를 분석 중입니다... (약 10초 소요)'):
    env = SP500Environment()
    agent_raw = RecommendationAgent(env, use_constraints=False)
    agent_static = RecommendationAgent(env, use_constraints=True)

# == 차트 및 UI 제어 사이드바 ==
st.sidebar.markdown("### >> Test Parameters")
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.05)

# == Plotly 차트: 마커 및 기준선 추가 (논문 스타일) ==
fig = go.Figure()

# 붉은선: 바닐라 RL (원형 빈 심벌)
fig.add_trace(go.Scatter(
    x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL (Unconstrained)</b>', 
    line=dict(color='red', width=3),
    marker=dict(symbol='circle-open', size=8, line_width=2)
))

# 파란선: STATIC RL (네모 빈 심벌)
fig.add_trace(go.Scatter(
    x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC (Ours)</b>', 
    line=dict(color='blue', width=3),
    marker=dict(symbol='square-open', size=8, line_width=2)
))

fig.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=28, color='black')),
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=22, color='black'), showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=22, color='black'), showgrid=True, gridcolor='lightgray'),
    legend=dict(font=dict(size=20, color='black'), x=0.01, y=0.99, borderwidth=1),
    plot_bgcolor='white', height=500
)

# y=0을 지나는 굵은 검은색 실선 추가
fig.add_hline(y=0, line_width=3, line_color="black", opacity=0.8)

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

# == 실시간 지표 카드 영역 ==
col1, col2 = st.columns(2)
m_u = col1.empty()
m_s = col2.empty()

# == 하단: 의사결정 시각화 영역 (초기엔 비워둠) ==
st.markdown("---")
st.markdown("### >> Agent Decision Analysis")
analysis_view = st.empty()

if st.button(">> Run Evaluation"):
    h_u, h_s, steps = [0], [0], [0]
    
    # 의사결정 로그 기록용 리스트
    log_data = []

    for i in range(20, 20 + episodes):
        # 1. 시뮬레이션 수행
        ticker_u, _, r_u = agent_raw.select_action(current_step=i)
        ticker_s, valid_s, r_s = agent_static.select_action(current_step=i)
        
        # 2. 데이터 누적
        h_u.append(h_u[-1] + r_u)
        h_s.append(h_s[-1] + r_s)
        current_day = i - 19
        steps.append(current_day) 
        
        # 3. 로그 기록
        log_data.append({
            "Day": current_day,
            "Vanilla Pick": ticker_u,
            "Vanilla Return(%)": round(r_u, 2),
            "STATIC Pick (Ours)": ticker_s,
            "STATIC Return(%)": round(r_s, 2)
        })
        
        # 4. 실시간 그래프 업데이트
        fig.data[0].x = steps
        fig.data[0].y = h_u
        fig.data[1].x = steps
        fig.data[1].y = h_s
        chart_view.plotly_chart(fig, use_container_width=True)
        
        # 5. 텍스트 지표 업데이트
        m_u.metric(label="<b>Unconstrained Return</b>", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
        m_s.metric(label=f"<b>STATIC Return - Bought: {ticker_s}</b>", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
        
        time.sleep(speed)

    st.success("== Evaluation Sequence Completed. ==")

    # == 시뮬레이션 종료 후 의사결정 분석 시각화 ==
    df_log = pd.DataFrame(log_data)
    
    with analysis_view.container():
        col3, col4 = st.columns([1.2, 1])
        
        with col3:
            st.markdown("#### 1. Detailed Trading Log")
            st.dataframe(df_log.set_index("Day"), height=350, use_container_width=True)
            
        with col4:
            st.markdown("#### 2. STATIC Portfolio Distribution")
            # STATIC 에이전트가 어떤 종목을 가장 많이 샀는지 빈도 계산
            dist_counts = df_log['STATIC Pick (Ours)'].value_counts().reset_index()
            dist_counts.columns = ['Ticker', 'Buy Count']
            
            fig_bar = px.bar(dist_counts, x='Ticker', y='Buy Count', 
                             title="Frequency of Safe-Asset Selection",
                             color='Buy Count', color_continuous_scale='Blues')
            fig_bar.update_layout(plot_bgcolor='white', height=350)
            st.plotly_chart(fig_bar, use_container_width=True)
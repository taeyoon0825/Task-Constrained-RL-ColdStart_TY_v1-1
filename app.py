import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import config
from agent import SP500Environment, RecommendationAgent

# == 대시보드 초기 설정 ==
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

st.markdown("## >> Test-Constrained-RL-ColdStart: S&P 500 Performance")
st.markdown("이 테스트는 STATIC 제약 조건(20일 이동평균선 기반)이 하락장과 변동성을 어떻게 방어하는지 측정합니다.")

# == 데이터 환경 및 에이전트 초기화 ==
# !! [개선점] yfinance에서 데이터를 받아오는 동안 사용자가 기다릴 수 있도록 로딩 표시기 추가
with st.spinner('>> 실시간 S&P 500 데이터를 분석 중입니다... (약 10초 소요)'):
    env = SP500Environment()
    agent_raw = RecommendationAgent(env, use_constraints=False)
    agent_static = RecommendationAgent(env, use_constraints=True)

# == 차트 및 UI 제어 사이드바 ==
st.sidebar.markdown("### >> Test Parameters")
# 총 거래일수에 맞춰 에피소드 슬라이더의 최대치 자동 조절
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.05)

# == Plotly 차트: 학술 논문 스타일 설정 ==
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>Vanilla RL (Unconstrained)</b>', line=dict(color='red', width=4)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>RL with STATIC (Ours)</b>', line=dict(color='blue', width=4)))

fig.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=28, color='black')),
    xaxis=dict(
        title="<b>Trading Days</b>", 
        titlefont=dict(size=22, color='black'), 
        tickfont=dict(size=18, color='black', family="Arial Black"),
        showgrid=True, gridcolor='lightgray'
    ),
    yaxis=dict(
        title="<b>Total Cumulative Return (%)</b>", 
        titlefont=dict(size=22, color='black'), 
        tickfont=dict(size=18, color='black', family="Arial Black"),
        showgrid=True, gridcolor='lightgray'
    ),
    legend=dict(font=dict(size=20, color='black'), x=0.01, y=0.99, borderwidth=1),
    plot_bgcolor='white',
    height=650
)

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

# == 실시간 지표 카드 영역 ==
col1, col2 = st.columns(2)
m_u = col1.empty()
m_s = col2.empty()

if st.button(">> Run Evaluation"):
    h_u, h_s, steps = [0], [0], [0]
    
    # 20일치(SMA 계산용) 이후부터 지정한 에피소드만큼 시뮬레이션 진행
    for i in range(20, 20 + episodes):
        # 1. 시뮬레이션 수행 (현재 날짜 i를 전달)
        _, _, r_u = agent_raw.select_action(current_step=i)
        ticker_s, valid_s, r_s = agent_static.select_action(current_step=i)
        
        # 2. 데이터 누적
        h_u.append(h_u[-1] + r_u)
        h_s.append(h_s[-1] + r_s)
        steps.append(i - 19) # X축이 1부터 시작하도록 조정
        
        # 3. 실시간 그래프 업데이트
        fig.data[0].x = steps
        fig.data[0].y = h_u
        fig.data[1].x = steps
        fig.data[1].y = h_s
        chart_view.plotly_chart(fig, use_container_width=True)
        
        # 4. 텍스트 지표 업데이트 (퍼센트 단위로 표시)
        m_u.metric(label="<b>Unconstrained Return</b>", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
        m_s.metric(label=f"<b>STATIC Return - Bought: {ticker_s}</b>", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
        
        time.sleep(speed)

    st.success("== Evaluation Sequence Completed. ==")
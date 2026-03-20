import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import (
    KOSPIEnvironment,
    RecommendationAgent,
    QLearningBanditAgent,
    PolicyGradientBanditAgent,
)

# st.set_page_config은 반드시 첫 번째 Streamlit 명령이어야 함
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

# == [UI 개선] CSS: 지표 카드 및 테이블 가독성 강화 ==
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] * { color: #e05050 !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"] * { color: #4a90d9 !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"] * { color: #2ea84a !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 2.2rem !important; }
thead tr th { font-size: 18px !important; color: var(--text-color) !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("## Test-Constrained-RL-ColdStart: KOSPI Performance")

# == 🛠 사이드바: 테스트 및 강화학습 파라미터 제어 ==
st.sidebar.markdown("### System Parameters")
env = KOSPIEnvironment()
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
# [수정됨] Frame Speed 기본값 0.03, step 0.01 반영
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.03, step=0.01)

# == 실험 재현성을 위한 랜덤 시드 설정 ==
base_seed = st.sidebar.number_input("Base Random Seed", value=2026, step=1, help="실험 재현성을 위한 기본 시드입니다. 각 Trial마다 이 값에 +1씩 더해집니다.")

# == 자동 반복 실행 횟수 설정 ==
auto_runs = st.sidebar.number_input("Auto Run Count", min_value=1, value=1, step=1, help="Run Evaluation 버튼 클릭 시 자동으로 반복 실행할 횟수입니다.")

st.sidebar.markdown("---")
st.sidebar.markdown("### RL Hyperparameters (Logic: STATIC)")
lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.5, 0.01, step=0.001)
# [수정됨] Gamma 기본값 0.98 반영
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.98, step=0.01)
st.sidebar.caption("**ε_s**: State·행동 탐색 (Vanilla RL, Q-Learning) · **ε_v**: Value·제약 쪽 탐색 (STATIC RL, Q+STATIC)")
eps_s = st.sidebar.slider("Exploration ε_s (State)", 0.0, 1.0, 0.10, step=0.05, help="상태·행동 공간 탐색률. Vanilla RL 및 Q-Learning(무제약)에 적용.")
eps_v = st.sidebar.slider("Exploration ε_v (Value)", 0.0, 1.0, 0.10, step=0.05, help="제약·가치 쪽 탐색률. STATIC RL 및 Q-Learning+STATIC에 적용.")
eps = (eps_s + eps_v) / 2.0  # Auto Sweep 등 단일 ε 표기용 (평균)

st.sidebar.markdown("---")
st.sidebar.markdown("### 학습형 RL (Q-Learning / Policy Gradient)")
_max_train = max(0, episodes - 1)
training_days = st.sidebar.slider(
    "학습 구간 (일)",
    0,
    _max_train,
    min(40, _max_train),
    help="처음 N일 동안 Q-Learning·Policy Gradient만 업데이트합니다. 이후 동일 기간에 모든 에이전트가 평가됩니다.",
)
pg_lr_scale = st.sidebar.slider("Policy Gradient LR 배율", 0.1, 2.0, 0.5, 0.1, help="PG는 보통 Q보다 작은 학습률이 안정적입니다.")

# == 자동 파라미터 스윕 설정 ==
st.sidebar.markdown("---")
st.sidebar.markdown("### Auto Hyperparameter Sweep")
runs_per_config = st.sidebar.number_input(
    "Runs per Config (Sweep)",
    min_value=1,
    value=5,
    step=1,
    help="각 파라미터 조합당 몇 번씩 시뮬레이션을 반복할지 설정합니다.",
)
start_sweep = st.sidebar.button("Run Auto Sweep (lr → γ → ε)")

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

agent_raw = RecommendationAgent(env, use_constraints=False, lr=lr, gamma=gamma, eps=eps_s)
agent_static = RecommendationAgent(env, use_constraints=True, lr=lr, gamma=gamma, eps=eps_v)

# == 📈 메인 수익률 비교 차트 ==
fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL</b>', line=dict(color='#e05050', width=2), marker=dict(symbol='circle-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC</b>', line=dict(color='#4a90d9', width=2), marker=dict(symbol='square-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Q-Learning</b>', line=dict(color='#ff9800', width=2), marker=dict(symbol='triangle-up-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Q-Learning + STATIC</b>', line=dict(color='#9c27b0', width=2), marker=dict(symbol='hexagon-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Policy Gradient</b>', line=dict(color='#795548', width=2), marker=dict(symbol='star-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>KOSPI Index</b>', line=dict(color='green', width=2, dash='dot'), marker=dict(symbol='diamond-open', size=6)))

fig_main.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (KOSPI)</b>", font=dict(size=28)),
    xaxis=dict(
        title=dict(text="<b>Trading Days</b>", font=dict(size=18)),
        showgrid=True
    ),
    yaxis=dict(
        title=dict(text="<b>Total Cumulative Return (%)</b>", font=dict(size=18)),
        showgrid=True
    ),
    legend=dict(
        font=dict(size=16),
        x=0.01,
        y=0.99,
        bgcolor='rgba(128,128,128,0.15)',
        bordercolor='rgba(128,128,128,0.3)',
        borderwidth=1
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=550,
    margin=dict(t=80, b=80, l=80, r=40)
)
fig_main.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")

chart_view = st.empty()

r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)
m_u, m_s, m_q = r1c1.empty(), r1c2.empty(), r1c3.empty()
m_qs, m_pg, m_b = r2c1.empty(), r2c2.empty(), r2c3.empty()

st.markdown("---")
# 분석 영역의 구조를 미리 정의 (ID 중복 에러 방지)
analysis_header = st.empty()
col_tbl, col_bar = st.columns([1.2, 1])
tbl_view = col_tbl.empty()
bar_view = col_bar.empty()

def style_df(val):
    if isinstance(val, (int, float)):
        if val < 0:
            return 'color: #e05050; font-weight: bold; font-size: 16px;'
        return 'font-weight: bold; font-size: 16px;'
    return 'font-weight: bold; font-size: 16px;'

if 'sweep_results' not in st.session_state:
    st.session_state.sweep_results = None

if st.button("Run Evaluation"):
    for run in range(auto_runs):
        trial_idx = len(st.session_state.trial_history)
        current_seed = base_seed + trial_idx
        np.random.seed(current_seed)
        rng = np.random.default_rng(current_seed)
        st.toast(f"Trial {trial_idx + 1} Started (Seed: {current_seed}) - Run {run + 1}/{auto_runs}")

        # --- 학습형 에이전트: 학습 구간에서만 업데이트 ---
        q_agent = QLearningBanditAgent(env, use_constraints=False, lr=lr, gamma=gamma, epsilon=eps_s)
        qs_agent = QLearningBanditAgent(env, use_constraints=True, lr=lr, gamma=gamma, epsilon=eps_v)
        pg_agent = PolicyGradientBanditAgent(env, use_constraints=False, lr=max(lr * pg_lr_scale, 1e-4))
        q_agent.reset()
        qs_agent.reset()
        pg_agent.reset()

        train_end = min(20 + training_days, len(env.data) - 2)
        for i in range(20, train_end):
            q_agent.train_step(i, rng)
            qs_agent.train_step(i, rng)
            pg_agent.train_step(i, rng)

        # --- 평가 구간: [train_end, 20+episodes) ∩ 유효 인덱스 (i+1 접근 가능) ---
        eval_start = train_end
        eval_limit = min(20 + episodes, len(env.data) - 1)

        h_u, h_s, h_q, h_qs, h_pg, h_b, steps = [0], [0], [0], [0], [0], [0], [0]
        log_data = []
        benchmark_col = env.benchmark

        if eval_start >= eval_limit:
            st.warning("평가할 거래일이 없습니다. Episodes를 늘리거나 학습 구간(일)을 줄이세요.")
            continue

        day_idx = 0
        for i in range(eval_start, eval_limit):
            ticker_u, _, r_u = agent_raw.select_action(current_step=i)
            ticker_s, _, r_s = agent_static.select_action(current_step=i)
            ticker_q, _, r_q = q_agent.select_action(i, rng, greedy=True)
            ticker_qs, _, r_qs = qs_agent.select_action(i, rng, greedy=True)
            ticker_pg, _, r_pg = pg_agent.select_action(i, rng)

            if benchmark_col in env.data.columns:
                sc, sn = float(env.data[benchmark_col].iloc[i]), float(env.data[benchmark_col].iloc[i + 1])
                r_b = ((sn - sc) / sc) * 100 if sc > 0 else 0.0
            else:
                r_b = 0.0

            h_u.append(h_u[-1] + r_u)
            h_s.append(h_s[-1] + r_s)
            h_q.append(h_q[-1] + r_q)
            h_qs.append(h_qs[-1] + r_qs)
            h_pg.append(h_pg[-1] + r_pg)
            h_b.append(h_b[-1] + r_b)
            day_idx += 1
            steps.append(day_idx)
            log_data.append({
                "Day": day_idx,
                "Vanilla Pick": ticker_u,
                "Vanilla Return(%)": r_u,
                "STATIC Pick (Ours)": ticker_s,
                "STATIC Return(%)": r_s,
                "Q-Learn Pick": ticker_q,
                "Q-Learn Return(%)": r_q,
                "Q+STATIC Pick": ticker_qs,
                "Q+STATIC Return(%)": r_qs,
                "PG Pick": ticker_pg,
                "PG Return(%)": r_pg,
            })

            fig_main.data[0].x = steps
            fig_main.data[0].y = h_u
            fig_main.data[1].x = steps
            fig_main.data[1].y = h_s
            fig_main.data[2].x = steps
            fig_main.data[2].y = h_q
            fig_main.data[3].x = steps
            fig_main.data[3].y = h_qs
            fig_main.data[4].x = steps
            fig_main.data[4].y = h_pg
            fig_main.data[5].x = steps
            fig_main.data[5].y = h_b

            m_u.metric(label="Vanilla Return", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
            m_s.metric(label=f"STATIC — {ticker_s}", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
            m_q.metric(label=f"Q-Learning — {ticker_q}", value=f"{h_q[-1]:.2f}%", delta=f"{r_q:.2f}%")
            m_qs.metric(label=f"Q+STATIC — {ticker_qs}", value=f"{h_qs[-1]:.2f}%", delta=f"{r_qs:.2f}%")
            m_pg.metric(label=f"Policy Grad — {ticker_pg}", value=f"{h_pg[-1]:.2f}%", delta=f"{r_pg:.2f}%")
            m_b.metric(label="KOSPI Index", value=f"{h_b[-1]:.2f}%", delta=f"{r_b:.2f}%")

            if speed > 0:
                time.sleep(speed)

        chart_view.plotly_chart(fig_main, use_container_width=True, key=f"main_eval_{trial_idx}_{run}")

        finals = {
            "Vanilla Final (%)": h_u[-1],
            "STATIC Final (%)": h_s[-1],
            "Q-Learn Final (%)": h_q[-1],
            "Q-STATIC Final (%)": h_qs[-1],
            "PG Final (%)": h_pg[-1],
            "KOSPI Final (%)": h_b[-1],
        }
        best_name = max(
            ("Vanilla", finals["Vanilla Final (%)"]),
            ("STATIC", finals["STATIC Final (%)"]),
            ("Q-Learning", finals["Q-Learn Final (%)"]),
            ("Q+STATIC", finals["Q-STATIC Final (%)"]),
            ("Policy Gradient", finals["PG Final (%)"]),
            key=lambda x: x[1],
        )[0]
        st.info(f"이번 Trial 최고 누적 수익: **{best_name}** (학습 {training_days}일 → 평가 {eval_limit - eval_start}일)")

        st.session_state.trial_history.append({
            "Trial": trial_idx + 1,
            "Seed": current_seed,
            **finals,
        })

        analysis_header.markdown("#### Agent Decision Analysis")
        df_log = pd.DataFrame(log_data).set_index("Day")
        num_cols = [
            "Vanilla Return(%)",
            "STATIC Return(%)",
            "Q-Learn Return(%)",
            "Q+STATIC Return(%)",
            "PG Return(%)",
        ]
        styled_df = df_log.style.map(style_df).format("{:.2f}", subset=num_cols)
        tbl_view.dataframe(styled_df, height=350, use_container_width=True)

        fig_bar = px.bar(
            df_log["STATIC Pick (Ours)"].value_counts().reset_index(),
            x="STATIC Pick (Ours)",
            y="count",
            title="<b>Safe-Asset Selection Frequency (STATIC)</b>",
            color="count",
            color_continuous_scale="Blues",
        )
        bar_view.plotly_chart(
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350),
            use_container_width=True,
            key=f"bar_static_{trial_idx}_{run}",
        )

# == 📊 자동 파라미터 스윕 실행 ==
if start_sweep:
    st.markdown("---")
    st.markdown("### Auto Hyperparameter Sweep Results")

    # 1단계: lr 스윕용 후보
    lr_candidates = [0.005, 0.01, 0.02, 0.05]
    # 2단계: gamma 스윕용 후보
    gamma_candidates = [0.90, 0.95, 0.98, 0.99]
    # 3단계: eps 스윕용 후보
    eps_candidates = [0.0, 0.05, 0.1, 0.2]

    # Episodes는 설명한 대로 100으로 고정 (데이터가 허용하는 범위 내에서)
    episodes_sweep = min(100, max_episodes)

    results = []

    # == (1) lr 스윕: gamma, eps는 현재 슬라이더 값 사용 ==
    for lr_val in lr_candidates:
        static_returns = []
        for run_idx in range(runs_per_config):
            # 재현성 있는 시드를 위해 base_seed 활용
            np.random.seed(base_seed + run_idx)
            agent_u = RecommendationAgent(env, use_constraints=False, lr=lr_val, gamma=gamma, eps=eps_s)
            agent_s = RecommendationAgent(env, use_constraints=True, lr=lr_val, gamma=gamma, eps=eps_v)

            h_s = 0.0
            benchmark_col = env.benchmark

            for i in range(20, 20 + episodes_sweep):
                _, _, r_u = agent_u.select_action(current_step=i)
                _, _, r_s = agent_s.select_action(current_step=i)

                # STATIC 성과만 누적 (lr 비교 목적)
                h_s += r_s

            static_returns.append(h_s)

        results.append({
            "Phase": "lr",
            "lr": lr_val,
            "gamma": gamma,
            "eps_s": eps_s,
            "eps_v": eps_v,
            "STATIC Mean (%)": np.mean(static_returns),
            "STATIC Median (%)": float(np.median(static_returns)),
            "STATIC Std (%)": float(np.std(static_returns)) if len(static_returns) > 1 else 0.0,
        })

    # lr 스윕에서 가장 좋은 lr 선택
    df_lr = pd.DataFrame([r for r in results if r["Phase"] == "lr"])
    best_lr = df_lr.sort_values("STATIC Mean (%)", ascending=False).iloc[0]["lr"]

    # == (2) gamma 스윕: best_lr 고정, eps는 현재 슬라이더 값 사용 ==
    for gamma_val in gamma_candidates:
        static_returns = []
        for run_idx in range(runs_per_config):
            np.random.seed(base_seed + 100 + run_idx)
            agent_u = RecommendationAgent(env, use_constraints=False, lr=best_lr, gamma=gamma_val, eps=eps_s)
            agent_s = RecommendationAgent(env, use_constraints=True, lr=best_lr, gamma=gamma_val, eps=eps_v)

            h_s = 0.0

            for i in range(20, 20 + episodes_sweep):
                _, _, r_u = agent_u.select_action(current_step=i)
                _, _, r_s = agent_s.select_action(current_step=i)
                h_s += r_s

            static_returns.append(h_s)

        results.append({
            "Phase": "gamma",
            "lr": best_lr,
            "gamma": gamma_val,
            "eps_s": eps_s,
            "eps_v": eps_v,
            "STATIC Mean (%)": np.mean(static_returns),
            "STATIC Median (%)": float(np.median(static_returns)),
            "STATIC Std (%)": float(np.std(static_returns)) if len(static_returns) > 1 else 0.0,
        })

    df_gamma = pd.DataFrame([r for r in results if r["Phase"] == "gamma"])
    best_gamma = df_gamma.sort_values("STATIC Mean (%)", ascending=False).iloc[0]["gamma"]

    # == (3) eps 스윕: best_lr, best_gamma 고정 ==
    for eps_val in eps_candidates:
        static_returns = []
        for run_idx in range(runs_per_config):
            np.random.seed(base_seed + 200 + run_idx)
            agent_u = RecommendationAgent(env, use_constraints=False, lr=best_lr, gamma=best_gamma, eps=eps_val)
            agent_s = RecommendationAgent(env, use_constraints=True, lr=best_lr, gamma=best_gamma, eps=eps_val)

            h_s = 0.0

            for i in range(20, 20 + episodes_sweep):
                _, _, r_u = agent_u.select_action(current_step=i)
                _, _, r_s = agent_s.select_action(current_step=i)
                h_s += r_s

            static_returns.append(h_s)

        results.append({
            "Phase": "eps",
            "lr": best_lr,
            "gamma": best_gamma,
            "eps_s": eps_val,
            "eps_v": eps_val,
            "eps_sym": eps_val,
            "STATIC Mean (%)": np.mean(static_returns),
            "STATIC Median (%)": float(np.median(static_returns)),
            "STATIC Std (%)": float(np.std(static_returns)) if len(static_returns) > 1 else 0.0,
        })

    st.session_state.sweep_results = results

if st.session_state.sweep_results:
    df_res = pd.DataFrame(st.session_state.sweep_results)

    st.markdown("#### Sweep Summary (STATIC 기준)")
    st.dataframe(
        df_res[
            ["Phase", "lr", "gamma", "eps_s", "eps_v", "STATIC Mean (%)", "STATIC Median (%)", "STATIC Std (%)"]
        ]
        .sort_values(["Phase", "STATIC Mean (%)"], ascending=[True, False])
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # 최종 추천 파라미터 (eps 단계에서 최고 조합)
    df_eps = df_res[df_res["Phase"] == "eps"].sort_values("STATIC Mean (%)", ascending=False)
    if not df_eps.empty:
        best_row = df_eps.iloc[0]
        st.success(
            f"추천 파라미터 조합 → lr={best_row['lr']}, γ={best_row['gamma']}, ε_s=ε_v={best_row['eps_sym']} "
            f"(STATIC Mean: {best_row['STATIC Mean (%)']:.2f}%, Median: {best_row['STATIC Median (%)']:.2f}%)"
        )

# == 📊 하단: 통계 분석 고도화 (누적 그래프 및 박스 플롯) ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### Trial History: Statistical Analysis (Alpha Performance)")
    df_h = pd.DataFrame(st.session_state.trial_history)
    for _col in ["Q-Learn Final (%)", "Q-STATIC Final (%)", "PG Final (%)"]:
        if _col not in df_h.columns:
            df_h[_col] = np.nan

    # 통계량 계산
    v_mean, v_max, v_min = df_h['Vanilla Final (%)'].mean(), df_h['Vanilla Final (%)'].max(), df_h['Vanilla Final (%)'].min()
    s_mean, s_max, s_min = df_h['STATIC Final (%)'].mean(), df_h['STATIC Final (%)'].max(), df_h['STATIC Final (%)'].min()
    v_std = df_h['Vanilla Final (%)'].std() if len(df_h) > 1 else 0.0
    s_std = df_h['STATIC Final (%)'].std() if len(df_h) > 1 else 0.0
    avg_kospi = df_h['KOSPI Final (%)'].mean()
    q_mean = df_h["Q-Learn Final (%)"].mean(skipna=True)
    qs_mean = df_h["Q-STATIC Final (%)"].mean(skipna=True)
    pg_mean = df_h["PG Final (%)"].mean(skipna=True)

    def _alpha_str(m):
        if pd.isna(m):
            return "—"
        return f"{m - avg_kospi:.2f}%p"

    st.success(
        f"시장 대비 Alpha(평균): STATIC **{_alpha_str(s_mean)}** | Vanilla **{_alpha_str(v_mean)}** | "
        f"Q-Learn **{_alpha_str(q_mean)}** | Q+STATIC **{_alpha_str(qs_mean)}** | PG **{_alpha_str(pg_mean)}**"
    )

    cand = [
        ("Vanilla", v_mean),
        ("STATIC", s_mean),
        ("Q-Learning", q_mean),
        ("Q+STATIC", qs_mean),
        ("Policy Grad", pg_mean),
    ]
    cand_f = [(n, float(v)) for n, v in cand if pd.notna(v)]
    if cand_f:
        best_overall, best_v = max(cand_f, key=lambda kv: kv[1])
        st.markdown(f"**누적 Trial 기준 평균 수익 최고 모델:** `{best_overall}` ({best_v:.2f}%)")

    q_mean_s = f"{q_mean:.2f}" if pd.notna(q_mean) else "N/A"
    qs_mean_s = f"{qs_mean:.2f}" if pd.notna(qs_mean) else "N/A"
    pg_mean_s = f"{pg_mean:.2f}" if pd.notna(pg_mean) else "N/A"

    # == 회차별 누적 성과 추이 그래프 ==
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Vanilla Final (%)'], mode='lines+markers', name='<b>Vanilla Return</b>', line=dict(color='#e05050', width=2), marker=dict(size=8)))
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['STATIC Final (%)'], mode='lines+markers', name='<b>STATIC Return (Ours)</b>', line=dict(color='#4a90d9', width=2), marker=dict(size=8)))
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Q-Learn Final (%)'], mode='lines+markers', name='<b>Q-Learning</b>', line=dict(color='#ff9800', width=2), marker=dict(size=8)))
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Q-STATIC Final (%)'], mode='lines+markers', name='<b>Q+STATIC</b>', line=dict(color='#9c27b0', width=2), marker=dict(size=8)))
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['PG Final (%)'], mode='lines+markers', name='<b>Policy Gradient</b>', line=dict(color='#795548', width=2), marker=dict(size=8)))

    # Vanilla 선들
    fig_trend.add_hline(y=v_mean, line_dash="solid", line_color="#e05050", opacity=0.4, annotation_text=f"Vanilla Mean", annotation_position="top right")
    fig_trend.add_hline(y=v_max, line_dash="dot", line_color="#e05050", opacity=0.3, annotation_text=f"Vanilla Max", annotation_position="top right")
    fig_trend.add_hline(y=v_min, line_dash="dot", line_color="#e05050", opacity=0.3, annotation_text=f"Vanilla Min", annotation_position="bottom right")
    # STATIC 선들
    fig_trend.add_hline(y=s_mean, line_dash="solid", line_color="#4a90d9", opacity=0.4, annotation_text=f"STATIC Mean", annotation_position="top left")
    fig_trend.add_hline(y=s_max, line_dash="dot", line_color="#4a90d9", opacity=0.3, annotation_text=f"STATIC Max", annotation_position="top left")
    fig_trend.add_hline(y=s_min, line_dash="dot", line_color="#4a90d9", opacity=0.3, annotation_text=f"STATIC Min", annotation_position="bottom left")

    fig_trend.update_layout(
        title=dict(text="<b>Trial-by-Trial Return Progression & Stability</b>", font=dict(size=24, family="Arial Black")),
        xaxis=dict(title="<b>Trial Number</b>", tickmode='linear', dtick=1),
        yaxis=dict(title="<b>Final Return (%)</b>"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=60, b=40, l=40, r=40)
    )
    fig_trend.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_trials")

    # == 하단 2단 레이아웃 (박스 플롯 & 통계 테이블) ==
    col_box, col_tbl_h = st.columns([2, 1])
    with col_box:
        fig_box = go.Figure()
        
        # 박스 형태 및 밀착 배치
        fig_box.add_trace(go.Box(y=df_h['Vanilla Final (%)'], x0=0.6, name='<b>Vanilla RL</b>', line=dict(color='#e05050', width=3), fillcolor='rgba(224,80,80,0.05)', boxmean=True, width=0.35))
        fig_box.add_trace(go.Box(y=df_h['STATIC Final (%)'], x0=1.35, name='<b>STATIC RL</b>', line=dict(color='#4a90d9', width=3), fillcolor='rgba(74,144,217,0.05)', boxmean=True, width=0.35))
        fig_box.add_trace(go.Box(y=df_h['Q-Learn Final (%)'].dropna(), x0=2.1, name='<b>Q-Learning</b>', line=dict(color='#ff9800', width=3), fillcolor='rgba(255,152,0,0.05)', boxmean=True, width=0.35))
        fig_box.add_trace(go.Box(y=df_h['Q-STATIC Final (%)'].dropna(), x0=2.85, name='<b>Q+STATIC</b>', line=dict(color='#9c27b0', width=3), fillcolor='rgba(156,39,176,0.05)', boxmean=True, width=0.35))
        fig_box.add_trace(go.Box(y=df_h['PG Final (%)'].dropna(), x0=3.6, name='<b>Policy Grad</b>', line=dict(color='#795548', width=3), fillcolor='rgba(121,85,72,0.05)', boxmean=True, width=0.35))

        med_v, med_s = df_h['Vanilla Final (%)'].median(), df_h['STATIC Final (%)'].median()

        # 수치 라벨 밀착 배치
        fig_box.add_annotation(x=0.75, y=v_mean, text=f"<b>Mean: {v_mean:.2f}%</b>", showarrow=False, xshift=-4, yshift=8, xanchor='right', font=dict(color='#e05050', size=13, family="Arial Black"))
        fig_box.add_annotation(x=0.75, y=med_v, text=f"<b>Median: {med_v:.2f}%</b>", showarrow=False, xshift=-4, yshift=-8, xanchor='right', font=dict(color='#e05050', size=13, family="Arial Black"))
        fig_box.add_annotation(x=2.5, y=med_s, text=f"<b>Median: {med_s:.2f}%</b>", showarrow=False, xshift=4, yshift=8, xanchor='left', font=dict(color='#4a90d9', size=13, family="Arial Black"))
        fig_box.add_annotation(x=2.5, y=s_mean, text=f"<b>Mean: {s_mean:.2f}%</b>", showarrow=False, xshift=4, yshift=-8, xanchor='left', font=dict(color='#4a90d9', size=13, family="Arial Black"))
        
        fig_box.add_hline(y=avg_kospi, line_width=2.5, line_dash="dot", line_color="green")
        fig_box.add_annotation(x=2.1, xref="x", y=avg_kospi, text=f"<b>KOSPI Index<br>{avg_kospi:.2f}%</b>", showarrow=False, yshift=18, xanchor='center', align='center', font=dict(color="green", size=13, family="Arial Black"), bgcolor="rgba(0,0,0,0)")

        fig_box.update_layout(
            title=dict(text="<b>Return Distribution across Trials</b>", font=dict(size=26, family="Arial Black")),
            yaxis=dict(
                title=dict(text="<b>Final Return (%)</b>", font=dict(size=22, family="Arial Black")),
                tickfont=dict(size=18, family="Arial Black"),
            ),
            xaxis=dict(
                title=dict(text="<b>Performance Metrics</b>", font=dict(size=22, family="Arial Black")),
                tickmode='array',
                tickvals=[0.6, 1.35, 2.1, 2.85, 3.6],
                ticktext=['<b>Vanilla</b>', '<b>STATIC</b>', '<b>Q-Learn</b>', '<b>Q+ST</b>', '<b>PG</b>'],
                tickfont=dict(size=14, family="Arial Black"),
                range=[0, 4.2],
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=550,
            margin=dict(t=120, b=100, l=80, r=80),
        )
        fig_box.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
        st.plotly_chart(fig_box, use_container_width=True, key="box_returns")
    
    with col_tbl_h:
        # 테이블 상단 요약 통계량 명시
        st.markdown(f"""
        <div style='background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.3); margin-bottom: 10px;'>
            <h4 style='margin-top:0px; color: var(--text-color); font-weight: 900;'> 통계 요약 (Expected & Risk)</h4>
            <ul style='font-size: 15px; margin-bottom: 0px; color: var(--text-color);'>
                <li><b style='color:#e05050;'>Vanilla 평균(기대치):</b> {v_mean:.2f}% (σ={v_std:.2f}%)</li>
                <li><b style='color:#e05050;'>Vanilla 범위:</b> {v_min:.2f}% ~ {v_max:.2f}%</li>
                <hr style='margin: 8px 0; border-color: rgba(128,128,128,0.3);'>
                <li><b style='color:#4a90d9;'>STATIC 평균(기대치):</b> {s_mean:.2f}% (σ={s_std:.2f}%)</li>
                <li><b style='color:#4a90d9;'>STATIC 범위:</b> {s_min:.2f}% ~ {s_max:.2f}%</li>
                <hr style='margin: 8px 0; border-color: rgba(128,128,128,0.3);'>
                <li><b style='color:#ff9800;'>Q-Learning 평균:</b> {q_mean_s}%</li>
                <li><b style='color:#9c27b0;'>Q+STATIC 평균:</b> {qs_mean_s}%</li>
                <li><b style='color:#795548;'>Policy Grad 평균:</b> {pg_mean_s}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            df_h.set_index("Trial")
                .style.map(style_df)
                .format({
                    "Vanilla Final (%)": "{:.2f}",
                    "STATIC Final (%)": "{:.2f}",
                    "Q-Learn Final (%)": "{:.2f}",
                    "Q-STATIC Final (%)": "{:.2f}",
                    "PG Final (%)": "{:.2f}",
                    "KOSPI Final (%)": "{:.2f}",
                    "Seed": "{:.0f}",
                }),
            height=320,
            use_container_width=True,
        )
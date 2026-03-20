import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import config

class KOSPIEnvironment:
    """ KOSPI 주요 종목 및 벤치마크(KOSPI 지수 또는 ETF) 데이터를 관리하는 환경 """
    def __init__(self):
        # KOSPI 대형주 예시 티커 목록 (yfinance 형식)
        self.tickers = [
            "005930.KS",  # 삼성전자
            "000660.KS",  # SK하이닉스
            "035420.KS",  # NAVER
            "051910.KS",  # LG화학
            "207940.KS",  # 삼성바이오로직스
            "005380.KS",  # 현대차
            "000270.KS",  # 기아
            "068270.KS",  # 셀트리온
            "035720.KS",  # 카카오
            "105560.KS",  # KB금융
            "055550.KS",  # 신한지주
            "028260.KS",  # 삼성물산
            "012330.KS",  # 현대모비스
            "096770.KS",  # SK이노베이션
            "034730.KS",  # SK
            "251270.KS",  # 넷마블
            "066570.KS",  # LG전자
            "003550.KS",  # LG
            "316140.KS",  # 우리금융지주
            "003490.KS",  # 대한항공
        ]

        # 코스피 지수 자체를 벤치마크로 사용 (KOSPI 지수: ^KS11)
        # 필요에 따라 KODEX200 ETF(069500.KS) 등으로 교체 가능
        self.benchmark = "^KS11"
        self.all_symbols = self.tickers + [self.benchmark]
        
        self.data, self.tickers = self._download_data()
        self.vocab_size = len(self.tickers)

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        # 벤치마크를 포함하여 KOSPI 관련 데이터 다운로드 (5년, 일봉)
        data = yf.download(_self.all_symbols, period="5y", interval="1d")['Close']
        data = data.ffill().bfill().dropna(axis=1)
        # _self를 직접 변형하지 않고 tickers를 반환값으로 전달 (mutation 경고 방지)
        tickers = [t for t in data.columns if t != _self.benchmark]
        return data, tickers

class StaticConstraintEngine:
    def __init__(self, env, current_step):
        self.env = env
        self.vocab_size = env.vocab_size
        self.valid_mask = np.ones(self.vocab_size, dtype=bool)
        
        if current_step >= 20:
            history = self.env.data[self.env.tickers].iloc[current_step-20 : current_step]
            sma_20 = history.mean()
            current_prices = self.env.data[self.env.tickers].iloc[current_step]
            
            for i, ticker in enumerate(self.env.tickers):
                if current_prices[ticker] < sma_20[ticker]:
                    self.valid_mask[i] = False
                    
            if not np.any(self.valid_mask):
                self.valid_mask = np.ones(self.vocab_size, dtype=bool)

    def apply_mask(self, logits):
        return np.where(self.valid_mask, logits, -np.inf)

class RecommendationAgent:
    # lr, gamma, eps 인자를 추가하고 기본값을 설정합니다.
    def __init__(self, env, use_constraints=True, lr=0.01, gamma=0.98, eps=0.1):
        self.env = env
        self.use_constraints = use_constraints
        
        # 주입받은 하이퍼파라미터를 에이전트 속성으로 저장
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        
    def select_action(self, current_step):
        engine = StaticConstraintEngine(self.env, current_step)
        mask = engine.valid_mask if self.use_constraints else np.ones(self.env.vocab_size, dtype=bool)

        # ε-greedy: ε 비율로 유효 행동 무작위 탐색, 나머지는 기존 랜덤 로짓 argmax
        if np.random.random() < self.epsilon:
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                chosen_action = int(np.random.randint(0, self.env.vocab_size))
            else:
                chosen_action = int(np.random.choice(idx))
        else:
            logits = np.random.randn(self.env.vocab_size)
            if self.use_constraints:
                logits = engine.apply_mask(logits)
            chosen_action = int(np.argmax(logits))
        
        if current_step + 1 < len(self.env.data):
            current_price = float(self.env.data[self.env.tickers[chosen_action]].iloc[current_step])
            next_price = float(self.env.data[self.env.tickers[chosen_action]].iloc[current_step + 1])
            reward = ((next_price - current_price) / current_price) * 100 if current_price > 0 else 0.0
        else:
            reward = 0.0
            
        is_valid = engine.valid_mask[chosen_action]
        chosen_ticker = self.env.tickers[chosen_action]
        
        return chosen_ticker, is_valid, reward


def _masked_argmax(q_values: np.ndarray, valid_mask: np.ndarray) -> int:
    """유효한 행동 중에서만 argmax. 모두 무효면 전체 argmax."""
    masked = np.where(valid_mask, q_values, -np.inf)
    if not np.any(np.isfinite(masked)):
        return int(np.argmax(q_values))
    return int(np.argmax(masked))


def _random_valid_action(valid_mask: np.ndarray, rng: np.random.Generator) -> int:
    """마스크가 True인 인덱스 중 균등 랜덤."""
    idx = np.flatnonzero(valid_mask)
    if len(idx) == 0:
        return int(rng.integers(0, len(valid_mask)))
    return int(rng.choice(idx))


class QLearningBanditAgent:
    """
    단일 상태(밴딧) Q-Learning. 종목 선택을 Q(a)로 학습.
    use_constraints=True 이면 StaticConstraintEngine 마스크를 탐색/그리디에 적용.
    """

    def __init__(self, env, use_constraints: bool = False, lr: float = 0.1, gamma: float = 0.98, epsilon: float = 0.2):
        self.env = env
        self.use_constraints = use_constraints
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros(env.vocab_size, dtype=np.float64)

    def reset(self):
        self.q = np.zeros(self.env.vocab_size, dtype=np.float64)

    def _step_reward(self, action: int, current_step: int) -> float:
        if current_step + 1 >= len(self.env.data):
            return 0.0
        t = self.env.tickers[action]
        p0 = float(self.env.data[t].iloc[current_step])
        p1 = float(self.env.data[t].iloc[current_step + 1])
        if p0 <= 0:
            return 0.0
        return ((p1 - p0) / p0) * 100.0

    def train_step(self, current_step: int, rng: np.random.Generator) -> None:
        engine = StaticConstraintEngine(self.env, current_step)
        mask = engine.valid_mask if self.use_constraints else np.ones(self.env.vocab_size, dtype=bool)

        if rng.random() < self.epsilon:
            a = _random_valid_action(mask, rng)
        else:
            a = _masked_argmax(self.q, mask)

        r = self._step_reward(a, current_step)
        max_q = float(np.max(self.q))
        self.q[a] += self.lr * (r + self.gamma * max_q - self.q[a])

    def select_action(self, current_step: int, rng: np.random.Generator, greedy: bool = False):
        """greedy=True: 실행(평가) 구간에서 ε=0에 가깝게 고정 그리디."""
        engine = StaticConstraintEngine(self.env, current_step)
        mask = engine.valid_mask if self.use_constraints else np.ones(self.env.vocab_size, dtype=bool)
        eps = 0.0 if greedy else self.epsilon

        if eps > 0 and rng.random() < eps:
            a = _random_valid_action(mask, rng)
        else:
            a = _masked_argmax(self.q, mask)

        r = self._step_reward(a, current_step)
        is_valid = engine.valid_mask[a]
        ticker = self.env.tickers[a]
        return ticker, is_valid, r


class PolicyGradientBanditAgent:
    """
    REINFORCE 스타일 밴딧: softmax 정책 π(a)를 일일 보상으로 업데이트.
    use_constraints=True이면 마스크 밖 로짓을 -inf로 두고 정규화.
    """

    def __init__(self, env, use_constraints: bool = False, lr: float = 0.05):
        self.env = env
        self.use_constraints = use_constraints
        self.lr = lr
        self.theta = np.zeros(env.vocab_size, dtype=np.float64)

    def reset(self):
        self.theta = np.zeros(self.env.vocab_size, dtype=np.float64)

    def _step_reward(self, action: int, current_step: int) -> float:
        if current_step + 1 >= len(self.env.data):
            return 0.0
        t = self.env.tickers[action]
        p0 = float(self.env.data[t].iloc[current_step])
        p1 = float(self.env.data[t].iloc[current_step + 1])
        if p0 <= 0:
            return 0.0
        return ((p1 - p0) / p0) * 100.0

    def _probs(self, current_step: int, rng: np.random.Generator) -> np.ndarray:
        engine = StaticConstraintEngine(self.env, current_step)
        mask = engine.valid_mask if self.use_constraints else np.ones(self.env.vocab_size, dtype=bool)
        logits = np.where(mask, self.theta, -1e9)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        s = exp.sum()
        if s <= 0 or not np.isfinite(s):
            p = mask.astype(np.float64)
            return p / (p.sum() + 1e-12)
        return exp / s

    def train_step(self, current_step: int, rng: np.random.Generator) -> None:
        p = self._probs(current_step, rng)
        a = int(rng.choice(self.env.vocab_size, p=p))
        r = self._step_reward(a, current_step)
        # REINFORCE: θ += lr * R * (e_a - π)
        grad = -p
        grad[a] += 1.0
        self.theta += self.lr * r * grad

    def select_action(self, current_step: int, rng: np.random.Generator):
        p = self._probs(current_step, rng)
        a = int(rng.choice(self.env.vocab_size, p=p))
        r = self._step_reward(a, current_step)
        engine = StaticConstraintEngine(self.env, current_step)
        is_valid = engine.valid_mask[a]
        ticker = self.env.tickers[a]
        return ticker, is_valid, r
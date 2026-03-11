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
        logits = np.random.randn(self.env.vocab_size)
        engine = StaticConstraintEngine(self.env, current_step)
        
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
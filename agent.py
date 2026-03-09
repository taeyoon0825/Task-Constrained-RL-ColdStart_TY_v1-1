import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import config

class SP500Environment:
    """ S&P 500 대표 종목 데이터를 관리하는 환경 """
    def __init__(self):
        # !! [수정됨] API 오류가 잦은 BRK-B를 XOM(엑슨모빌)으로 교체하여 안정성 확보
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "XOM", "LLY", "V",
                        "JPM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "CVX", "MRK"]
        self.data = self._download_data()
        self.vocab_size = len(self.tickers)

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        # 최근 6개월 데이터 다운로드
        data = yf.download(_self.tickers, period="6mo", interval="1d")['Close']
        
        # == [결측치 완벽 방어 로직] ==
        # 1. 앞뒤 날짜의 가격으로 빈칸을 채움 (ffill, bfill)
        # 2. 그래도 남는 결측치가 있는 티커(열)는 아예 삭제 (dropna)
        data = data.ffill().bfill().dropna(axis=1)
        
        # 살아남은 검증된 티커 목록으로 업데이트
        _self.tickers = list(data.columns)
        return data

class StaticConstraintEngine:
    def __init__(self, env, current_step):
        self.env = env
        self.vocab_size = env.vocab_size
        self.valid_mask = np.ones(self.vocab_size, dtype=bool)
        
        # == VNTK 마스킹 로직: 20일 이동평균선(SMA) 이탈 종목 차단 ==
        if current_step >= 20:
            history = self.env.data.iloc[current_step-20 : current_step]
            sma_20 = history.mean()
            current_prices = self.env.data.iloc[current_step]
            
            for i, ticker in enumerate(self.env.tickers):
                if current_prices[ticker] < sma_20[ticker]:
                    self.valid_mask[i] = False
                    
            # !! [예외 처리] 시장 전체가 폭락하여 모든 종목이 20일선 아래일 경우
            # 모든 마스크가 False가 되어 에이전트가 고장나는 것을 방지 (마스크 전면 해제)
            if not np.any(self.valid_mask):
                self.valid_mask = np.ones(self.vocab_size, dtype=bool)

    def apply_mask(self, logits):
        return np.where(self.valid_mask, logits, -np.inf)

class RecommendationAgent:
    def __init__(self, env, use_constraints=False):
        self.env = env
        self.use_constraints = use_constraints
        
    def select_action(self, current_step):
        logits = np.random.randn(self.env.vocab_size)
        engine = StaticConstraintEngine(self.env, current_step)
        
        if self.use_constraints:
            logits = engine.apply_mask(logits)
            
        chosen_action = int(np.argmax(logits))
        
        # == [수익률 계산 및 0분모 방어] ==
        if current_step + 1 < len(self.env.data):
            current_price = float(self.env.data.iloc[current_step, chosen_action])
            next_price = float(self.env.data.iloc[current_step + 1, chosen_action])
            
            if current_price > 0:
                reward = ((next_price - current_price) / current_price) * 100 
            else:
                reward = 0.0
        else:
            reward = 0.0
            
        is_valid = engine.valid_mask[chosen_action]
        chosen_ticker = self.env.tickers[chosen_action]
        
        return chosen_ticker, is_valid, reward
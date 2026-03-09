import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import config

class SP500Environment:
    """ S&P 500 대표 종목 및 벤치마크(SPY) 데이터를 관리하는 환경 """
    def __init__(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "XOM", "LLY", "V",
                        "JPM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "CVX", "MRK"]
        self.benchmark = "SPY"
        self.all_symbols = self.tickers + [self.benchmark]
        
        self.data = self._download_data()
        self.vocab_size = len(self.tickers)

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        # 벤치마크를 포함하여 데이터 다운로드
        data = yf.download(_self.all_symbols, period="6mo", interval="1d")['Close']
        data = data.ffill().bfill().dropna(axis=1)
        
        # 에이전트가 선택할 수 있는 종목풀(tickers)에서 SPY는 제외
        _self.tickers = [t for t in data.columns if t != _self.benchmark]
        return data

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
    def __init__(self, env, use_constraints=False):
        self.env = env
        self.use_constraints = use_constraints
        
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
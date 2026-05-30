"""
╔══════════════════════════════════════════════════════════════════╗
║     MNQ/NQ PREDICTOR — ML DASHBOARD v6.1                        ║
║     Deep OLED Dark · TradingView-style · Multi-Page · Pro Suite ║
║                                                                  ║
║     NEW IN v6.0:                                                 ║
║     · AI Natural-Language Explainability (per-driver reasoning)  ║
║     · Bias Invalidation & Flip Conditions                        ║
║     · CVD Proxy (Cumulative Volume Delta)                        ║
║     · News Sentiment (keyword NLP)                               ║
║     · HMM-style Regime Detector (statistical Markov proxy)       ║
║     · Interactive Chart Toggles (VP, Targets, Projection)        ║
║     · Global UI Tooltips on all key metrics                      ║
║     · Hardened RSS News Fetcher with robust fallbacks            ║
║                                                                  ║
║     FIX IN v6.1:                                                 ║
║     · Gemini Vision: 1.5-flash (retired, 404) -> 2.5-flash       ║
║     · Model fallback chain + x-goog-api-key header               ║
║     · Robust 404 / empty-response / safety-block handling        ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    streamlit run live-nq.py

Dependencies:
    pip install streamlit streamlit-lightweight-charts-ntf yfinance xgboost
                lightgbm scikit-learn scipy requests beautifulsoup4 feedparser
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import json
import logging
import hashlib
import pickle
import traceback
import time
import re
import feedparser
import concurrent.futures
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import streamlit as st

import yfinance as yf
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

import requests
from bs4 import BeautifulSoup

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("NQ_PREDICTOR")

# ── Storage ──────────────────────────────────────────────────────
DATA_DIR = Path("nq_data")
DATA_DIR.mkdir(exist_ok=True)
MODEL_CACHE = DATA_DIR / "model_v6.pkl"

SYMBOLS = {
    "NQ":    "NQ=F",
    "VIX":   "^VIX",
    "DXY":   "DX-Y.NYB",
    "US10Y": "^TNX",
    "QQQ":   "QQQ",
    "ES":    "ES=F",
}

TF_CONFIG = {
    "1M":  {"interval": "1m",  "period": "5d",   "proj_bars": 30,  "label": "1 Minute"},
    "5M":  {"interval": "5m",  "period": "7d",   "proj_bars": 24,  "label": "5 Minutes"},
    "15M": {"interval": "15m", "period": "30d",  "proj_bars": 16,  "label": "15 Minutes"},
    "30M": {"interval": "30m", "period": "60d",  "proj_bars": 12,  "label": "30 Minutes"},
    "1H":  {"interval": "1h",  "period": "60d",  "proj_bars": 8,   "label": "1 Hour"},
    "4H":  {"interval": "1h",  "period": "60d",  "proj_bars": 6,   "label": "4 Hours",   "resample": "4h"},
    "D":   {"interval": "1d",  "period": "2y",   "proj_bars": 5,   "label": "Daily"},
}

PAGES = ["Dashboard", "Charts & Orderflow", "Validation & Internals", "News & Outlook", "Chart Analysis (AI)"]

# ── Sentiment keywords ────────────────────────────────────────────
BULLISH_WORDS = [
    "surge", "rally", "gain", "rise", "bullish", "upside", "beat", "strong",
    "record", "high", "positive", "growth", "recovery", "soar", "jump",
    "outperform", "upgrade", "buy", "boost", "upbeat", "optimism", "breakout",
]
BEARISH_WORDS = [
    "drop", "fall", "decline", "bearish", "downside", "miss", "weak",
    "low", "negative", "recession", "sell", "downgrade", "slump", "crash",
    "fear", "risk", "concern", "selloff", "plunge", "tumble", "loss",
    "inflation", "rate hike", "hike", "hawkish", "volatility spike",
]

# ═══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════
@dataclass
class Prediction:
    bias: str = "NEUTRAL"
    prob_bull: float = 0.50
    prob_bear: float = 0.50
    confidence: float = 0.50
    current_price: float = 0.0
    atr: float = 0.0
    upper_t1_lo: float = 0.0
    upper_t1_hi: float = 0.0
    upper_t2_lo: float = 0.0
    upper_t2_hi: float = 0.0
    lower_t1_lo: float = 0.0
    lower_t1_hi: float = 0.0
    lower_t2_lo: float = 0.0
    lower_t2_hi: float = 0.0
    upper_t1_conf: float = 0.0
    upper_t2_conf: float = 0.0
    lower_t1_conf: float = 0.0
    lower_t2_conf: float = 0.0
    upper_t1_label: str = "T1 (ATR)"
    upper_t2_label: str = "T2 (ATR)"
    lower_t1_label: str = "T1 (ATR)"
    lower_t2_label: str = "T2 (ATR)"
    regime: str = "Normal Market"
    regime_phase: str = "Trend"
    macro_regime: str = "Normal Market"
    macro_phase: str = "Trend"
    micro_regime: str = "Normal Market"
    micro_phase: str = "Trend"
    micro_bias: str = "NEUTRAL"
    vix: float = 20.0
    vol_regime: str = "Normal"
    trend_prob: float = 0.50
    vix_divergence: str = "ALIGNED"
    dxy_divergence: str = "ALIGNED"
    internals_health: str = "NEUTRAL"
    add_proxy: float = 0.0
    session_outlook: Dict[str, str] = field(default_factory=dict)
    high_time_est: str = "12:00"
    low_time_est: str = "15:00"
    bias_horizon: str = "Intraday (4-6h)"
    wf_accuracy: float = 0.0
    wf_auc: float = 0.0
    delta_pct: float = 0.0
    rsi: float = 50.0
    momentum: str = "NEUTRAL"
    top_factors: List[Dict] = field(default_factory=list)
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    mins_to_ny_open: int = 0
    # v6 additions
    invalidation_price: float = 0.0
    invalidation_condition: str = ""
    invalidation_flip_detail: str = ""
    cvd_trend: str = "NEUTRAL"
    cvd_value: float = 0.0
    hmm_regime: str = "Trend"
    hmm_confidence: float = 0.5
    ai_explanation: str = ""
    news_sentiment: str = "NEUTRAL"
    news_sentiment_score: float = 0.0
    expected_volume_slots: List[Dict] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ════════════════════════════════════════════════════════════════
class DataPipeline:
    _cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
    TTL = 300  # 300s cache — reduces repeated yfinance calls on cloud

    _FETCH_TIMEOUT = 20  # seconds per yfinance request on cloud

    @classmethod
    def _fetch_raw(cls, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Inner fetch — runs inside a thread so we can enforce a wall-clock timeout."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.columns = [c.lower() for c in df.columns]
        df = df[df.index <= datetime.now()]
        return df if not df.empty else None

    @classmethod
    def fetch(cls, symbol: str, period: str = "2y",
              interval: str = "1d") -> Optional[pd.DataFrame]:
        key = f"{symbol}_{period}_{interval}"
        if key in cls._cache:
            df, ts = cls._cache[key]
            if (datetime.now() - ts).total_seconds() < cls.TTL:
                return df.copy()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(cls._fetch_raw, symbol, period, interval)
                try:
                    df = fut.result(timeout=cls._FETCH_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Fetch timeout ({cls._FETCH_TIMEOUT}s): {symbol} p={period} i={interval}")
                    return None
            if df is None:
                logger.warning(f"Empty data for {symbol} p={period} i={interval}")
                return None
            cls._cache[key] = (df.copy(), datetime.now())
            return df
        except Exception as e:
            logger.warning(f"Fetch error {symbol}: {e}")
            return None

    @classmethod
    def fetch_correlated(cls) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch all correlated symbols in parallel to cut cloud init time."""
        targets = {k: v for k, v in SYMBOLS.items() if k != "NQ"}
        results: Dict[str, Optional[pd.DataFrame]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as ex:
            futs = {ex.submit(cls.fetch, sym, "2y", "1d"): name
                    for name, sym in targets.items()}
            for fut in concurrent.futures.as_completed(futs, timeout=30):
                name = futs[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:
                    logger.warning(f"Correlated fetch failed for {name}: {e}")
                    results[name] = None
        return results

    @classmethod
    def fetch_for_timeframe(cls, tf: str) -> Optional[pd.DataFrame]:
        cfg = TF_CONFIG[tf]
        df = cls.fetch(SYMBOLS["NQ"], cfg["period"], cfg["interval"])
        if df is None or df.empty:
            return None
        if cfg.get("resample"):
            df = df.resample(cfg["resample"]).agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum",
            }).dropna(subset=["open", "close"])
        return df

    @classmethod
    def fetch_intraday(cls, interval: str = "15m", period: str = "5d") -> Optional[pd.DataFrame]:
        return cls.fetch(SYMBOLS["NQ"], period, interval)

    @classmethod
    def fetch_premarket(cls) -> Optional[pd.DataFrame]:
        try:
            df = cls.fetch(SYMBOLS["NQ"], "5d", "5m")
            if df is None or df.empty:
                return None
            cutoff = datetime.now() - timedelta(hours=4)
            result = df[df.index >= cutoff]
            return result if len(result) >= 2 else df.tail(10)
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════
#  VOLUME PROFILE
# ════════════════════════════════════════════════════════════════
class VolumeProfile:
    @staticmethod
    def calculate(df: pd.DataFrame, n_levels: int = 50) -> Dict[str, float]:
        try:
            if df is None or len(df) < 5:
                return {"poc": 0.0, "vah": 0.0, "val": 0.0}
            price_min = df["low"].min()
            price_max = df["high"].max()
            if price_max <= price_min:
                cp = float(df["close"].iloc[-1])
                return {"poc": cp, "vah": price_max, "val": price_min}
            levels = np.linspace(price_min, price_max, n_levels + 1)
            vol_at_price = np.zeros(n_levels)
            for _, row in df.iterrows():
                lo, hi, vol = row["low"], row["high"], row.get("volume", 0)
                if vol <= 0 or hi <= lo:
                    continue
                for i in range(n_levels):
                    lev_lo, lev_hi = levels[i], levels[i + 1]
                    overlap = max(0, min(hi, lev_hi) - max(lo, lev_lo))
                    if overlap > 0:
                        vol_at_price[i] += vol * overlap / (hi - lo + 1e-9)
            total_vol = vol_at_price.sum()
            if total_vol == 0:
                cp = float(df["close"].iloc[-1])
                return {"poc": cp, "vah": cp + 50, "val": cp - 50}
            poc_idx = int(np.argmax(vol_at_price))
            poc = float((levels[poc_idx] + levels[poc_idx + 1]) / 2)
            target_vol = total_vol * 0.70
            val_idx, vah_idx = poc_idx, poc_idx
            cum_vol = vol_at_price[poc_idx]
            while cum_vol < target_vol:
                expand_up = vah_idx + 1 < n_levels
                expand_down = val_idx - 1 >= 0
                if expand_up and expand_down:
                    if vol_at_price[vah_idx + 1] >= vol_at_price[val_idx - 1]:
                        vah_idx += 1; cum_vol += vol_at_price[vah_idx]
                    else:
                        val_idx -= 1; cum_vol += vol_at_price[val_idx]
                elif expand_up:
                    vah_idx += 1; cum_vol += vol_at_price[vah_idx]
                elif expand_down:
                    val_idx -= 1; cum_vol += vol_at_price[val_idx]
                else:
                    break
            vah = float((levels[vah_idx] + levels[vah_idx + 1]) / 2)
            val = float((levels[val_idx] + levels[val_idx + 1]) / 2)
            return {"poc": round(poc, 2), "vah": round(vah, 2), "val": round(val, 2)}
        except Exception as e:
            logger.warning(f"VP error: {e}")
            cp = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0
            return {"poc": cp, "vah": cp + 50, "val": cp - 50}


# ════════════════════════════════════════════════════════════════
#  CVD PROXY (v6 new)
# ════════════════════════════════════════════════════════════════
class CVDCalculator:
    """
    Cumulative Volume Delta proxy from OHLCV data.
    Each bar's delta is estimated as:
      - Buy volume  = volume * (close - low)  / (high - low)
      - Sell volume = volume * (high - close) / (high - low)
      CVD delta per bar = buy_vol - sell_vol
    """
    @staticmethod
    def calculate(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        try:
            if df is None or len(df) < 10:
                return {"cvd_trend": "NEUTRAL", "cvd_value": 0.0, "cvd_series": []}
            tail = df.tail(lookback).copy()
            hl_range = (tail["high"] - tail["low"]).replace(0, np.nan)
            buy_vol  = tail["volume"] * (tail["close"] - tail["low"])  / hl_range
            sell_vol = tail["volume"] * (tail["high"] - tail["close"]) / hl_range
            delta    = buy_vol.fillna(0) - sell_vol.fillna(0)
            cvd      = delta.cumsum()
            cvd_last = float(cvd.iloc[-1])
            cvd_mean = float(cvd.mean())
            cvd_std  = float(cvd.std()) if cvd.std() > 0 else 1.0
            cvd_z    = (cvd_last - cvd_mean) / cvd_std

            # Slope of last 10 bars for trend direction
            if len(cvd) >= 10:
                slope_vals = cvd.values[-10:]
                x = np.arange(len(slope_vals))
                slope = np.polyfit(x, slope_vals, 1)[0]
            else:
                slope = 0.0

            if slope > 0 and cvd_z > 0.3:
                trend = "BULLISH"
            elif slope < 0 and cvd_z < -0.3:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            return {
                "cvd_trend": trend,
                "cvd_value": round(cvd_last, 0),
                "cvd_z": round(cvd_z, 2),
                "cvd_slope": round(slope, 1),
            }
        except Exception as e:
            logger.warning(f"CVD error: {e}")
            return {"cvd_trend": "NEUTRAL", "cvd_value": 0.0, "cvd_z": 0.0, "cvd_slope": 0.0}


# ════════════════════════════════════════════════════════════════
#  HMM-STYLE REGIME (v6 new — statistical Markov proxy)
# ════════════════════════════════════════════════════════════════
class HMMRegimeDetector:
    """
    Gaussian HMM proxy using return distribution statistics.
    We model two hidden states: Trend (directional) vs Chop (mean-reverting).
    Criteria:
      - Hurst exponent > 0.55 → persistent / trending
      - Hurst exponent < 0.45 → mean-reverting / choppy
      - 0.45–0.55            → neutral / transitioning
    Combined with ADX and autocorrelation of returns.
    """
    @staticmethod
    def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
        """Estimate Hurst exponent via R/S analysis."""
        try:
            lags = range(2, min(max_lag, len(ts) // 2))
            tau = []
            for lag in lags:
                tau.append(np.std(np.subtract(ts[lag:], ts[:-lag])))
            if len(tau) < 2 or np.all(np.array(tau) == 0):
                return 0.5
            reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return float(reg[0])
        except Exception:
            return 0.5

    @staticmethod
    def autocorr(ts: np.ndarray, lag: int = 1) -> float:
        try:
            if len(ts) < lag + 2:
                return 0.0
            return float(np.corrcoef(ts[:-lag], ts[lag:])[0, 1])
        except Exception:
            return 0.0

    @classmethod
    def detect(cls, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            rets = df["close"].pct_change().dropna()
            if len(rets) < 30:
                return {"hmm_regime": "Trend", "hmm_confidence": 0.5,
                        "hurst": 0.5, "autocorr": 0.0}

            log_prices = np.log(df["close"].values[-60:] + 1e-9)
            hurst = cls.hurst_exponent(log_prices)

            ar1 = cls.autocorr(rets.values[-40:], lag=1)

            # ADX from feature engine
            adx_s = FeatureEngine.adx(df, 14)
            adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 20.0

            # Vote system
            score = 0.0
            if hurst > 0.55:   score += 1.5
            elif hurst < 0.45: score -= 1.5
            if ar1 > 0.08:     score += 1.0    # positive autocorr = trend
            elif ar1 < -0.08:  score -= 1.0    # negative autocorr = mean-revert
            if adx_v > 28:     score += 1.0
            elif adx_v < 18:   score -= 1.0

            confidence = min(0.95, max(0.35, 0.5 + abs(score) * 0.08))

            if score > 0.5:
                regime = "Trend"
            elif score < -0.5:
                regime = "Chop"
            else:
                regime = "Transitioning"

            return {
                "hmm_regime": regime,
                "hmm_confidence": round(confidence, 2),
                "hurst": round(hurst, 3),
                "autocorr": round(ar1, 3),
                "adx": round(adx_v, 1),
            }
        except Exception as e:
            logger.warning(f"HMM error: {e}")
            return {"hmm_regime": "Trend", "hmm_confidence": 0.5, "hurst": 0.5, "autocorr": 0.0}


# ════════════════════════════════════════════════════════════════
#  FEATURE ENGINE
# ════════════════════════════════════════════════════════════════
class FeatureEngine:

    @staticmethod
    def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d  = s.diff()
        g  = d.clip(lower=0).rolling(p).mean()
        ls = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - (100 / (1 + g / (ls + 1e-9)))

    @staticmethod
    def adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        up = h.diff(); dn = -l.diff()
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
        a   = FeatureEngine.atr(df, p)
        pdi = 100 * pd.Series(pdm, index=df.index).rolling(p).mean() / (a + 1e-9)
        ndi = 100 * pd.Series(ndm, index=df.index).rolling(p).mean() / (a + 1e-9)
        dx  = (np.abs(pdi - ndi) / (pdi + ndi + 1e-9)) * 100
        return dx.rolling(p).mean()

    @staticmethod
    def _mins_to_ny_open() -> int:
        now_utc = datetime.utcnow()
        # DST-aware: NY open is always 09:30 NY time = 13:30 UTC (EDT) or 14:30 UTC (EST)
        try:
            import zoneinfo
            from datetime import timezone as _tzmod
            _tz_ny = zoneinfo.ZoneInfo("America/New_York")
            _now_ny = datetime.now(_tzmod.utc).astimezone(_tz_ny)
            _open_ny = _now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
            if _now_ny >= _open_ny:
                _open_ny = (_open_ny + timedelta(days=1))
            delta = int((_open_ny - _now_ny).total_seconds() / 60)
        except Exception:
            # Fallback: EDT Apr-Oct
            _edt = 3 <= now_utc.month <= 11
            ny_open_utc = now_utc.replace(hour=13 if _edt else 14, minute=30, second=0, microsecond=0)
            if now_utc >= ny_open_utc:
                ny_open_utc += timedelta(days=1)
            delta = int((ny_open_utc - now_utc).total_seconds() / 60)
        return min(delta, 1440)

    @classmethod
    def build(cls, df: pd.DataFrame,
              corr: Optional[Dict] = None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c = df["close"]

        for n in [1, 2, 5, 10, 20]:
            f[f"ret_{n}d"] = c.pct_change(n)
        f["log_ret"] = np.log(c / c.shift(1))
        f["gap"]     = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-9)
        f["body"]    = (c - df["open"]) / (df["open"] + 1e-9)
        f["range"]   = (df["high"] - df["low"]) / (c + 1e-9)

        body_sz = (c - df["open"]).abs()
        f["upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)) / (body_sz + 1e-9)
        f["lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"])  / (body_sz + 1e-9)

        atr14 = cls.atr(df, 14)
        atr7  = cls.atr(df, 7)
        f["atr14"]     = atr14 / (c + 1e-9)
        f["atr7"]      = atr7  / (c + 1e-9)
        f["vol10"]     = f["ret_1d"].rolling(10).std()
        f["vol20"]     = f["ret_1d"].rolling(20).std()
        f["vol_ratio"] = f["vol10"] / (f["vol20"] + 1e-9)
        f["vol_regime"]= (f["vol10"] > f["vol10"].rolling(60).quantile(0.75)).astype(int)

        rsi14 = cls.rsi(c, 14)
        f["rsi14"]      = rsi14
        f["rsi7"]       = cls.rsi(c, 7)
        f["rsi_slope"]  = rsi14.diff(3)

        ema9   = c.ewm(span=9).mean()
        ema21  = c.ewm(span=21).mean()
        ema50  = c.ewm(span=50).mean()
        ema200 = c.ewm(span=200).mean()
        for e, s in [(9,ema9),(21,ema21),(50,ema50),(200,ema200)]:
            f[f"ema{e}_dist"] = (c - s) / (s + 1e-9)

        f["ema9_21_cross"]  = (ema9 - ema21)  / (c + 1e-9)
        f["ema21_50_cross"] = (ema21 - ema50) / (c + 1e-9)

        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig  = macd.ewm(span=9).mean()
        f["macd_hist"]  = macd - sig
        f["macd_slope"] = (macd - sig).diff()

        bb_m = c.rolling(20).mean()
        bb_s = c.rolling(20).std()
        f["bb_pos"]   = (c - (bb_m - 2*bb_s)) / ((4*bb_s) + 1e-9)
        f["bb_width"] = (4*bb_s) / (bb_m + 1e-9)

        f["adx14"]    = cls.adx(df, 14)
        f["trending"] = (f["adx14"] > 25).astype(int)

        for n in [5, 10, 20]:
            f[f"mom{n}"] = c / c.shift(n) - 1

        for p in [5, 10, 20]:
            f[f"dist_hi{p}"] = (df["high"].rolling(p).max() - c) / (c + 1e-9)
            f[f"dist_lo{p}"] = (c - df["low"].rolling(p).min())  / (c + 1e-9)

        f["dow"]       = df.index.dayofweek
        f["month"]     = df.index.month
        f["is_mon"]    = (df.index.dayofweek == 0).astype(int)
        f["is_fri"]    = (df.index.dayofweek == 4).astype(int)
        f["month_end"] = df.index.is_month_end.astype(int)
        f["qtr_end"]   = df.index.is_quarter_end.astype(int)

        mins_to_open = cls._mins_to_ny_open()
        f["mins_to_ny_open"] = int(mins_to_open)
        f["near_ny_open"]    = int(mins_to_open < 60)

        if corr:
            for name, cdf in corr.items():
                if cdf is not None and not cdf.empty:
                    try:
                        cr = cdf["close"].pct_change(1).reindex(df.index, method="ffill")
                        f[f"ret_{name}"]  = cr
                        f[f"corr_{name}"] = f["ret_1d"].rolling(20).corr(cr)
                    except Exception:
                        pass
            vix_series = None
            dxy_series = None
            for name, cdf in corr.items():
                if cdf is None or cdf.empty:
                    continue
                try:
                    if name == "VIX":
                        vix_series = cdf["close"].reindex(df.index, method="ffill")
                    if name == "DXY":
                        dxy_series = cdf["close"].reindex(df.index, method="ffill")
                except Exception:
                    pass
            if vix_series is not None:
                vix_ret = vix_series.pct_change(5)
                nq_ret5 = c.pct_change(5)
                f["vix_nq_div"] = vix_ret * nq_ret5
            if dxy_series is not None:
                dxy_ret = dxy_series.pct_change(5)
                nq_ret5 = c.pct_change(5)
                f["dxy_nq_div"] = dxy_ret * nq_ret5

        f["label"] = (c.shift(-1) > c).astype(int)
        f = f.replace([np.inf, -np.inf], np.nan)
        f = f.dropna(thresh=int(len(f.columns) * 0.5))
        return f


# ════════════════════════════════════════════════════════════════
#  ML ENSEMBLE
# ════════════════════════════════════════════════════════════════
class MLEnsemble:

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.wf_metrics: Dict[str, float] = {}
        self.importances: Dict[str, float] = {}

    def _base_models(self) -> Dict[str, Any]:
        return {
            "xgb": xgb.XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.06,
                subsample=0.75, colsample_bytree=0.65,
                min_child_weight=8, reg_alpha=0.2, reg_lambda=1.5,
                eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1),
            "lgb": lgb.LGBMClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.06,
                subsample=0.75, colsample_bytree=0.65,
                min_child_samples=25, reg_alpha=0.2, reg_lambda=1.5,
                random_state=42, n_jobs=-1, verbose=-1),
            "rf":  RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=15,
                max_features="sqrt", random_state=42, n_jobs=-1),
            "et":  ExtraTreesClassifier(
                n_estimators=80, max_depth=5, min_samples_leaf=15,
                max_features="sqrt", random_state=42, n_jobs=-1),
            "lr":  LogisticRegression(C=0.1, max_iter=500, random_state=42),
        }

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        corr  = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop  = [c for c in upper.columns if any(upper[c] > 0.92)]
        X2    = X.drop(columns=drop, errors="ignore")
        rf    = RandomForestClassifier(n_estimators=80, max_depth=4,
                                       random_state=42, n_jobs=-1)
        rf.fit(X2.fillna(0), y)
        imp = pd.Series(rf.feature_importances_, index=X2.columns)
        return imp.nlargest(min(45, len(imp))).index.tolist()

    def walk_forward(self, feat: pd.DataFrame, n_splits: int = 3,
                     min_train: int = 100) -> Dict[str, float]:
        df = feat.dropna()
        if len(df) < min_train + 30:
            return {"accuracy": 0.5, "auc": 0.5}
        X    = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
        y    = df["label"]
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
        accs, aucs = [], []
        # Use only fast models for walk-forward validation
        _wf_models = {
            "xgb": xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.08,
                                      eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1),
            "lgb": lgb.LGBMClassifier(n_estimators=60, max_depth=3, learning_rate=0.08,
                                      random_state=42, n_jobs=-1, verbose=-1),
        }
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < min_train:
                continue
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            sel = self._select_features(Xtr.fillna(0), ytr)
            Xtr, Xte = Xtr[sel].fillna(0), Xte[sel].fillna(0)
            sc   = RobustScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            fold_probs = []
            for m in list(_wf_models.values()):
                try:
                    m.fit(Xtr_s, ytr)
                    fold_probs.append(m.predict_proba(Xte_s)[:, 1])
                except Exception:
                    pass
            if not fold_probs:
                continue
            avg = np.mean(fold_probs, axis=0)
            accs.append(accuracy_score(yte, (avg > 0.5).astype(int)))
            try:
                aucs.append(roc_auc_score(yte, avg))
            except Exception:
                aucs.append(0.5)
        return {
            "accuracy": float(np.mean(accs)) if accs else 0.5,
            "auc":      float(np.mean(aucs)) if aucs else 0.5,
        }

    def train(self, feat: pd.DataFrame) -> Dict[str, float]:
        df = feat.dropna().copy()
        if len(df) < 50:
            return {}
        X   = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
        y   = df["label"]
        sel = self._select_features(X.fillna(0), y)
        self.feature_names = sel
        X   = X[sel].fillna(0)
        self.scaler.fit(X)
        Xs  = self.scaler.transform(X)
        wf  = self.walk_forward(feat)
        self.wf_metrics = wf
        for name, model in self._base_models().items():
            try:
                model.fit(Xs, y)
                self.models[name] = model
            except Exception as e:
                logger.warning(f"Train {name}: {e}")
        imps: Dict[str, List[float]] = {f: [] for f in self.feature_names}
        for m in self.models.values():
            if hasattr(m, "feature_importances_"):
                for fn, iv in zip(self.feature_names, m.feature_importances_):
                    imps[fn].append(iv)
        self.importances = {k: float(np.mean(v)) for k, v in imps.items() if v}
        self.is_trained = True
        return wf

    def predict(self, row: Dict[str, float]) -> Tuple[float, float]:
        if not self.is_trained:
            return 0.5, 0.3
        try:
            x = pd.DataFrame([row])
            for col in self.feature_names:
                if col not in x.columns:
                    x[col] = 0.0
            x  = x[self.feature_names].fillna(0)
            xs = self.scaler.transform(x)
            probs = []
            for m in self.models.values():
                try:
                    probs.append(m.predict_proba(xs)[0][1])
                except Exception:
                    pass
            if not probs:
                return 0.5, 0.3
            mean_p = float(np.mean(probs))
            std_p  = float(np.std(probs)) if len(probs) > 1 else 0.2
            conf   = max(0.1, min(0.95, 1.0 - std_p * 4))
            return mean_p, conf
        except Exception:
            return 0.5, 0.3

    def get_top_factors(self, row: Dict[str, float], bias: str, n: int = 5) -> List[Dict]:
        if not self.importances:
            return []
        try:
            sorted_imp = sorted(self.importances.items(), key=lambda x: x[1], reverse=True)
            top = sorted_imp[:n]
            label_map = {
                "rsi14": "RSI(14)", "rsi7": "RSI(7)", "adx14": "ADX",
                "macd_hist": "MACD Histogram", "macd_slope": "MACD Slope",
                "bb_pos": "Bollinger Band Position", "bb_width": "BB Width",
                "vol_ratio": "Volatility Ratio", "vol_regime": "Vol Regime",
                "ema9_21_cross": "EMA 9/21 Cross", "ema21_50_cross": "EMA 21/50 Cross",
                "ema200_dist": "Distance from EMA200", "ema50_dist": "Distance from EMA50",
                "ret_VIX": "VIX Return", "corr_VIX": "NQ/VIX Correlation",
                "ret_DXY": "DXY Return", "corr_DXY": "NQ/DXY Correlation",
                "vix_nq_div": "VIX/NQ Divergence", "dxy_nq_div": "DXY/NQ Divergence",
                "ret_1d": "1-Day Return", "ret_5d": "5-Day Return",
                "gap": "Gap vs Prior Close", "body": "Candle Body",
                "mins_to_ny_open": "Time to NY Open", "near_ny_open": "Near NY Open",
                "trending": "Trend Strength", "mom10": "10-Day Momentum",
                "vol10": "10-Day Volatility", "ret_2d": "2-Day Return",
            }
            factors = []
            for fname, imp in top:
                lbl = label_map.get(fname, fname.replace("_", " ").title())
                val = row.get(fname, 0.0)
                pct = round(imp * 100, 1)
                if bias == "BULLISH":
                    sign  = "+" if val > 0 else "-"
                    color = "#00D4AA" if val > 0 else "#FF6B6B"
                else:
                    sign  = "-" if val > 0 else "+"
                    color = "#FF6B6B" if val > 0 else "#00D4AA"
                factors.append({
                    "label": lbl,
                    "fname": fname,
                    "pct":   pct,
                    "sign":  sign,
                    "color": color,
                    "val":   round(float(val), 4),
                })
            return factors
        except Exception:
            return []


# ════════════════════════════════════════════════════════════════
#  AI EXPLANATION GENERATOR (v6 new)
# ════════════════════════════════════════════════════════════════
class AIExplainer:
    """
    Generates natural-language explanations for each top driver,
    then synthesizes a cohesive analyst-style paragraph.
    """

    _DRIVER_TEMPLATES: Dict[str, Dict] = {
        "RSI(14)": {
            "bull_high": "RSI(14) is elevated at {val:.0f}, indicating strong momentum — however, it also signals the market may be approaching overbought territory, which typically precedes either a breakout acceleration or a short-term pullback.",
            "bull_low":  "RSI(14) sits at {val:.0f}, leaving significant room for upside momentum to build before reaching overbought conditions. This gives buyers a clear runway.",
            "bear_high": "RSI(14) at {val:.0f} is stretched into overbought territory, a historically reliable precursor to mean-reversion selling pressure in NQ futures.",
            "bear_low":  "RSI(14) reads {val:.0f}, dangerously close to oversold. While bearish momentum is confirmed, a relief bounce risk is elevated.",
            "neutral":   "RSI(14) is neutral at {val:.0f}, offering no strong directional conviction from a momentum perspective.",
        },
        "MACD Histogram": {
            "bull_pos": "The MACD histogram is expanding positively (value: {val:.4f}), confirming accelerating bullish momentum across the intermediate trend.",
            "bull_neg": "The MACD histogram is still in negative territory ({val:.4f}), but the current bullish ML bias anticipates a potential crossover in the coming sessions.",
            "bear_neg": "The MACD histogram is contracting negatively ({val:.4f}), a classic confirmation of intensifying bearish pressure with no near-term reversal signal.",
            "bear_pos": "Despite a still-positive MACD histogram ({val:.4f}), momentum is decelerating — a warning sign that the bull trend is losing conviction.",
        },
        "Bollinger Band Position": {
            "high": "Price is trading near the upper Bollinger Band (BB position: {val:.2f}), suggesting the market is statistically extended. In trending regimes this can persist; in choppy regimes it typically reverts.",
            "low":  "Price is pressing against the lower Bollinger Band (BB position: {val:.2f}), indicating statistically oversold conditions relative to recent volatility.",
            "mid":  "Price sits comfortably within the Bollinger Bands ({val:.2f}), with no immediate mean-reversion signals — momentum has room to extend in either direction.",
        },
        "VIX/NQ Divergence": {
            "bearish": "A critical divergence is detected: NQ has been rising while the VIX is also climbing ({val:.4f}). This 'fear + rally' combination is historically unsustainable and implies smart money is hedging, applying latent downward pressure.",
            "bullish": "VIX is falling as NQ rises — a healthy, aligned bull environment ({val:.4f}). The absence of fear confirms genuine institutional conviction in the current upswing.",
            "neutral": "VIX and NQ are moving in a broadly aligned fashion ({val:.4f}), presenting no divergence signal to contradict the current bias.",
        },
        "EMA 9/21 Cross": {
            "bull":    "The short-term EMA(9) is above EMA(21) ({val:.4f}), a textbook 'golden cross' on the micro timeframe — institutional momentum algos are structurally long.",
            "bear":    "EMA(9) has crossed below EMA(21) ({val:.4f}), triggering systematic selling signals in momentum-following algorithms. The structure is bearish at current.",
            "flat":    "EMA(9) and EMA(21) are nearly equal ({val:.4f}), signaling a compression event. A breakout from this coil is imminent — direction TBD by the next catalyst.",
        },
        "ADX": {
            "strong":  "ADX at {val:.0f} confirms a strong trending environment. Trend-following strategies historically have significant edge when ADX exceeds 25.",
            "weak":    "ADX at {val:.0f} flags a low-momentum, non-trending environment. Range-bound strategies outperform; breakout entries carry elevated failure risk.",
            "neutral": "ADX at {val:.0f} is transitional — neither clearly trending nor clearly ranging. Caution is warranted when entering in either direction.",
        },
        "1-Day Return": {
            "strong_bull": "Yesterday's session closed with a strong {val_pct:.1f}% return, establishing strong upward price history that the ML models weight significantly.",
            "strong_bear": "Yesterday's session fell {val_pct:.1f}%, embedding a bearish data point that is directly influencing the model's current directional probability.",
            "neutral":     "Yesterday's return was modest ({val_pct:.1f}%), providing no overwhelming directional signal on its own.",
        },
        "2-Day Return": {
            "strong_bull": "The two-day return of {val_pct:.1f}% demonstrates sustained buying across multiple sessions — a key input for the ML ensemble's bullish probability.",
            "strong_bear": "A two-day decline of {val_pct:.1f}% reflects persistent selling pressure and is reinforcing the bearish model output.",
            "neutral":     "The two-day price change ({val_pct:.1f}%) is within normal noise; no strong directional inference from this feature alone.",
        },
        "10-Day Volatility": {
            "high":   "10-day realized volatility is elevated ({val:.4f}), consistent with an active institutional participation environment. High-vol regimes amplify both targets and risk.",
            "normal": "10-day volatility ({val:.4f}) is within normal bounds — the current price action is neither panicked nor complacent.",
            "low":    "10-day volatility is compressed ({val:.4f}). Low-vol coiling periods historically precede sharp directional moves; the ML models treat this as a preparatory setup.",
        },
        "Distance from EMA200": {
            "far_above": "Price is significantly above EMA(200) ({val_pct:.1f}%), which is technically extended on a macro basis. Mean-reversion risk increases the further price deviates.",
            "near":      "Price is trading close to EMA(200) ({val_pct:.1f}%), a critical macro level. The market is at a key decision point.",
            "far_below": "Price is trading well below EMA(200) ({val_pct:.1f}%), in technically bearish macro territory. Any rally faces significant structural overhead resistance.",
        },
    }

    @classmethod
    def _explain_driver(cls, factor: Dict, bias: str, rsi: float) -> str:
        fname = factor.get("fname", "")
        label = factor.get("label", fname)
        val   = factor.get("val", 0.0)
        val_pct = val * 100

        # ── RSI ─────────────────────────────────────────────────
        if label == "RSI(14)":
            t = cls._DRIVER_TEMPLATES["RSI(14)"]
            if bias == "BULLISH":
                key = "bull_high" if val > 60 else "bull_low"
            elif bias == "BEARISH":
                key = "bear_high" if val > 60 else "bear_low"
            else:
                key = "neutral"
            return t[key].format(val=val)

        # ── MACD ────────────────────────────────────────────────
        if label == "MACD Histogram":
            t = cls._DRIVER_TEMPLATES["MACD Histogram"]
            if bias == "BULLISH":
                key = "bull_pos" if val > 0 else "bull_neg"
            else:
                key = "bear_neg" if val < 0 else "bear_pos"
            return t[key].format(val=val)

        # ── BB Position ─────────────────────────────────────────
        if label == "Bollinger Band Position":
            t = cls._DRIVER_TEMPLATES["Bollinger Band Position"]
            key = "high" if val > 0.80 else "low" if val < 0.20 else "mid"
            return t[key].format(val=val)

        # ── VIX Divergence ──────────────────────────────────────
        if label == "VIX/NQ Divergence":
            t = cls._DRIVER_TEMPLATES["VIX/NQ Divergence"]
            key = "bearish" if val < -0.0005 else "bullish" if val > 0.0005 else "neutral"
            return t[key].format(val=val)

        # ── EMA Cross ───────────────────────────────────────────
        if label == "EMA 9/21 Cross":
            t = cls._DRIVER_TEMPLATES["EMA 9/21 Cross"]
            key = "bull" if val > 0.001 else "bear" if val < -0.001 else "flat"
            return t[key].format(val=val)

        # ── ADX ─────────────────────────────────────────────────
        if label == "ADX":
            t = cls._DRIVER_TEMPLATES["ADX"]
            key = "strong" if val > 28 else "weak" if val < 18 else "neutral"
            return t[key].format(val=val)

        # ── 1-Day Return ────────────────────────────────────────
        if label == "1-Day Return":
            t = cls._DRIVER_TEMPLATES["1-Day Return"]
            key = "strong_bull" if val > 0.005 else "strong_bear" if val < -0.005 else "neutral"
            return t[key].format(val_pct=val_pct)

        # ── 2-Day Return ────────────────────────────────────────
        if label == "2-Day Return":
            t = cls._DRIVER_TEMPLATES["2-Day Return"]
            key = "strong_bull" if val > 0.008 else "strong_bear" if val < -0.008 else "neutral"
            return t[key].format(val_pct=val_pct)

        # ── 10-Day Volatility ───────────────────────────────────
        if label == "10-Day Volatility":
            t = cls._DRIVER_TEMPLATES["10-Day Volatility"]
            key = "high" if val > 0.015 else "low" if val < 0.006 else "normal"
            return t[key].format(val=val)

        # ── EMA 200 Distance ────────────────────────────────────
        if label == "Distance from EMA200":
            t = cls._DRIVER_TEMPLATES["Distance from EMA200"]
            key = "far_above" if val > 0.03 else "far_below" if val < -0.03 else "near"
            return t[key].format(val_pct=val_pct)

        # ── Generic fallback ────────────────────────────────────
        direction = "bullish" if (bias == "BULLISH" and val > 0) or (bias == "BEARISH" and val < 0) else "bearish"
        return (f"{label} currently reads {val:.4f}, contributing a {direction} signal "
                f"to the ensemble with {factor.get('pct', 0):.1f}% relative importance weight.")

    @classmethod
    def generate(cls, top_factors: List[Dict], bias: str, rsi: float,
                 regime_phase: str, vix: float, cvd_trend: str) -> str:
        if not top_factors:
            return "Insufficient data for AI explanation generation."
        try:
            driver_sentences = []
            for f in top_factors[:3]:
                driver_sentences.append(cls._explain_driver(f, bias, rsi))

            # Bias-level synthesis sentence
            if bias == "BULLISH":
                opening = (
                    f"The ML ensemble is registering a **BULLISH** directional bias with "
                    f"{int(top_factors[0]['pct'])}% of the model's predictive weight driven "
                    f"by the top three features outlined below."
                )
            elif bias == "BEARISH":
                opening = (
                    f"The ML ensemble is registering a **BEARISH** directional bias. "
                    f"The top model drivers collectively point toward downside price pressure "
                    f"in the current {regime_phase.lower()} regime environment."
                )
            else:
                opening = (
                    f"The ML ensemble is registering a **NEUTRAL** directional bias. "
                    f"Competing signals across the feature set are producing an inconclusive "
                    f"directional probability, which is appropriate for the current "
                    f"{regime_phase.lower()} regime."
                )

            # CVD overlay
            cvd_sentence = ""
            if cvd_trend == "BULLISH":
                cvd_sentence = "Order flow analysis (CVD proxy) confirms net aggressive buying — buyers are actively lifting the offer."
            elif cvd_trend == "BEARISH":
                cvd_sentence = "Order flow analysis (CVD proxy) shows net aggressive selling — sellers are actively hitting the bid, which diverges from or reinforces the directional bias."
            else:
                cvd_sentence = "Order flow (CVD proxy) is balanced, with no clear aggressive buyer or seller dominance at current."

            # VIX overlay
            vix_sentence = ""
            if vix > 30:
                vix_sentence = f"With VIX at {vix:.1f}, the broader market is in an elevated fear state — expect wider bid-ask spreads, erratic price action, and false breakouts."
            elif vix < 15:
                vix_sentence = f"VIX at {vix:.1f} reflects complacency and low risk premium — ideal for trend-following but also a warning that a volatility spike may be overdue."
            else:
                vix_sentence = f"VIX at {vix:.1f} is within normal parameters, supporting a standard risk environment."

            full_text = (
                f"{opening}\n\n"
                f"**Primary Driver:** {driver_sentences[0]}\n\n"
                + (f"**Secondary Driver:** {driver_sentences[1]}\n\n" if len(driver_sentences) > 1 else "")
                + (f"**Tertiary Driver:** {driver_sentences[2]}\n\n" if len(driver_sentences) > 2 else "")
                + f"**Order Flow:** {cvd_sentence}\n\n"
                f"**Macro Context:** {vix_sentence}"
            )
            return full_text
        except Exception as e:
            logger.warning(f"AI explain error: {e}")
            return "AI explanation temporarily unavailable."


# ════════════════════════════════════════════════════════════════
#  BIAS INVALIDATION ENGINE (v6 new)
# ════════════════════════════════════════════════════════════════
class BiasInvalidation:
    """
    Computes specific intraday price levels and conditions that, if reached,
    would invalidate the current ML bias. Uses intraday data for relevant levels.
    """
    @staticmethod
    def compute(df_daily: pd.DataFrame, bias: str, atr: float,
                cp: float, regime_phase: str,
                df_intraday: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        try:
            # ── Prefer intraday (15m/1h) data for levels, fall back to daily ──
            df = df_intraday if (df_intraday is not None and len(df_intraday) >= 20) else df_daily

            # Intraday ATR (smaller, more relevant for intraday invalidation)
            if df_intraday is not None and len(df_intraday) >= 14:
                atr_intra = float(FeatureEngine.atr(df_intraday, 14).iloc[-1])
                # Use intraday ATR if it makes sense (not zero, not too tiny)
                if atr_intra > 5:
                    atr = atr_intra

            ema9   = float(df["close"].ewm(span=9).mean().iloc[-1])
            ema21  = float(df["close"].ewm(span=21).mean().iloc[-1])

            # Recent intraday swing (last 10 bars on 15m = ~2.5h of price action)
            swing_lo = float(df["low"].rolling(10).min().iloc[-1])
            swing_hi = float(df["high"].rolling(10).max().iloc[-1])

            # Wider structure: last 20 bars on 15m = ~5h
            pivot_lo_20 = float(df["low"].rolling(20).min().iloc[-1])
            pivot_hi_20 = float(df["high"].rolling(20).max().iloc[-1])

            # Safety: invalidation level must be within 2x intraday ATR of current price
            # If not, clamp to ATR-based level (prevents stale daily levels far from price)
            max_dist = atr * 2.0

            if bias == "BULLISH":
                raw_inv = max(swing_lo, ema21 - atr * 0.15)
                # If level is too far below, use 1 ATR below current price
                if cp - raw_inv > max_dist:
                    raw_inv = round(cp - atr * 0.8, 0)
                inv_level = round(raw_inv, 0)
                condition = (
                    f"BULLISH bias invalidates on a **15M close below {inv_level:,.0f}** "
                    f"(intraday swing low / EMA21 zone — {cp - inv_level:.0f} pts below current price)."
                )
                flip_detail = (
                    f"A confirmed 15M close below **{inv_level:,.0f}** shifts intraday bias to BEARISH. "
                    f"EMA(21) at **{ema21:,.0f}** is the key intraday support. "
                    f"Session structure low at **{pivot_lo_20:,.0f}** — break here means sellers are in control."
                )

            elif bias == "BEARISH":
                raw_inv = min(swing_hi, ema21 + atr * 0.15)
                # If level is too far above, clamp to 1 ATR above current price
                if raw_inv - cp > max_dist:
                    raw_inv = round(cp + atr * 0.8, 0)
                inv_level = round(raw_inv, 0)
                condition = (
                    f"BEARISH bias invalidates on a **15M close above {inv_level:,.0f}** "
                    f"(intraday swing high / EMA21 zone — {inv_level - cp:.0f} pts above current price)."
                )
                flip_detail = (
                    f"A confirmed 15M close above **{inv_level:,.0f}** shifts intraday bias to BULLISH. "
                    f"EMA(21) at **{ema21:,.0f}** is key intraday resistance. "
                    f"Session high at **{pivot_hi_20:,.0f}** — reclaim confirms bullish structure shift."
                )

            else:  # NEUTRAL
                inv_level_up = round(min(pivot_hi_20, cp + atr * 0.8), 0)
                inv_level_dn = round(max(pivot_lo_20, cp - atr * 0.8), 0)
                inv_level    = inv_level_up
                condition = (
                    f"NEUTRAL: breaks BULLISH above **{inv_level_up:,.0f}** (+{inv_level_up - cp:.0f} pts) "
                    f"or BEARISH below **{inv_level_dn:,.0f}** (-{cp - inv_level_dn:.0f} pts)."
                )
                flip_detail = (
                    f"Market coiling between **{inv_level_dn:,.0f}** (support) and **{inv_level_up:,.0f}** (resistance). "
                    f"15M close + retest above **{inv_level_up:,.0f}** = BULLISH flip. "
                    f"15M close below **{inv_level_dn:,.0f}** = BEARISH. No trade until breakout."
                )

            return {
                "invalidation_price":       inv_level,
                "invalidation_condition":   condition,
                "invalidation_flip_detail": flip_detail,
            }
        except Exception as e:
            logger.warning(f"Invalidation error: {e}")
            return {
                "invalidation_price":       round(cp - atr * 0.75, 0),
                "invalidation_condition":   "Bias invalidates on intraday structural break.",
                "invalidation_flip_detail": "Monitor EMA(21) and recent session extremes.",
            }


# ════════════════════════════════════════════════════════════════
#  NEWS SENTIMENT (v6 new)
# ════════════════════════════════════════════════════════════════
class NewsSentimentAnalyzer:
    """Lightweight keyword-based NLP sentiment scorer."""

    @staticmethod
    def score(articles: List[Dict]) -> Tuple[str, float]:
        if not articles:
            return "NEUTRAL", 0.0
        try:
            total_bull = 0
            total_bear = 0
            for a in articles:
                text = (a.get("event", "") + " " + a.get("title", "")).lower()
                bull_hits = sum(1 for w in BULLISH_WORDS if w in text)
                bear_hits = sum(1 for w in BEARISH_WORDS if w in text)
                total_bull += bull_hits
                total_bear += bear_hits

            total = total_bull + total_bear
            if total == 0:
                return "NEUTRAL", 0.0

            net_score = (total_bull - total_bear) / (total + 1e-9)  # -1 to +1

            if net_score > 0.15:
                sentiment = "BULLISH"
            elif net_score < -0.15:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"

            return sentiment, round(net_score, 3)
        except Exception:
            return "NEUTRAL", 0.0


# ════════════════════════════════════════════════════════════════
#  REGIME DETECTOR
# ════════════════════════════════════════════════════════════════
class RegimeDetector:

    @staticmethod
    def _compute_regime(df: pd.DataFrame,
                        vix_df: Optional[pd.DataFrame] = None) -> Dict:
        try:
            c    = df["close"]
            rets = c.pct_change()
            adx_s = FeatureEngine.adx(df, 14)
            adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 20.0

            vol10      = rets.rolling(10).std().iloc[-1]
            hist_std   = rets.std()
            q25        = max(hist_std * 0.6, 1e-6)
            q75        = hist_std * 1.4
            q90        = hist_std * 1.8

            vol_regime = ("Extreme" if vol10 > q90
                          else "High"   if vol10 > q75
                          else "Low"    if vol10 < q25
                          else "Normal")

            vix = 20.0
            risk_mode = "Neutral"
            if vix_df is not None and not vix_df.empty:
                vix = float(vix_df["close"].iloc[-1])
                risk_mode = ("Risk-Off" if vix > 30
                              else "Neutral" if vix > 20
                              else "Risk-On")

            trend_prob = (0.75 if adx_v > 30
                          else 0.55 if adx_v > 20
                          else 0.25)

            last_10 = rets.tail(10).mean()
            bb_m   = c.rolling(20).mean()
            bb_s   = c.rolling(20).std()
            bb_pos = float((c.iloc[-1] - (bb_m.iloc[-1] - 2*bb_s.iloc[-1]))
                           / (4*bb_s.iloc[-1] + 1e-9))

            if adx_v > 25 and abs(last_10) > 0:
                phase = "Trend"
            elif adx_v < 15:
                phase = "Chop"
            elif bb_pos > 0.85:
                phase = "Distribution"
            elif bb_pos < 0.15:
                phase = "Accumulation"
            else:
                phase = "Consolidation"

            label_map = {
                ("High",    "Risk-Off"): "Crisis / Panic",
                ("Extreme", "Risk-Off"): "Extreme Panic",
                ("Normal",  "Risk-On"):  "Bull Trend",
                ("Low",     "Risk-On"):  "Low-Vol Bull",
                ("High",    "Neutral"):  "High-Vol Chop",
                ("Normal",  "Neutral"):  "Normal Market",
                ("Low",     "Neutral"):  "Low-Vol Range",
            }
            regime_label = label_map.get((vol_regime, risk_mode),
                                          f"{vol_regime} Vol / {risk_mode}")

            return {
                "label": regime_label, "phase": phase, "vix": vix,
                "vol_regime": vol_regime, "risk_mode": risk_mode,
                "adx": adx_v, "trend_prob": trend_prob,
            }
        except Exception:
            return {"label": "Normal Market", "phase": "Trend",
                    "vix": 20.0, "vol_regime": "Normal",
                    "risk_mode": "Neutral", "adx": 20.0, "trend_prob": 0.5}

    @classmethod
    def detect_macro_micro(cls, df_daily, df_intraday, df_premarket, vix_df=None) -> Dict:
        macro = cls._compute_regime(df_daily, vix_df)
        if df_intraday is not None and len(df_intraday) >= 20:
            if df_premarket is not None and len(df_premarket) >= 6:
                combined = pd.concat([df_intraday.tail(60), df_premarket, df_premarket])
                combined = combined.sort_index().drop_duplicates()
                micro = cls._compute_regime(combined, vix_df)
            else:
                micro = cls._compute_regime(df_intraday.tail(80), vix_df)
            c       = df_intraday["close"]
            ema9    = c.ewm(span=9).mean()
            ema21   = c.ewm(span=21).mean()
            micro_bull = float(ema9.iloc[-1]) > float(ema21.iloc[-1])
            micro_rsi  = float(FeatureEngine.rsi(c, 14).iloc[-1])
            if micro_bull and micro_rsi > 50:
                micro_bias = "BULLISH"
            elif not micro_bull and micro_rsi < 50:
                micro_bias = "BEARISH"
            else:
                micro_bias = "NEUTRAL"
        else:
            micro = macro.copy()
            micro_bias = "NEUTRAL"
        return {"macro": macro, "micro": micro, "micro_bias": micro_bias}

    @classmethod
    def compute_internals(cls, df_nq, vix_df, dxy_df, qqq_df) -> Dict:
        try:
            nq_ret5    = float(df_nq["close"].pct_change(5).iloc[-1])
            health_score = 0
            vix_div = "ALIGNED"
            if vix_df is not None and not vix_df.empty:
                vix_ret5 = float(vix_df["close"].pct_change(5).iloc[-1])
                if nq_ret5 > 0.01 and vix_ret5 > 0.05:
                    vix_div = "BEARISH DIV"; health_score -= 1
                elif nq_ret5 < -0.01 and vix_ret5 < -0.05:
                    vix_div = "BULLISH DIV"; health_score += 1
            dxy_div = "ALIGNED"
            if dxy_df is not None and not dxy_df.empty:
                dxy_ret5 = float(dxy_df["close"].pct_change(5).iloc[-1])
                if nq_ret5 > 0.01 and dxy_ret5 > 0.01:
                    dxy_div = "DXY HEADWIND"; health_score -= 1
                elif nq_ret5 < -0.01 and dxy_ret5 < -0.01:
                    dxy_div = "DXY TAILWIND"; health_score += 1
            add_proxy = 0.0
            if qqq_df is not None and not qqq_df.empty:
                add_proxy = float(qqq_df["close"].pct_change(1).iloc[-1]) * 1000
            health = "HEALTHY" if health_score >= 1 else "STRESSED" if health_score <= -1 else "NEUTRAL"
            return {
                "vix_divergence": vix_div, "dxy_divergence": dxy_div,
                "internals_health": health, "add_proxy": round(add_proxy, 1),
            }
        except Exception:
            return {"vix_divergence": "N/A", "dxy_divergence": "N/A",
                    "internals_health": "NEUTRAL", "add_proxy": 0.0}


# ════════════════════════════════════════════════════════════════
#  SMART TARGETS
# ════════════════════════════════════════════════════════════════
class SmartTargets:

    @staticmethod
    def find_pivots(df: pd.DataFrame, window: int = 5) -> Tuple[float, float]:
        try:
            highs = df["high"]
            lows  = df["low"]
            cp    = float(df["close"].iloc[-1])
            swing_highs, swing_lows = [], []
            n = len(df)
            for i in range(window, n - window):
                if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                    swing_highs.append(float(highs.iloc[i]))
                if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                    swing_lows.append(float(lows.iloc[i]))
            above = [h for h in swing_highs if h > cp]
            below = [l for l in swing_lows  if l < cp]
            return (min(above) if above else cp + 200,
                    max(below) if below else cp - 200)
        except Exception:
            cp = float(df["close"].iloc[-1])
            return cp + 200, cp - 200

    @staticmethod
    def vwap_bands(df: pd.DataFrame, atr: float) -> Tuple[float, float]:
        try:
            if "volume" not in df.columns:
                cp = float(df["close"].iloc[-1])
                return cp + atr * 1.5, cp - atr * 1.5
            typical   = (df["high"] + df["low"] + df["close"]) / 3
            vol       = df["volume"].replace(0, np.nan).ffill()
            cum_tpv   = (typical * vol).cumsum()
            cum_vol   = vol.cumsum()
            vwap      = (cum_tpv / (cum_vol + 1e-9)).iloc[-1]
            std       = float(df["close"].rolling(20).std().iloc[-1])
            return round(float(vwap) + 2 * std, 2), round(float(vwap) - 2 * std, 2)
        except Exception:
            cp = float(df["close"].iloc[-1])
            return cp + atr * 1.5, cp - atr * 1.5

    @classmethod
    def compute(cls, df_1h, df_15m, cp, atr, conf, vp) -> Dict:
        half_w = atr * 0.10

        def make_range(center):
            lo = round((center - half_w) * 4) / 4
            hi = round((center + half_w) * 4) / 4
            return lo, hi

        def touch_prob(target, cp, atr, conf, bonus=False):
            dist_atr = abs(target - cp) / (atr + 1e-9)
            base     = max(0.35, min(0.95, conf * 0.6 + 0.3))
            decay    = max(0.0, 1.0 - (dist_atr - 0.8) * 0.25)
            return round(min(0.95, max(0.30, base * decay + (0.05 if bonus else 0.0))), 2)

        pivot_hi, pivot_lo = cp + atr * 0.85, cp - atr * 0.85
        t1_label = "T1 (ATR)"
        if df_15m is not None and len(df_15m) >= 20:
            ph, pl = cls.find_pivots(df_15m, window=4)
            if cp < ph < cp + atr * 2.0:
                pivot_hi = ph; t1_label = "T1 (Pivot)"
            if cp - atr * 2.0 < pl < cp:
                pivot_lo = pl; t1_label = "T1 (Pivot)"
        elif df_1h is not None and len(df_1h) >= 20:
            ph, pl = cls.find_pivots(df_1h, window=3)
            if cp < ph < cp + atr * 2.5:
                pivot_hi = ph; t1_label = "T1 (Pivot)"
            if cp - atr * 2.5 < pl < cp:
                pivot_lo = pl; t1_label = "T1 (Pivot)"

        vwap_hi = cp + atr * 1.60
        vwap_lo = cp - atr * 1.60
        t2_label = "T2 (ATR)"
        if df_1h is not None and len(df_1h) >= 10:
            vh, vl = cls.vwap_bands(df_1h, atr)
            if cp < vh < cp + atr * 3.5:
                vwap_hi = vh; t2_label = "T2 (VWAP+2σ)"
            if cp - atr * 3.5 < vl < cp:
                vwap_lo = vl; t2_label = "T2 (VWAP-2σ)"
        if vp.get("poc", 0) > 0:
            poc = vp["poc"]
            if cp < poc < cp + atr * 2.5:
                vwap_hi = poc; t2_label = "T2 (POC)"
            elif cp - atr * 2.5 < poc < cp:
                vwap_lo = poc; t2_label = "T2 (POC)"

        u1_lo, u1_hi = make_range(pivot_hi)
        u2_lo, u2_hi = make_range(vwap_hi)
        l1_lo, l1_hi = make_range(pivot_lo)
        l2_lo, l2_hi = make_range(vwap_lo)
        return {
            "u1_lo": u1_lo, "u1_hi": u1_hi, "u1_label": t1_label,
            "u2_lo": u2_lo, "u2_hi": u2_hi, "u2_label": t2_label,
            "l1_lo": l1_lo, "l1_hi": l1_hi, "l1_label": t1_label,
            "l2_lo": l2_lo, "l2_hi": l2_hi, "l2_label": t2_label,
            "u1_conf": touch_prob(pivot_hi, cp, atr, conf),
            "u2_conf": touch_prob(vwap_hi,  cp, atr, conf),
            "l1_conf": touch_prob(pivot_lo, cp, atr, conf),
            "l2_conf": touch_prob(vwap_lo,  cp, atr, conf),
        }


# ════════════════════════════════════════════════════════════════
#  TIME ESTIMATOR
# ════════════════════════════════════════════════════════════════
class TimeEstimator:
    HIGH_DIST: Dict[Tuple[int,int], float] = {
        (9,0):0.08,(9,1):0.10,(9,2):0.09,(9,3):0.07,
        (10,0):0.07,(10,1):0.07,(10,2):0.05,(10,3):0.04,
        (11,0):0.04,(11,1):0.04,(11,2):0.03,(11,3):0.03,
        (13,0):0.04,(13,1):0.05,(13,2):0.04,(13,3):0.03,
        (14,0):0.03,(14,1):0.03,(14,2):0.03,(14,3):0.03,
        (15,0):0.03,(15,1):0.03,(15,2):0.02,(15,3):0.02,
        (2,0):0.02,(3,0):0.02,(4,0):0.02,
        (8,0):0.03,(8,1):0.03,(8,2):0.02,(8,3):0.02,
    }
    LOW_DIST: Dict[Tuple[int,int], float] = {
        (9,0):0.06,(9,1):0.08,(9,2):0.07,(9,3):0.06,
        (10,0):0.06,(10,1):0.05,(10,2):0.04,(10,3):0.04,
        (11,0):0.04,(11,1):0.03,(11,2):0.03,(11,3):0.03,
        (12,0):0.04,(12,1):0.05,(12,2):0.04,(12,3):0.03,
        (13,0):0.05,(13,1):0.06,(13,2):0.05,(13,3):0.04,
        (14,0):0.04,(14,1):0.04,(14,2):0.03,(14,3):0.03,
        (15,0):0.03,(15,1):0.03,(15,2):0.02,(15,3):0.02,
        (2,0):0.02,(3,0):0.02,(4,0):0.02,
        (8,0):0.03,(8,1):0.02,(8,2):0.02,(8,3):0.02,
    }

    @classmethod
    def estimate(cls, bias, regime_phase, vol_regime) -> Tuple[str, str]:
        h_dist = dict(cls.HIGH_DIST)
        l_dist = dict(cls.LOW_DIST)
        if bias == "BULLISH":
            for k in [(9,1),(9,2),(10,0)]:
                l_dist[k] = l_dist.get(k, 0) * 1.4
            for k in [(13,1),(14,0),(15,0)]:
                h_dist[k] = h_dist.get(k, 0) * 1.3
        elif bias == "BEARISH":
            for k in [(9,1),(9,2),(10,0)]:
                h_dist[k] = h_dist.get(k, 0) * 1.4
            for k in [(13,1),(14,0),(15,0)]:
                l_dist[k] = l_dist.get(k, 0) * 1.3
        if regime_phase == "Trend":
            for k in [(9,1),(9,2)]:
                h_dist[k] = h_dist.get(k, 0) * 1.2
                l_dist[k] = l_dist.get(k, 0) * 1.2

        def pick_slot(dist):
            total = sum(dist.values())
            probs = {k: v/total for k, v in dist.items()}
            return max(probs, key=probs.get)

        def fmt(slot):
            h, q = slot
            return f"{h:02d}:{q*15:02d} NY"

        return fmt(pick_slot(h_dist)), fmt(pick_slot(l_dist))

    @staticmethod
    def bias_horizon(bias, regime_phase, vol_regime, mins_to_open) -> str:
        if mins_to_open < 60:
            return "NY Open Window (Next 2h)"
        if regime_phase == "Trend":
            return "Intraday (Next 2-4h)" if vol_regime in ("High","Extreme") else "Intraday (NY Session)"
        if regime_phase == "Chop":
            return "Short-term (Next 1-2h)"
        if regime_phase in ("Accumulation","Distribution"):
            return "Swing (1-3 Days)"
        return "Intraday (Next 4-6h)"


# ════════════════════════════════════════════════════════════════
#  EXPECTED VOLUME ESTIMATOR
# ════════════════════════════════════════════════════════════════
class ExpectedVolume:
    """
    Estimates relative volume intensity per 30-min NY slot.
    Based on empirical NQ intraday volume distribution.
    """

    _SLOTS: List[Tuple[int, int, float, str]] = [
        (8,  0,  0.55, "08:00–08:30"),
        (8, 30,  0.80, "08:30–09:00"),
        (9,  0,  1.10, "09:00–09:30"),
        (9, 30,  2.50, "09:30–10:00"),
        (10, 0,  2.20, "10:00–10:30"),
        (10,30,  1.70, "10:30–11:00"),
        (11, 0,  1.20, "11:00–11:30"),
        (11,30,  0.90, "11:30–12:00"),
        (12, 0,  0.65, "12:00–12:30"),
        (12,30,  0.60, "12:30–13:00"),
        (13, 0,  0.70, "13:00–13:30"),
        (13,30,  0.85, "13:30–14:00"),
        (14, 0,  1.00, "14:00–14:30"),
        (14,30,  1.30, "14:30–15:00"),
        (15, 0,  1.80, "15:00–15:30"),
        (15,30,  2.10, "15:30–16:00"),
        (16, 0,  0.50, "16:00–16:30"),
        (16,30,  0.35, "16:30–17:00"),
    ]

    @classmethod
    def compute(cls, bias: str, vol_regime: str, now_ny: datetime) -> List[Dict]:
        vol_mult = {"Extreme": 1.35, "High": 1.20, "Normal": 1.0, "Low": 0.75}.get(vol_regime, 1.0)
        now_mins = now_ny.hour * 60 + now_ny.minute

        result = []
        for (h, m, base_rv, label) in cls._SLOTS:
            slot_mins = h * 60 + m
            if slot_mins < now_mins - 30:
                continue
            if len(result) >= 8:
                break
            rv = base_rv * vol_mult
            if bias == "BULLISH" and h == 9 and m == 30:
                rv *= 1.10
            elif bias == "BEARISH" and h == 15 and m == 30:
                rv *= 1.10
            tier = "HIGH" if rv >= 1.5 else "LOW" if rv < 0.75 else "NORMAL"
            is_now = (now_mins >= slot_mins) and (now_mins < slot_mins + 30)
            result.append({
                "slot":    label,
                "rel_vol": round(rv, 2),
                "tier":    tier,
                "now":     is_now,
            })
        return result


# ════════════════════════════════════════════════════════════════
#  SESSION OUTLOOK
# ════════════════════════════════════════════════════════════════
def build_session_outlook(bias: str, regime: Dict, rsi: float) -> Dict[str, str]:
    b   = bias
    vr  = regime.get("vol_regime", "Normal")
    phase = regime.get("phase", "Trend")
    adx = regime.get("adx", 20.0)

    def asia_text():
        if vr in ("High","Extreme"):
            return (
                f"Overnight Asia session is expected to carry elevated volatility — a direct "
                f"consequence of the current {vr.lower()} volatility regime. Expect a session "
                f"range of roughly 60–90 pts. Watch for false breakouts during thin liquidity "
                f"windows (approx 20:00–01:00 NY). Key Asian support/resistance levels may be "
                f"tested aggressively before NY reasserts direction."
            )
        elif vr == "Low":
            return (
                "Asia overnight session is anticipated to be exceptionally quiet, reflecting the "
                "current low-volatility regime and compressed ATR. Range likely 15–30 pts with "
                "price coiling near the prior session close. Watch for a potential stop-hunt at "
                "obvious levels before the London open injects meaningful liquidity."
            )
        return (
            f"Asia session sets up as a controlled ranging environment with moderate liquidity. "
            f"Current regime ({vr} vol) suggests typical overnight ranges of 30–55 pts around "
            f"the prior day's close. ML bias is {b}, which may result in a slight directional drift."
        )

    def london_text():
        if b == "BULLISH":
            return (
                "The London open is statistically the session most likely to establish the intraday "
                "low on a bullish ML day. Expect an initial 'Judas swing' — a brief dip below the "
                f"Asian range low to trigger stop-losses — before a sustained push higher "
                f"into the NY overlap. With RSI at {rsi:.0f} and a {phase} regime, London "
                "buyers are likely to step in aggressively near Asian lows."
            )
        elif b == "BEARISH":
            return (
                f"London open on a bearish ML day historically shows an initial push higher — a "
                f"'Judas rally' — before the real downside move develops. With RSI at {rsi:.0f} "
                f"and {vr.lower()} volatility, this false rally is likely contained within 40–70 pts "
                "above the Asia close before sellers reassert control."
            )
        return (
            f"With a NEUTRAL ML bias and ADX at {adx:.0f}, London is expected to deliver a choppy, "
            "two-sided session. Focus on identifying the developing range boundaries rather than "
            "fading or breakout trading."
        )

    def ny_open_text():
        if b == "BULLISH" and rsi < 65:
            return (
                f"NY Open is the highest-probability window for the primary bull leg. "
                f"With RSI at {rsi:.0f} (room to run), expect a momentum push through the "
                f"London high within the first 60–90 minutes. The 09:30–10:30 NY window "
                f"historically captures the majority of the daily range on trending days."
            )
        elif b == "BEARISH" and rsi > 35:
            return (
                f"NY Open sets up as high-risk, high-velocity on this bearish ML day. "
                f"RSI at {rsi:.0f} gives room to extend lower. A confirmed break with volume "
                f"expansion below the London low initiates the primary bear leg."
            )
        elif phase == "Chop":
            return (
                "NY Open on a choppy regime day is treacherous. Expect the first 30 minutes to "
                "establish a range that is then violated in both directions. Professional traders "
                "often wait for a 10:15–11:00 NY setup where institutional intent becomes clearer."
            )
        return (
            f"NY Open carries moderate directional conviction. With the {phase} regime and "
            f"RSI at {rsi:.0f}, the open is likely to test either the London high or low within "
            f"the first 30 minutes. Wait for the first 30-minute range then trade the breakout."
        )

    def ny_pm_text():
        if vr == "Low":
            return (
                "NY afternoon session is expected to drift lazily in the direction of the morning "
                "bias. The 15:00–15:30 NY window may see a brief uptick in activity as MOC orders "
                "hit. PM session is not the time for new directional entries — protect winners."
            )
        elif vr in ("High","Extreme"):
            return (
                f"With a {vr.lower()} volatility regime still active, the NY afternoon carries real "
                "reversal risk. The 13:00–14:00 NY window has historically produced sharp "
                "counter-trend moves. Be alert for a potential 60–100 pt reversal from the morning extreme."
            )
        return (
            f"NY PM session sets up with moderate reversal risk. The primary driver will be whether "
            f"the morning's ML-driven move achieved its initial target levels. If T1 was reached, "
            "expect consolidation or mild profit-taking into the close."
        )

    return {
        "Asia":    asia_text(),
        "London":  london_text(),
        "NY Open": ny_open_text(),
        "NY PM":   ny_pm_text(),
    }


# ════════════════════════════════════════════════════════════════
#  HARDENED RSS NEWS FETCHER (v6 rewrite)
# ════════════════════════════════════════════════════════════════
_ROBUST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

RSS_SOURCES = [
    ("Yahoo Finance Markets",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NQ=F,^IXIC,^GSPC&region=US&lang=en-US"),
    ("Yahoo Finance Top",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,^VIX&region=US&lang=en-US"),
    ("MarketWatch Pulse",
     "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
    ("MarketWatch Top",
     "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC Markets",
     "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("CNBC Finance",
     "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    ("Reuters Markets",
     "https://feeds.reuters.com/reuters/businessNews"),
    ("Investing.com",
     "https://www.investing.com/rss/news_25.rss"),
]

MARKET_KEYWORDS = [
    "fed", "rate", "inflation", "cpi", "gdp", "jobs", "nasdaq", "nq",
    "s&p", "market", "tech", "stock", "futures", "treasury", "yield",
    "fomc", "powell", "equity", "trade", "tariff", "earnings", "macro",
    "bond", "dollar", "recession", "ism", "pce", "nonfarm", "payroll",
]


@st.cache_data(ttl=600)
def fetch_market_news() -> List[Dict]:
    import re as _re2
    import html as _html2

    def _strip_html(text: str) -> str:
        text = _re2.sub(r'<[^>]+>', '', text)
        return _html2.unescape(text).strip()

    articles = []
    feedparser.USER_AGENT = _ROBUST_HEADERS["User-Agent"]

    for source_name, url in RSS_SOURCES:
        if len(articles) >= 15:
            break
        try:
            feed = feedparser.parse(url)
            if not feed or not hasattr(feed, "entries"):
                continue
            for entry in feed.entries[:8]:
                title = _strip_html(entry.get("title", "").strip())
                if not title:
                    continue
                pub   = entry.get("published", entry.get("updated", ""))
                link  = entry.get("link", "")
                summary = _strip_html(entry.get("summary", ""))
                full_text = (title + " " + summary).lower()
                if not any(kw in full_text for kw in MARKET_KEYWORDS):
                    continue
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub)
                    time_str = dt.strftime("%H:%M")
                except Exception:
                    try:
                        import re as _re
                        m = _re.search(r"(\d{2}:\d{2})", pub)
                        time_str = m.group(1) if m else "—"
                    except Exception:
                        time_str = "—"
                articles.append({
                    "time":     time_str,
                    "event":    title[:120],
                    "source":   source_name,
                    "link":     link,
                    "forecast": "—",
                    "previous": "—",
                })
        except Exception as e:
            logger.warning(f"RSS {source_name}: {e}")
            continue

    # Deduplicate
    seen   = set()
    unique = []
    for a in articles:
        key = a["event"][:60].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    if len(unique) < 3:
        fallback = _fetch_fallback_news()
        for item in fallback:
            key = item["event"][:60].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(item)

    return unique[:12]


def _fetch_fallback_news() -> List[Dict]:
    """Multiple fallback strategies for news."""
    results = []

    # Fallback 1: Forex Factory high-impact USD
    try:
        url = "https://www.forexfactory.com/calendar"
        r   = requests.get(url, headers=_ROBUST_HEADERS, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("tr.calendar__row")
            last_time = "—"
            for row in rows[:50]:
                imp_tag = row.select_one("td.calendar__impact span")
                if not imp_tag:
                    continue
                cls_str = " ".join(imp_tag.get("class", []))
                if "high" not in cls_str.lower():
                    continue
                cur_tag = row.select_one("td.calendar__currency")
                if not cur_tag or "USD" not in cur_tag.text:
                    continue
                time_tag = row.select_one("td.calendar__time")
                if time_tag and time_tag.text.strip():
                    last_time = time_tag.text.strip()
                name_tag = row.select_one(
                    "td.calendar__event span.calendar__event-title")
                if not name_tag:
                    continue
                fc_tag = row.select_one("td.calendar__forecast")
                pr_tag = row.select_one("td.calendar__previous")
                results.append({
                    "time":     last_time,
                    "event":    name_tag.text.strip()[:100],
                    "source":   "Forex Factory",
                    "link":     "",
                    "forecast": fc_tag.text.strip() if fc_tag else "—",
                    "previous": pr_tag.text.strip() if pr_tag else "—",
                })
                if len(results) >= 6:
                    break
    except Exception as e:
        logger.warning(f"FF fallback failed: {e}")

    # Fallback 2: Minimal static placeholder
    if not results:
        results = [{
            "time":     "—",
            "event":    "Live news feeds temporarily unavailable. Retry in 60 seconds.",
            "source":   "System",
            "link":     "",
            "forecast": "—",
            "previous": "—",
        }]

    return results


# ════════════════════════════════════════════════════════════════
#  CHART BUILDER (with toggle support)
# ════════════════════════════════════════════════════════════════
def build_lwc_chart(df: pd.DataFrame, pred: Prediction, tf: str,
                    show_projection: bool,
                    show_vp: bool = True,
                    show_targets: bool = True) -> str:
    if df is None or df.empty:
        return "<div style='color:#9CA3AF;padding:40px;text-align:center;'>Chart data unavailable.</div>"

    df   = df.dropna(subset=["open","high","low","close"]).copy()
    max_bars = {"1M":200,"5M":200,"15M":150,"30M":120,"1H":120,"4H":100,"D":200}
    df   = df.tail(max_bars.get(tf, 150))

    cp      = pred.current_price
    atr     = pred.atr
    is_bull = pred.bias == "BULLISH"
    cfg     = TF_CONFIG[tf]
    proj_bars = cfg["proj_bars"]

    candles = []
    for idx, row in df.iterrows():
        t = idx.strftime("%Y-%m-%d") if tf == "D" else int(idx.timestamp())
        candles.append({
            "time":  t,
            "open":  round(float(row["open"]),  2),
            "high":  round(float(row["high"]),  2),
            "low":   round(float(row["low"]),   2),
            "close": round(float(row["close"]), 2),
        })

    proj_center, proj_upper, proj_lower = [], [], []
    if show_projection and cp > 0 and len(df) > 0:
        last_ts  = df.index[-1]
        bar_secs = {"1M":60,"5M":300,"15M":900,"30M":1800,"1H":3600,"4H":14400,"D":86400}.get(tf, 3600)
        drift    = (atr / 24) * (bar_secs / 3600) * (1 if is_bull else -1)
        cone     = atr * 0.06
        for i in range(1, proj_bars + 1):
            if tf == "D":
                ts = (last_ts + timedelta(days=i)).strftime("%Y-%m-%d")
            else:
                ts = int((last_ts + timedelta(seconds=bar_secs * i)).timestamp())
            proj_center.append({"time": ts, "value": round(cp + drift * i, 2)})
            proj_upper.append({"time": ts, "value": round(cp + cone * i * 1.3, 2)})
            proj_lower.append({"time": ts, "value": round(cp - cone * i * 1.3, 2)})

    targets = []
    if show_targets:
        if pred.upper_t2_hi > 0:
            targets.append({"price": round((pred.upper_t2_lo + pred.upper_t2_hi)/2, 2),
                             "color":"#00FFCC","label":f"T2▲ {pred.upper_t2_lo:,.0f}–{pred.upper_t2_hi:,.0f}","style":1})
        if pred.upper_t1_hi > 0:
            targets.append({"price": round((pred.upper_t1_lo + pred.upper_t1_hi)/2, 2),
                             "color":"#00D4AA","label":f"T1▲ {pred.upper_t1_lo:,.0f}–{pred.upper_t1_hi:,.0f}","style":1})
    if cp > 0:
        targets.append({"price": round(cp,2),"color":"#FFB800",
                         "label":f"NOW {cp:,.2f}","style":0})
    if show_targets:
        if pred.lower_t1_hi > 0:
            targets.append({"price": round((pred.lower_t1_lo + pred.lower_t1_hi)/2, 2),
                             "color":"#FF6B6B","label":f"T1▼ {pred.lower_t1_lo:,.0f}–{pred.lower_t1_hi:,.0f}","style":1})
        if pred.lower_t2_hi > 0:
            targets.append({"price": round((pred.lower_t2_lo + pred.lower_t2_hi)/2, 2),
                             "color":"#FF4B4B","label":f"T2▼ {pred.lower_t2_lo:,.0f}–{pred.lower_t2_hi:,.0f}","style":1})

    vp_lines = []
    if show_vp:
        if pred.poc > 0:
            vp_lines.append({"price":pred.poc,"color":"#FFB800","label":f"POC {pred.poc:,.0f}","style":2,"width":1})
        if pred.vah > 0:
            vp_lines.append({"price":pred.vah,"color":"#9B59B6","label":f"VAH {pred.vah:,.0f}","style":2,"width":1})
        if pred.val > 0:
            vp_lines.append({"price":pred.val,"color":"#3498DB","label":f"VAL {pred.val:,.0f}","style":2,"width":1})

    bias_clr     = "#00D4AA" if is_bull else "#FF4B4B"
    candles_json = json.dumps(candles)
    proj_c_json  = json.dumps(proj_center)
    proj_u_json  = json.dumps(proj_upper)
    proj_l_json  = json.dumps(proj_lower)
    targets_json = json.dumps(targets)
    vp_json      = json.dumps(vp_lines)
    proj_enabled = "true" if show_projection else "false"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#0A0A10;overflow:hidden;}}
#chart{{width:100%;height:440px;}}</style></head>
<body><div id="chart"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function(){{
  const chart=LightweightCharts.createChart(document.getElementById('chart'),{{
    width:document.getElementById('chart').clientWidth,height:440,
    layout:{{background:{{color:'#0A0A10'}},textColor:'#9CA3AF',
             fontFamily:"'JetBrains Mono',monospace",fontSize:11}},
    grid:{{vertLines:{{color:'#1F2937',style:1}},horzLines:{{color:'#1F2937',style:1}}}},
    crosshair:{{mode:LightweightCharts.CrosshairMode.Normal,
      vertLine:{{color:'#4B5563',width:1,style:1,labelBackgroundColor:'#1F2937'}},
      horzLine:{{color:'#4B5563',width:1,style:1,labelBackgroundColor:'#1F2937'}}}},
    rightPriceScale:{{borderColor:'#1F2937',scaleMargins:{{top:0.08,bottom:0.08}}}},
    timeScale:{{borderColor:'#1F2937',timeVisible:true,secondsVisible:false}},
    handleScroll:{{mouseWheel:true,pressedMouseMove:true}},
    handleScale:{{axisPressedMouseMove:true,mouseWheel:true,pinch:true}},
  }});
  const cs=chart.addCandlestickSeries({{
    upColor:'#00D4AA',downColor:'#FF4B4B',
    borderUpColor:'#00D4AA',borderDownColor:'#FF4B4B',
    wickUpColor:'#00D4AA',wickDownColor:'#FF4B4B',
  }});
  cs.setData({candles_json});
  {targets_json}.forEach(t=>cs.createPriceLine({{price:t.price,color:t.color,
    lineWidth:1,lineStyle:t.style,axisLabelVisible:true,title:t.label}}));
  {vp_json}.forEach(v=>cs.createPriceLine({{price:v.price,color:v.color,
    lineWidth:v.width||1,lineStyle:v.style||2,axisLabelVisible:true,title:v.label}}));
  if({proj_enabled}){{
    const pc={proj_c_json},pu={proj_u_json},pl={proj_l_json};
    if(pc.length>0){{
      chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,lineStyle:2,
        lastValueVisible:false,priceLineVisible:false}}).setData(pu);
      chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,lineStyle:2,
        lastValueVisible:false,priceLineVisible:false}}).setData(pl);
      chart.addLineSeries({{color:'{bias_clr}',lineWidth:2,
        lineStyle:LightweightCharts.LineStyle.Dashed,
        lastValueVisible:true,priceLineVisible:false,title:'Projection'
      }}).setData(pc);
    }}
  }}
  window.addEventListener('resize',()=>
    chart.applyOptions({{width:document.getElementById('chart').clientWidth}}));
  chart.timeScale().fitContent();
}})();
</script></body></html>"""
    return html


# ════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════
class PredictionEngine:

    def __init__(self):
        self.dp    = DataPipeline()
        self.fe    = FeatureEngine()
        self.rd    = RegimeDetector()
        self.ml    = MLEnsemble()
        self.ready = False

    def _cache_key(self, df_daily: pd.DataFrame) -> str:
        return hashlib.md5(str(df_daily.index[-1]).encode()).hexdigest()[:8]

    def initialize(self) -> bool:
        try:
            df_daily = None
            for period in ("1y","6mo","2y"):
                df_daily = self.dp.fetch(SYMBOLS["NQ"], period, "1d")
                if df_daily is not None and len(df_daily) >= 60:
                    break
            if df_daily is None or len(df_daily) < 60:
                logger.error(f"NQ daily data unavailable: {len(df_daily) if df_daily is not None else 0} rows")
                return False
            corr = self.dp.fetch_correlated()
            feat = self.fe.build(df_daily, corr)
            if len(feat) < 50:
                logger.error(f"Feature set too small: {len(feat)} rows")
                return False
            dk = self._cache_key(df_daily)
            if MODEL_CACHE.exists():
                try:
                    with open(MODEL_CACHE, "rb") as f:
                        cache = pickle.load(f)
                    if cache.get("hash") == dk:
                        self.ml    = cache["model"]
                        self.ready = True
                        return True
                except Exception:
                    pass
            self.ml.train(feat)
            try:
                with open(MODEL_CACHE, "wb") as f:
                    pickle.dump({"hash": dk, "model": self.ml}, f)
            except Exception:
                pass
            self.ready = True
            return True
        except Exception as e:
            logger.error(f"Init error: {e}")
            return False

    def predict(self) -> Optional[Prediction]:
        try:
            df_d = None
            for period in ("1y","6mo","2y"):
                df_d = self.dp.fetch(SYMBOLS["NQ"], period, "1d")
                if df_d is not None and len(df_d) >= 60:
                    break
            if df_d is None or len(df_d) < 20:
                return None

            corr   = self.dp.fetch_correlated()
            vix_df = corr.get("VIX")
            dxy_df = corr.get("DXY")
            qqq_df = corr.get("QQQ")

            feat = self.fe.build(df_d, corr)
            if feat.empty:
                return None

            latest_feat  = feat.drop(columns=["label"], errors="ignore")
            latest_feat  = latest_feat.select_dtypes(include=[np.number])
            last_row     = latest_feat.iloc[-1].to_dict()

            prob_bull, conf = self.ml.predict(last_row)
            prob_bear       = 1 - prob_bull

            cp      = float(df_d["close"].iloc[-1])
            atr     = float(FeatureEngine.atr(df_d, 14).iloc[-1])
            rsi_val = float(FeatureEngine.rsi(df_d["close"], 14).iloc[-1])

            # ── Intraday-weighted bias blend ─────────────────────
            df_15m_early = self.dp.fetch_intraday("15m", "5d")
            intraday_bull_score = 0.0
            if df_15m_early is not None and len(df_15m_early) >= 20:
                _c15 = df_15m_early["close"]
                _ema9_15  = float(_c15.ewm(span=9).mean().iloc[-1])
                _ema21_15 = float(_c15.ewm(span=21).mean().iloc[-1])
                _rsi15    = float(FeatureEngine.rsi(_c15, 14).iloc[-1])
                _ret1h    = float(_c15.pct_change(4).iloc[-1])
                _ret30m   = float(_c15.pct_change(2).iloc[-1])
                intraday_bull_score = 0.5
                if _ema9_15 > _ema21_15:  intraday_bull_score += 0.15
                else:                      intraday_bull_score -= 0.15
                if _rsi15 > 55:            intraday_bull_score += 0.15
                elif _rsi15 < 45:          intraday_bull_score -= 0.15
                if _ret1h > 0:             intraday_bull_score += 0.10
                elif _ret1h < 0:           intraday_bull_score -= 0.10
                if _ret30m > 0:            intraday_bull_score += 0.10
                elif _ret30m < 0:          intraday_bull_score -= 0.10
                intraday_bull_score = max(0.0, min(1.0, intraday_bull_score))
                prob_bull = 0.40 * prob_bull + 0.60 * intraday_bull_score
                prob_bear = 1 - prob_bull

            bias = ("BULLISH" if prob_bull > 0.545
                    else "BEARISH" if prob_bear > 0.545
                    else "NEUTRAL")

            df_15m = df_15m_early
            df_pm  = self.dp.fetch_premarket()
            regime_dict  = self.rd.detect_macro_micro(df_d, df_15m, df_pm, vix_df)
            macro_regime = regime_dict["macro"]
            micro_regime = regime_dict["micro"]
            micro_bias   = regime_dict["micro_bias"]

            df_vp = self.dp.fetch_intraday("5m", "1d")
            vp    = VolumeProfile.calculate(df_vp)

            df_1h   = self.dp.fetch_for_timeframe("1H")
            targets = SmartTargets.compute(df_1h, df_15m, cp, atr, conf, vp)

            internals = self.rd.compute_internals(df_d, vix_df, dxy_df, qqq_df)

            mins_to_open = FeatureEngine._mins_to_ny_open()
            hi_time, lo_time = TimeEstimator.estimate(
                bias, macro_regime["phase"], macro_regime["vol_regime"])
            horizon = TimeEstimator.bias_horizon(
                bias, macro_regime["phase"], macro_regime["vol_regime"], mins_to_open)

            top_factors = self.ml.get_top_factors(last_row, bias, n=5)

            cvd_data = {"cvd_trend": "NEUTRAL", "cvd_value": 0.0}
            if df_15m is not None and len(df_15m) >= 10:
                cvd_data = CVDCalculator.calculate(df_15m, lookback=60)
            elif df_vp is not None and len(df_vp) >= 10:
                cvd_data = CVDCalculator.calculate(df_vp, lookback=60)

            hmm_data = HMMRegimeDetector.detect(df_d)

            inv_data = BiasInvalidation.compute(
                df_d, bias, atr, cp, macro_regime["phase"],
                df_intraday=df_15m)

            ai_explanation = AIExplainer.generate(
                top_factors, bias, rsi_val,
                macro_regime["phase"], macro_regime["vix"],
                cvd_data.get("cvd_trend", "NEUTRAL"))

            session_out = build_session_outlook(bias, macro_regime, rsi_val)

            try:
                import zoneinfo as _zi
                from datetime import timezone as _tzmod2
                _tz_ny2 = _zi.ZoneInfo("America/New_York")
                _now_ny2 = datetime.now(_tzmod2.utc).astimezone(_tz_ny2).replace(tzinfo=None)
            except Exception:
                _edt2 = 3 <= datetime.utcnow().month <= 11
                _now_ny2 = datetime.utcnow() - timedelta(hours=4 if _edt2 else 5)
            vol_slots = ExpectedVolume.compute(bias, macro_regime["vol_regime"], _now_ny2)

            momentum  = ("POSITIVE" if rsi_val > 55
                          else "NEGATIVE" if rsi_val < 45
                          else "NEUTRAL")
            prev_close = float(df_d["close"].iloc[-2]) if len(df_d) > 1 else cp
            delta_pct  = (cp - prev_close) / (prev_close + 1e-9) * 100

            return Prediction(
                bias=bias,
                prob_bull=prob_bull, prob_bear=prob_bear, confidence=conf,
                current_price=cp, atr=atr,
                upper_t1_lo=targets["u1_lo"], upper_t1_hi=targets["u1_hi"],
                upper_t2_lo=targets["u2_lo"], upper_t2_hi=targets["u2_hi"],
                lower_t1_lo=targets["l1_lo"], lower_t1_hi=targets["l1_hi"],
                lower_t2_lo=targets["l2_lo"], lower_t2_hi=targets["l2_hi"],
                upper_t1_conf=targets["u1_conf"], upper_t2_conf=targets["u2_conf"],
                lower_t1_conf=targets["l1_conf"], lower_t2_conf=targets["l2_conf"],
                upper_t1_label=targets["u1_label"], upper_t2_label=targets["u2_label"],
                lower_t1_label=targets["l1_label"], lower_t2_label=targets["l2_label"],
                regime=macro_regime["label"], regime_phase=macro_regime["phase"],
                macro_regime=macro_regime["label"], macro_phase=macro_regime["phase"],
                micro_regime=micro_regime["label"], micro_phase=micro_regime["phase"],
                micro_bias=micro_bias,
                vix=macro_regime["vix"], vol_regime=macro_regime["vol_regime"],
                trend_prob=macro_regime["trend_prob"],
                vix_divergence=internals["vix_divergence"],
                dxy_divergence=internals["dxy_divergence"],
                internals_health=internals["internals_health"],
                add_proxy=internals["add_proxy"],
                session_outlook=session_out,
                high_time_est=hi_time, low_time_est=lo_time,
                bias_horizon=horizon,
                wf_accuracy=self.ml.wf_metrics.get("accuracy", 0.0),
                wf_auc=self.ml.wf_metrics.get("auc", 0.0),
                delta_pct=delta_pct, rsi=rsi_val, momentum=momentum,
                top_factors=top_factors,
                poc=vp.get("poc",0.0), vah=vp.get("vah",0.0), val=vp.get("val",0.0),
                mins_to_ny_open=mins_to_open,
                invalidation_price=inv_data["invalidation_price"],
                invalidation_condition=inv_data["invalidation_condition"],
                invalidation_flip_detail=inv_data["invalidation_flip_detail"],
                cvd_trend=cvd_data.get("cvd_trend", "NEUTRAL"),
                cvd_value=cvd_data.get("cvd_value", 0.0),
                hmm_regime=hmm_data.get("hmm_regime", "Trend"),
                hmm_confidence=hmm_data.get("hmm_confidence", 0.5),
                ai_explanation=ai_explanation,
                news_sentiment="NEUTRAL",
                news_sentiment_score=0.0,
                expected_volume_slots=vol_slots,
            )
        except Exception as e:
            logger.error(f"Predict error: {traceback.format_exc()}")
            return None


# ════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html,body,[data-testid="stAppViewContainer"]{
  background:#0A0A10!important;color:#F3F4F6!important;
  font-family:'Inter',sans-serif!important;}

[data-testid="stHeader"]{
  background:rgba(10,10,16,0.95)!important;
  border-bottom:1px solid #1F2937!important;
  height:3rem!important;min-height:3rem!important;}
[data-testid="stHeader"]>*{opacity:0.15!important;transition:opacity 0.2s ease!important;}
[data-testid="stHeader"]:hover>*{opacity:1!important;}
[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"]{opacity:1!important;}

[data-testid="stSidebar"]{background:#0D0D18!important;border-right:1px solid #1F2937!important;}
[data-testid="stSidebarContent"]{padding:1.5rem 1rem!important;}
[data-testid="stMainBlockContainer"]{max-width:1600px!important;padding:0.5rem 1.5rem 2rem!important;}
hr{border-color:#1F2937!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0A0A10;}
::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px;}

[data-testid="stMetric"]{background:#12121A!important;border:1px solid #1F2937!important;
  border-radius:10px!important;padding:12px 16px!important;}
[data-testid="stMetricLabel"]{color:#9CA3AF!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:#F3F4F6!important;font-family:'JetBrains Mono'!important;}

[data-testid="stRadio"]>div{display:flex!important;flex-direction:row!important;
  gap:6px!important;flex-wrap:wrap!important;}
[data-testid="stRadio"] label{background:#12121A!important;border:1px solid #2A2A3E!important;
  border-radius:8px!important;padding:6px 14px!important;color:#9CA3AF!important;
  font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:600!important;cursor:pointer!important;transition:all 0.15s ease!important;}
[data-testid="stRadio"] label:has(input:checked){background:rgba(0,212,170,0.12)!important;
  border-color:#00D4AA!important;color:#00D4AA!important;}
[data-testid="stRadio"] input[type="radio"]{display:none!important;}

button[kind="primary"]{background:#00D4AA!important;border:none!important;
  border-radius:8px!important;color:#000!important;font-weight:700!important;}
button[kind="secondary"]{background:#12121A!important;border:1px solid #1F2937!important;
  border-radius:8px!important;color:#9CA3AF!important;}

iframe{border:none!important;background:transparent!important;}
[data-testid="stToggle"] span{color:#9CA3AF!important;font-size:13px!important;}
[data-testid="stSelectbox"]>div>div{background:#12121A!important;
  border:1px solid #2A2A3E!important;border-radius:8px!important;color:#F3F4F6!important;}

/* Checkbox row styling */
[data-testid="stCheckbox"] label{color:#9CA3AF!important;font-size:12px!important;
  font-family:'JetBrains Mono',monospace!important;}
</style>
"""


# ════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ════════════════════════════════════════════════════════════════

def _header_html(now_ny: datetime, page: str) -> str:
    ts_str   = now_ny.strftime("%B %d, %Y  |  NQZ5 (NAS-100 FUTURES)")
    time_str = now_ny.strftime("%H:%M NY")
    return f"""
<div style="padding:16px 4px 14px 4px;border-bottom:1px solid #1F2937;
            margin-bottom:20px;display:flex;justify-content:space-between;align-items:flex-end;">
  <div>
    <div style="font-family:'Inter',sans-serif;font-weight:900;font-size:22px;
                color:#F3F4F6;letter-spacing:-0.5px;">MNQ/NQ FUTURES
      <span style="color:#00D4AA;"> | </span>ML SUITE v6
      <span style="font-size:13px;font-weight:400;color:#4B5563;margin-left:12px;">
        {page.upper()}</span></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#9CA3AF;margin-top:4px;letter-spacing:1px;">{ts_str}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                font-weight:700;color:#FFB800;">{time_str}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#4B5563;letter-spacing:2px;">LIVE DATA · AUTO REFRESH</div>
  </div>
</div>"""


def _bias_panel(pred: Prediction) -> str:
    is_bull = pred.bias == "BULLISH"
    is_bear = pred.bias == "BEARISH"
    if is_bull:
        glow="rgba(0,212,170,0.18)"; border="#00D4AA"; color="#00D4AA"
        arrow="▲"; shadow="0 0 32px rgba(0,212,170,0.25)"
    elif is_bear:
        glow="rgba(255,75,75,0.15)"; border="#FF4B4B"; color="#FF4B4B"
        arrow="▼"; shadow="0 0 32px rgba(255,75,75,0.20)"
    else:
        glow="rgba(255,184,0,0.12)"; border="#FFB800"; color="#FFB800"
        arrow="◆"; shadow="0 0 24px rgba(255,184,0,0.15)"

    conf_pct = int(pred.confidence * 100)
    prob_pct = int(max(pred.prob_bull, pred.prob_bear) * 100)
    inv_color = "#FF6B6B" if is_bull else "#00D4AA" if is_bear else "#FFB800"

    cvd_color = "#00D4AA" if pred.cvd_trend == "BULLISH" else "#FF4B4B" if pred.cvd_trend == "BEARISH" else "#9CA3AF"
    cvd_arrow = "▲" if pred.cvd_trend == "BULLISH" else "▼" if pred.cvd_trend == "BEARISH" else "◆"

    inv_condition_html = _md_to_html(pred.invalidation_condition, inv_color)

    return f"""
<div style="background:linear-gradient(135deg,{glow},rgba(10,10,16,0.95));
            border:1.5px solid {border};border-radius:16px;
            padding:24px 28px;box-shadow:{shadow};position:relative;overflow:hidden;">
  <div style="position:absolute;top:-20px;right:-20px;width:120px;height:120px;
              border-radius:50%;background:{glow};filter:blur(30px);pointer-events:none;"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:10px;">⬡ DAILY ML BIAS &amp; TREND</div>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    <div style="font-size:52px;line-height:1;color:{color};
                filter:drop-shadow(0 0 12px {color});">{arrow}</div>
    <div>
      <div style="font-family:'Inter',sans-serif;font-weight:900;font-size:36px;
                  color:{color};letter-spacing:-1px;line-height:1;">{pred.bias}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:3px;">CURRENT BIAS</div>
    </div>
    <div style="margin-left:auto;text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                  color:#4B5563;letter-spacing:2px;margin-bottom:3px;">CVD FLOW</div>
      <div style="font-size:18px;color:{cvd_color};font-weight:700;">
        {cvd_arrow} <span style="font-size:11px;">{pred.cvd_trend}</span></div>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.04);border:1px solid #2A2A3E;
              border-radius:6px;padding:5px 12px;margin-bottom:10px;display:inline-block;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 color:#FFB800;letter-spacing:1px;">⏱ {pred.bias_horizon}</span>
  </div>
  <div style="background:rgba(255,107,107,0.07);border:1px solid {inv_color}44;
              border-radius:8px;padding:8px 12px;margin-bottom:12px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                color:#9CA3AF;letter-spacing:2px;margin-bottom:4px;">⚠ INVALIDATION</div>
    <div style="font-family:'Inter',sans-serif;font-size:11px;
                color:{inv_color};line-height:1.5;">{inv_condition_html}</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9CA3AF;">
        ML MODEL CONFIDENCE</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                   font-weight:700;color:{color};">{conf_pct}%</span>
    </div>
    <div style="height:4px;background:rgba(255,255,255,0.07);border-radius:2px;">
      <div style="width:{conf_pct}%;height:4px;background:{color};
                  border-radius:2px;box-shadow:0 0 8px {color};"></div></div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9CA3AF;">
        SIGNAL PROBABILITY</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                   font-weight:700;color:{color};">{prob_pct}%</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9CA3AF;">
        TREND DIRECTION</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;
                   color:{'#00D4AA' if pred.trend_prob>0.55 else '#FFB800'};">
        {'UPWARD' if is_bull else 'DOWNWARD' if is_bear else 'SIDEWAYS'}</span>
    </div>
  </div>
</div>"""


def _md_to_html(text: str, highlight_color: str = "#F3F4F6") -> str:
    """Convert **bold** markdown to styled HTML spans. Removes raw ** artifacts."""
    return re.sub(
        r'\*\*(.+?)\*\*',
        rf'<strong style="color:{highlight_color};font-weight:700;">\1</strong>',
        text
    )


def _invalidation_panel(pred: Prediction) -> str:
    """Dedicated invalidation & bias-flip explanation panel (v6)."""
    is_bull = pred.bias == "BULLISH"
    is_bear = pred.bias == "BEARISH"
    color   = "#00D4AA" if is_bull else "#FF4B4B" if is_bear else "#FFB800"

    condition_html  = _md_to_html(pred.invalidation_condition, color)
    flip_html       = _md_to_html(pred.invalidation_flip_detail, "#FFB800")

    return f"""
<div style="background:#0D0D18;border:1px solid {color}44;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:12px;">⚠ BIAS INVALIDATION &amp; FLIP CONDITIONS</div>
  <div style="background:rgba(255,107,107,0.06);border:1px solid {color}33;
              border-radius:10px;padding:14px 16px;margin-bottom:10px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#9CA3AF;letter-spacing:2px;margin-bottom:6px;">CURRENT CONDITION</div>
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:{color};line-height:1.6;">{condition_html}</div>
  </div>
  <div style="background:#12121A;border-radius:10px;padding:14px 16px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#9CA3AF;letter-spacing:2px;margin-bottom:6px;">WHAT MAKES THE BIAS FLIP</div>
    <div style="font-family:'Inter',sans-serif;font-size:12px;
                color:#D1D5DB;line-height:1.7;">{flip_html}</div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
              color:#2A2A3E;margin-top:8px;text-align:center;">
    MONITOR ON 1H TIMEFRAME · CANDLE CLOSE CONFIRMATION REQUIRED</div>
</div>"""


def _ai_explainability_panel(pred: Prediction) -> str:
    """Full dedicated AI Reasoning section (v6)."""
    if not pred.top_factors:
        return ""

    driver_rows = ""
    for f in pred.top_factors[:5]:
        bar_w = min(100, int(f["pct"] * 9))
        driver_rows += f"""
<div style="margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;">{f['label']}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
                 color:{f['color']};">{f['sign']}{f['pct']}%</span>
  </div>
  <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;">
    <div style="width:{bar_w}%;height:3px;background:{f['color']};
                border-radius:2px;box-shadow:0 0 6px {f['color']};"></div>
  </div>
</div>"""

    explanation_html = pred.ai_explanation
    explanation_html = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#F3F4F6;">\1</strong>', explanation_html)
    explanation_html = explanation_html.replace("\n\n", "<br><br>")

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:24px 26px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:18px;">⬡ AI REASONING &amp; TOP DRIVERS</div>
  <div style="display:grid;grid-template-columns:1fr 1.3fr;gap:24px;">
    <div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#4B5563;letter-spacing:2px;margin-bottom:12px;">
        FEATURE IMPORTANCE WEIGHTS</div>
      {driver_rows}
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                  color:#2A2A3E;margin-top:4px;text-align:center;">
        BASED ON ENSEMBLE TREE IMPORTANCES</div>
    </div>
    <div style="background:#12121A;border-radius:12px;padding:18px 20px;
                border:1px solid rgba(255,255,255,0.05);">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#00D4AA;letter-spacing:2px;margin-bottom:10px;">
        AI ANALYST NOTE</div>
      <div style="font-family:'Inter',sans-serif;font-size:12px;
                  color:#D1D5DB;line-height:1.75;">
        {explanation_html}</div>
    </div>
  </div>
</div>"""


def _cvd_hmm_panel(pred: Prediction) -> str:
    """CVD + HMM Regime panel (v6)."""
    cvd_color = "#00D4AA" if pred.cvd_trend == "BULLISH" else "#FF4B4B" if pred.cvd_trend == "BEARISH" else "#9CA3AF"
    cvd_arrow = "▲" if pred.cvd_trend == "BULLISH" else "▼" if pred.cvd_trend == "BEARISH" else "◆"

    hmm_color = ("#00D4AA" if pred.hmm_regime == "Trend"
                 else "#FF4B4B" if pred.hmm_regime == "Chop"
                 else "#FFB800")
    hmm_conf_pct = int(pred.hmm_confidence * 100)

    if pred.cvd_trend == "BULLISH":
        cvd_interp_text = "Aggressive buyers are in control — net buying pressure is building in the order flow."
    elif pred.cvd_trend == "BEARISH":
        cvd_interp_text = "Aggressive sellers are dominant — net selling pressure is accumulating in the order flow."
    else:
        cvd_interp_text = "Order flow is balanced. Neither buyers nor sellers have a dominant edge at current price levels."

    if pred.hmm_regime != "Transitioning":
        hmm_interp_text = (
            f'HMM statistical analysis confirms a '
            f'<strong style="color:{hmm_color};">{pred.hmm_regime}</strong> '
            f'market structure.'
        )
    else:
        hmm_interp_text = "HMM analysis shows a transitioning structure — watch for a definitive regime break."

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◈ ORDER FLOW &amp; REGIME MATH</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
    <div style="background:#12121A;border-radius:12px;padding:14px 16px;
                border:1px solid {cvd_color}33;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                  color:#4B5563;letter-spacing:3px;margin-bottom:6px;">CVD PROXY</div>
      <div style="font-size:24px;color:{cvd_color};margin-bottom:4px;">{cvd_arrow}</div>
      <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:18px;
                  color:{cvd_color};">{pred.cvd_trend}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:4px;">
        Delta: {pred.cvd_value:+,.0f}</div>
    </div>
    <div style="background:#12121A;border-radius:12px;padding:14px 16px;
                border:1px solid {hmm_color}33;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                  color:#4B5563;letter-spacing:3px;margin-bottom:6px;">HMM REGIME</div>
      <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:18px;
                  color:{hmm_color};">{pred.hmm_regime.upper()}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:6px;">Confidence: {hmm_conf_pct}%</div>
      <div style="height:3px;background:rgba(255,255,255,0.07);border-radius:2px;margin-top:5px;">
        <div style="width:{hmm_conf_pct}%;height:3px;background:{hmm_color};border-radius:2px;"></div>
      </div>
    </div>
  </div>
  <div style="background:#12121A;border-radius:8px;padding:10px 14px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                color:#4B5563;letter-spacing:2px;margin-bottom:4px;">INTERPRETATION</div>
    <div style="font-family:'Inter',sans-serif;font-size:11px;color:#9CA3AF;line-height:1.6;">
      {cvd_interp_text}  {hmm_interp_text}</div>
  </div>
</div>"""


def _vp_panel(pred: Prediction) -> str:
    """Volume Profile panel: POC, VAH, VAL with distance-to-price info."""

    def _dist(level: float) -> str:
        if pred.current_price <= 0 or level <= 0:
            return ""
        diff = level - pred.current_price
        pct  = diff / pred.current_price * 100
        sign = "+" if diff >= 0 else ""
        col  = "#00D4AA" if diff >= 0 else "#FF6B6B"
        return (
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:{col};margin-left:6px;">{sign}{pct:.2f}%</span>'
        )

    def _row(label: str, value: float, color: str, border_color: str) -> str:
        if value <= 0:
            return ""
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:9px 14px;border-radius:8px;'
            f'background:rgba(255,255,255,0.03);'
            f'border:1px solid {border_color};margin-bottom:4px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:#9CA3AF;letter-spacing:1px;">{label}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
            f'font-weight:700;color:{color};">'
            f'{value:,.0f}{_dist(value)}'
            f'</span></div>'
        )

    no_data = pred.poc <= 0 and pred.vah <= 0 and pred.val <= 0
    if no_data:
        body = (
            '<div style="text-align:center;padding:18px 0;'
            'font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#4B5563;">'
            'Volume Profile data not available</div>'
        )
    else:
        body = (
            _row("VAH — Value Area High", pred.vah, "#9B59B6", "rgba(155,89,182,0.25)")
            + _row("POC — Point of Control", pred.poc, "#FFB800", "rgba(255,184,0,0.25)")
            + _row("VAL — Value Area Low",  pred.val, "#3498DB", "rgba(52,152,219,0.25)")
        )

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◈ VOLUME PROFILE</div>
  {body}
  <div style="margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:8px;
              color:#374151;letter-spacing:1px;">
    POC = highest-volume price · VAH/VAL = 70 % value area bounds
  </div>
</div>"""


def _targets_panel(pred: Prediction) -> str:
    cp = pred.current_price
    d  = pred.delta_pct
    sign = "+" if d >= 0 else ""
    dc   = "#00D4AA" if d >= 0 else "#FF4B4B"

    def conf_badge(c: float, color: str) -> str:
        pct = int(c * 100)
        return (f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                f'font-weight:700;color:{color};background:rgba(255,255,255,0.05);'
                f'border:1px solid {color}44;border-radius:4px;padding:2px 6px;">'
                f'{pct}%</span>')

    def row(label, lo, hi, color, conf=None, sublabel=""):
        conf_html = conf_badge(conf, color) if conf is not None else ""
        sub_html  = (f'<span style="font-size:8px;color:#4B5563;margin-left:4px;">{sublabel}</span>'
                     if sublabel else "")
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:9px 14px;border-radius:8px;background:rgba(255,255,255,0.03);'
            f'border:1px solid {color}22;margin-bottom:4px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:#9CA3AF;letter-spacing:1px;">{label}{sub_html}</span>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'{conf_html}'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
            f'font-weight:700;color:{color};">{lo:,.0f} – {hi:,.0f}</span>'
            f'</div></div>'
        )

    vp_html = ""
    if pred.poc > 0:
        vp_html = f"""
<div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
               background:rgba(255,184,0,0.1);border:1px solid rgba(255,184,0,0.3);
               border-radius:4px;padding:3px 8px;color:#FFB800;">POC {pred.poc:,.0f}</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
               background:rgba(155,89,182,0.1);border:1px solid rgba(155,89,182,0.3);
               border-radius:4px;padding:3px 8px;color:#9B59B6;">VAH {pred.vah:,.0f}</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
               background:rgba(52,152,219,0.1);border:1px solid rgba(52,152,219,0.3);
               border-radius:4px;padding:3px 8px;color:#3498DB;">VAL {pred.val:,.0f}</span>
</div>"""

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◈ SMART TARGETS</div>
  {vp_html}
  {row("UPPER TARGET 2", pred.upper_t2_lo, pred.upper_t2_hi, "#00FFCC", pred.upper_t2_conf, pred.upper_t2_label)}
  {row("UPPER TARGET 1", pred.upper_t1_lo, pred.upper_t1_hi, "#00D4AA", pred.upper_t1_conf, pred.upper_t1_label)}
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:9px 14px;border-radius:8px;background:rgba(255,184,0,0.07);
              border:1.5px solid rgba(255,184,0,0.3);margin-bottom:4px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                 color:#F3F4F6;letter-spacing:1px;">CURRENT PRICE (NQ)</span>
    <span style="font-family:'JetBrains Mono',monospace;">
      <span style="font-size:16px;font-weight:800;color:#FFB800;">{cp:,.2f}</span>
      <span style="font-size:11px;font-weight:600;color:{dc};margin-left:6px;">({sign}{d:.2f}%)</span>
    </span>
  </div>
  {row("LOWER TARGET 1", pred.lower_t1_lo, pred.lower_t1_hi, "#FF6B6B", pred.lower_t1_conf, pred.lower_t1_label)}
  {row("LOWER TARGET 2", pred.lower_t2_lo, pred.lower_t2_hi, "#FF4B4B", pred.lower_t2_conf, pred.lower_t2_label)}
</div>"""


def _validation_panel(pred: Prediction) -> str:
    vix   = pred.vix
    rsi   = pred.rsi
    vol_c = "#FF4B4B" if pred.vol_regime in ("High","Extreme") else "#00D4AA"
    mom_c = "#00D4AA" if pred.momentum=="POSITIVE" else "#FF4B4B" if pred.momentum=="NEGATIVE" else "#FFB800"
    acc_pct = int(pred.wf_accuracy * 100)
    auc_pct = int(pred.wf_auc * 100)
    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 22px;height:100%;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;">
    ◉ MODEL VALIDATION</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">VOLATILITY</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:{vol_c};">{pred.vol_regime.upper()}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">ATR: {pred.atr:.0f} pts</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">MOMENTUM</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:{mom_c};">{pred.momentum}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">RSI: {rsi:.0f}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">VIX LEVEL</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:{'#FF4B4B' if vix>30 else '#FFB800' if vix>20 else '#00D4AA'};">
        {vix:.1f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">{pred.regime}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">WF ACCURACY</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:#00D4AA;">{acc_pct}%</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">AUC: {auc_pct}%</div>
    </div>
  </div>
  <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;">
        LAST BIAS CALL</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                   color:#00D4AA;font-weight:700;">{pred.bias} — ACTIVE</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;">
        SIGNAL STATUS</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#00D4AA;">
        ⬤ LIVE</span>
    </div>
  </div>
</div>"""


def _regime_panel(pred: Prediction) -> str:
    phase_colors = {
        "Trend":"#00D4AA","Accumulation":"#00AAFF","Distribution":"#FF9500",
        "Chop":"#FFB800","Consolidation":"#6B7280","Manipulation":"#FF4B4B",
    }
    macro_clr = phase_colors.get(pred.macro_phase, "#9CA3AF")
    micro_clr = phase_colors.get(pred.micro_phase, "#9CA3AF")
    micro_bias_clr = "#00D4AA" if pred.micro_bias=="BULLISH" else "#FF4B4B" if pred.micro_bias=="BEARISH" else "#FFB800"
    macro_bull = pred.macro_phase in ("Trend","Accumulation")
    micro_bull = pred.micro_bias == "BULLISH"
    aligned    = (macro_bull == micro_bull)
    align_color = "#00D4AA" if aligned else "#FF6B6B"
    align_label = "ALIGNED ✓" if aligned else "CONFLICT ✗"

    phases = ["Accumulation","Manipulation","Trend","Distribution","Chop","Consolidation"]
    dots = ""
    for p in phases:
        active = (p == pred.macro_phase)
        pc = phase_colors.get(p, "#4B5563")
        bg = pc if active else "#1F2937"
        dots += (f'<div style="background:{bg};border-radius:6px;padding:5px 10px;'
                 f'font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                 f'color:{"#000" if active else "#9CA3AF"};font-weight:{"700" if active else "400"};'
                 f'white-space:nowrap;">{p}</div>')

    hmm_color  = "#00D4AA" if pred.hmm_regime=="Trend" else "#FF4B4B" if pred.hmm_regime=="Chop" else "#FFB800"
    hmm_pct    = int(pred.hmm_confidence * 100)

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">▣ MARKET REGIME &amp; PHASE</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;">
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;
                border:1px solid rgba(255,255,255,0.05);">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                  color:#4B5563;letter-spacing:3px;margin-bottom:6px;">MACRO / SWING</div>
      <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:16px;
                  color:{macro_clr};line-height:1.1;">{pred.macro_phase.upper()}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:4px;">{pred.macro_regime}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;
                border:1px solid {micro_bias_clr}33;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                  color:#4B5563;letter-spacing:3px;margin-bottom:6px;">MICRO / INTRADAY</div>
      <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:16px;
                  color:{micro_clr};line-height:1.1;">{pred.micro_phase.upper()}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:{micro_bias_clr};margin-top:4px;font-weight:700;">{pred.micro_bias}</div>
    </div>
  </div>
  <div style="text-align:center;margin-bottom:12px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                 background:{align_color}22;border:1px solid {align_color}44;
                 border-radius:20px;padding:4px 14px;color:{align_color};
                 font-weight:700;letter-spacing:1px;">
      MACRO ↔ MICRO: {align_label}</span>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;">{dots}</div>
  <div style="background:#12121A;border-radius:8px;padding:10px 14px;margin-top:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                   color:#9CA3AF;letter-spacing:2px;">HMM REGIME STATE</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                   font-weight:700;color:{hmm_color};">{pred.hmm_regime} ({hmm_pct}%)</span>
    </div>
    <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;">
      <div style="width:{hmm_pct}%;height:3px;background:{hmm_color};border-radius:2px;"></div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
    <div style="background:#12121A;border-radius:8px;padding:10px 12px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;">TREND PROB</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                  font-weight:700;color:#00D4AA;">{int(pred.trend_prob*100)}%</div>
    </div>
    <div style="background:#12121A;border-radius:8px;padding:10px 12px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;">VOL REGIME</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                  font-weight:700;color:{'#FF4B4B' if pred.vol_regime in ('High','Extreme') else '#00D4AA'};">
        {pred.vol_regime.upper()}</div>
    </div>
  </div>
</div>"""


def _internals_panel(pred: Prediction) -> str:
    hc = pred.internals_health
    hc_color = "#00D4AA" if hc=="HEALTHY" else "#FF4B4B" if hc=="STRESSED" else "#FFB800"
    hc_icon  = "✓" if hc=="HEALTHY" else "✗" if hc=="STRESSED" else "~"
    vix_d_clr = "#FF4B4B" if "BEARISH" in pred.vix_divergence else "#00D4AA" if "BULLISH" in pred.vix_divergence else "#9CA3AF"
    dxy_d_clr = "#FF4B4B" if "HEADWIND" in pred.dxy_divergence else "#00D4AA" if "TAILWIND" in pred.dxy_divergence else "#9CA3AF"
    add_clr   = "#00D4AA" if pred.add_proxy > 0 else "#FF4B4B"
    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;">
      ⬡ MARKET INTERNALS</div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
                 background:{hc_color}22;border:1px solid {hc_color}44;
                 border-radius:6px;padding:3px 10px;color:{hc_color};">
      {hc} {hc_icon}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">VIX DIVERGENCE</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                  font-weight:700;color:{vix_d_clr};">{pred.vix_divergence}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">DXY DIVERGENCE</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                  font-weight:700;color:{dxy_d_clr};">{pred.dxy_divergence}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">ADD PROXY</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                  font-weight:700;color:{add_clr};">
        {'+' if pred.add_proxy>=0 else ''}{pred.add_proxy:.1f}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">NY OPEN IN</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                  font-weight:700;color:#FFB800;">{pred.mins_to_ny_open}m</div>
    </div>
  </div>
</div>"""


def _time_panel(pred: Prediction) -> str:
    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;">
    ◷ TIME PROJECTIONS</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div style="background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.20);
                border-radius:10px;padding:14px 16px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#00D4AA;letter-spacing:2px;margin-bottom:6px;">HIGH EXPECTED</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                  font-weight:700;color:#00D4AA;">{pred.high_time_est}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:4px;">Statistical estimate</div>
    </div>
    <div style="background:rgba(255,75,75,0.06);border:1px solid rgba(255,75,75,0.20);
                border-radius:10px;padding:14px 16px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#FF6B6B;letter-spacing:2px;margin-bottom:6px;">LOW EXPECTED</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                  font-weight:700;color:#FF6B6B;">{pred.low_time_est}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:4px;">Statistical estimate</div>
    </div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
              color:#4B5563;margin-top:12px;text-align:center;">
    15-min precision · Based on historical intraday volatility distribution</div>
</div>"""


def _expected_volume_panel(pred: Prediction) -> str:
    slots = pred.expected_volume_slots
    if not slots:
        return ""
    rows = ""
    for s in slots:
        tier  = s["tier"]
        rv    = s["rel_vol"]
        is_now = s.get("now", False)
        bar_w  = min(100, int(rv / 2.8 * 100))
        if tier == "HIGH":
            bar_color  = "#00D4AA"
            tier_badge = f'<span style="font-size:8px;background:rgba(0,212,170,0.15);color:#00D4AA;border-radius:3px;padding:1px 5px;">HIGH</span>'
        elif tier == "LOW":
            bar_color  = "#4B5563"
            tier_badge = f'<span style="font-size:8px;background:rgba(75,85,99,0.2);color:#6B7280;border-radius:3px;padding:1px 5px;">LOW</span>'
        else:
            bar_color  = "#FFB800"
            tier_badge = f'<span style="font-size:8px;background:rgba(255,184,0,0.12);color:#FFB800;border-radius:3px;padding:1px 5px;">NORM</span>'

        now_style = "border-left:2px solid #00D4AA;" if is_now else ""
        now_dot   = '<span style="color:#00D4AA;font-size:8px;"> ← NOW</span>' if is_now else ""
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;padding:5px 0;{now_style}{'padding-left:8px;' if is_now else ''}">
  <div style="min-width:80px;font-family:'JetBrains Mono',monospace;font-size:9px;
              color:{'#F3F4F6' if is_now else '#9CA3AF'};">{s['slot']}{now_dot}</div>
  <div style="flex:1;background:#1F2937;border-radius:3px;height:6px;overflow:hidden;">
    <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:3px;
                transition:width 0.3s;"></div>
  </div>
  <div style="min-width:32px;font-family:'JetBrains Mono',monospace;font-size:9px;
              color:{bar_color};text-align:right;">{rv:.1f}x</div>
  <div style="min-width:42px;">{tier_badge}</div>
</div>"""

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:18px 20px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;">
    📊 EXPECTED VOLUME — NY SESSION</div>
  <div style="font-family:'Inter',sans-serif;font-size:10px;color:#4B5563;margin-bottom:12px;">
    Relative to avg 30-min volume · ML-adjusted · NY time slots</div>
  {rows}
  <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#2A2A3E;
              margin-top:8px;">1.0x = avg volume · HIGH ≥1.5x · LOW &lt;0.75x</div>
</div>"""


def _session_panel(pred: Prediction) -> str:
    icons   = {"Asia":"🌏","London":"🇬🇧","NY Open":"🗽","NY PM":"🌆"}
    h       = datetime.utcnow().hour
    current = ("Asia" if 0<=h<7 else "London" if 7<=h<12 else "NY Open" if 12<=h<17 else "NY PM")
    rows    = ""
    for sess, outlook in pred.session_outlook.items():
        active = (sess == current)
        border = "rgba(0,212,170,0.35)" if active else "#1F2937"
        bg     = "rgba(0,212,170,0.06)" if active else "#12121A"
        icon   = icons.get(sess, "◈")
        rows += f"""
<div style="background:{bg};border:1px solid {border};border-radius:10px;
            padding:14px 16px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span style="font-size:15px;">{icon}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                 color:{'#00D4AA' if active else '#9CA3AF'};
                 font-weight:{'700' if active else '500'};letter-spacing:2px;">
      {sess.upper()}{"  ← NOW" if active else ""}</span>
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:12px;color:#D1D5DB;
              line-height:1.65;padding-left:23px;">{outlook}</div>
</div>"""
    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;">
    ◎ SESSION OUTLOOK — AI ANALYSIS</div>
  {rows}
</div>"""


def _news_panel(events: List[Dict], sentiment: str, sentiment_score: float) -> str:
    if not events:
        return ""
    import html as _html
    import re as _re

    def _clean(text: str) -> str:
        """Strip any leftover HTML tags from RSS titles."""
        text = _re.sub(r'<[^>]+>', '', text)
        text = _html.unescape(text)
        return text.strip()

    sent_color = "#00D4AA" if sentiment=="BULLISH" else "#FF4B4B" if sentiment=="BEARISH" else "#FFB800"
    sent_pct   = int(abs(sentiment_score) * 100)
    rows = ""
    for e in events:
        title_clean = _clean(e.get("event", ""))
        if not title_clean:
            continue
        link_html = (f'<a href="{_html.escape(e["link"])}" target="_blank" '
                     f'style="color:#9CA3AF;font-size:9px;text-decoration:none;">[→]</a> '
                     if e.get("link") else "")
        src_raw   = _clean(e.get("source", ""))
        src_color = "#FF6B6B" if "Forex" in src_raw else "#00AAFF"
        rows += f"""
<div style="display:flex;align-items:flex-start;gap:10px;
            padding:9px 14px;border-bottom:1px solid #0E0E1A;">
  <div style="min-width:44px;font-family:'JetBrains Mono',monospace;
              font-size:10px;color:#FF6B6B;padding-top:1px;flex-shrink:0;">{_clean(e.get('time','—'))}</div>
  <div style="flex:1;min-width:0;">
    <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                color:#F3F4F6;margin-bottom:2px;word-break:break-word;">
      {link_html}{_html.escape(title_clean)}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:{src_color};">
      {_html.escape(src_raw)}</div>
  </div>
  <div style="width:6px;height:6px;border-radius:50%;background:#FF4B4B;
              margin-top:5px;flex-shrink:0;"></div>
</div>"""

    return f"""
<div style="background:#0D0D18;border:1px solid #1E0A0A;border-radius:16px;overflow:hidden;">
  <div style="padding:14px 18px 10px 18px;border-bottom:1px solid #0E0E1A;
              display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                   color:#FF6B6B;letter-spacing:3px;">⚡ LIVE NEWS</span>
      <span style="background:#FF4B4B;color:#000;font-family:'JetBrains Mono',monospace;
                   font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;">LIVE</span>
    </div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
                 background:{sent_color}22;border:1px solid {sent_color}44;
                 border-radius:6px;padding:3px 10px;color:{sent_color};">
      {sentiment} ({sent_pct}%)</span>
  </div>
  <div style="max-height:420px;overflow-y:auto;overflow-x:hidden;">
    {rows}
  </div>
</div>"""


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
            color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
            margin-bottom:20px;">⬡ NQ PREDICTOR v6</div>
""", unsafe_allow_html=True)

    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    st.sidebar.markdown("<hr style='border-color:#1F2937;margin:18px 0;'>", unsafe_allow_html=True)

    show_proj    = st.sidebar.toggle("Show Future Projection", value=True,
                                     help="Display the ML model's directional projection cone on the chart.")
    auto_refresh = st.sidebar.toggle("Auto Refresh (5 min)", value=False,
                                     help="Automatically reload the page every 5 minutes to fetch fresh market data.")

    st.sidebar.markdown("<hr style='border-color:#1F2937;margin:18px 0;'>", unsafe_allow_html=True)

    retrain = st.sidebar.button("🔄 Retrain Model", use_container_width=True,
                                 help="Force a full model retrain from scratch. Use after major market regime changes or if predictions seem stale.")
    if retrain:
        try:
            MODEL_CACHE.unlink()
        except Exception:
            pass
        st.cache_data.clear()
        st.cache_resource.clear()   # clears _get_engine() so engine re-trains
        st.rerun()

    st.sidebar.markdown("<hr style='border-color:#1F2937;margin:18px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
            color:#4B5563;letter-spacing:2px;line-height:1.8;">
MODELS<br>
XGBoost · LightGBM<br>
RandomForest · ExtraTrees<br>
LogisticRegression (meta)<br><br>
VALIDATION<br>
Walk-Forward · TimeSeriesSplit<br>
4-fold · No lookahead<br><br>
DATA<br>
NQ Futures · VIX · DXY<br>
US10Y · QQQ · ES Futures<br><br>
NEW IN v6<br>
AI Explainability (NL)<br>
Bias Invalidation Levels<br>
CVD Order Flow Proxy<br>
News Sentiment NLP<br>
HMM Regime (Hurst/ADX)<br>
Chart Toggle Controls<br>
Global UI Tooltips<br>
Hardened RSS Fetcher<br>
AI Chart Analysis (Vision)
</div>
""", unsafe_allow_html=True)
    st.sidebar.markdown("""
<div style="padding:0 4px;margin-top:16px;font-family:'JetBrains Mono',monospace;
            font-size:8px;color:#2A2A3E;text-align:center;line-height:1.6;">
PROBABILISTIC MODEL ONLY<br>NOT FINANCIAL ADVICE<br>FOR RESEARCH PURPOSES ONLY
</div>
""", unsafe_allow_html=True)

    return page, show_proj, auto_refresh


# ════════════════════════════════════════════════════════════════
#  PAGE RENDERERS
# ════════════════════════════════════════════════════════════════

def page_dashboard(pred: Prediction, show_proj: bool,
                   df_chart: Optional[pd.DataFrame], selected_tf: str):
    left, right = st.columns([1, 1.85], gap="medium")
    with left:
        st.markdown(_bias_panel(pred), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_invalidation_panel(pred), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_targets_panel(pred), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_expected_volume_panel(pred), unsafe_allow_html=True)
    with right:
        st.markdown(_ai_explainability_panel(pred), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.markdown(_regime_panel(pred), unsafe_allow_html=True)
        with c2:
            st.markdown(_cvd_hmm_panel(pred), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(_internals_panel(pred), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(_time_panel(pred), unsafe_allow_html=True)


def page_charts(pred: Prediction, show_proj: bool,
                df_chart: Optional[pd.DataFrame], selected_tf: str):
    st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;
            letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">
  📈 NQ (NAS-100) CHART — SELECT TIMEFRAME</div>""", unsafe_allow_html=True)

    tf_options  = list(TF_CONFIG.keys())
    default_idx = tf_options.index(selected_tf) if selected_tf in tf_options else tf_options.index("1H")
    new_tf      = st.radio("TF", tf_options, index=default_idx, horizontal=True,
                           label_visibility="collapsed", key="chart_tf_radio")
    if new_tf != selected_tf:
        st.session_state["selected_tf"] = new_tf
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        show_vp_toggle = st.checkbox(
            "Volume Profile (VP)",
            value=True,
            help="Show/hide the Volume Profile lines on the chart: POC (Point of Control), VAH (Value Area High), VAL (Value Area Low). These identify where most volume was traded.")
    with tc2:
        show_targets_toggle = st.checkbox(
            "Smart Targets (T1/T2)",
            value=True,
            help="Show/hide T1 and T2 price target zones. T1 is the nearest pivot/structural target; T2 is the VWAP-band or POC-level extended target.")
    with tc3:
        show_proj_toggle = st.checkbox(
            "Projection Cone",
            value=show_proj,
            help="Show/hide the ML directional projection cone. The center line is the model's expected drift; the outer bounds show the uncertainty cone based on ATR.")
    with tc4:
        st.metric(
            "Current Price",
            f"{pred.current_price:,.2f}",
            delta=f"{pred.delta_pct:+.2f}%",
            help="Last available NQ futures price from yfinance, with % change vs prior close."
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if df_chart is not None and not df_chart.empty:
        chart_html = build_lwc_chart(
            df_chart, pred, selected_tf,
            show_projection=show_proj_toggle,
            show_vp=show_vp_toggle,
            show_targets=show_targets_toggle,
        )
        st.components.v1.html(chart_html, height=460, scrolling=False)
    else:
        st.warning(f"Chart data unavailable for {selected_tf}.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(_vp_panel(pred), unsafe_allow_html=True)
    with c2:
        st.markdown(_targets_panel(pred), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(_cvd_hmm_panel(pred), unsafe_allow_html=True)


def page_validation(pred: Prediction):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("WF Accuracy", f"{int(pred.wf_accuracy*100)}%",
                  help="Walk-Forward cross-validated accuracy. Measures how often the model correctly predicted next-bar direction on out-of-sample test folds.")
    with col2:
        st.metric("ROC-AUC", f"{int(pred.wf_auc*100)}%",
                  help="Area Under the ROC Curve. Values above 55% indicate the model has meaningful predictive power beyond random chance.")
    with col3:
        st.metric("VIX", f"{pred.vix:.1f}",
                  help="CBOE Volatility Index. VIX > 30 = fear/panic. VIX 20–30 = neutral. VIX < 20 = complacency/risk-on. High VIX amplifies NQ moves in both directions.")
    with col4:
        st.metric("ATR (14)", f"{pred.atr:.0f} pts",
                  help="Average True Range over 14 bars. This is the expected intraday range. T1 and T2 targets are calibrated as multiples of ATR.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(_validation_panel(pred), unsafe_allow_html=True)
    with c2:
        st.markdown(_regime_panel(pred), unsafe_allow_html=True)
    with c3:
        st.markdown(_time_panel(pred), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        st.markdown(_internals_panel(pred), unsafe_allow_html=True)
    with c5:
        st.markdown(_cvd_hmm_panel(pred), unsafe_allow_html=True)
    with c6:
        st.markdown(_invalidation_panel(pred), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(_ai_explainability_panel(pred), unsafe_allow_html=True)


def page_news(pred: Prediction, news: List[Dict]):
    sentiment, score = NewsSentimentAnalyzer.score(news)
    s1, s2 = st.columns([1.1, 0.9], gap="medium")
    with s1:
        st.markdown(_session_panel(pred), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_expected_volume_panel(pred), unsafe_allow_html=True)
    with s2:
        st.markdown(_news_panel(news, sentiment, score), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(_invalidation_panel(pred), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  CHART ANALYSIS PAGE — AI Vision (v6 new)
# ════════════════════════════════════════════════════════════════
def _generate_ml_only_analysis(pred: Prediction) -> str:
    """Fallback: generate a text-only analysis from ML model data when vision API is unavailable."""
    bias_word = pred.bias
    rsi_note  = ("overbought territory" if pred.rsi > 70
                 else "oversold territory" if pred.rsi < 30
                 else f"neutral at {pred.rsi:.0f}")

    return f"""## 📊 ML Model Analysis (Vision API Unavailable)

*Note: The chart image could not be processed via the vision API. Below is the ML model's current analysis.*

## 📊 Market Structure
The ML ensemble is registering a **{bias_word}** directional bias with {pred.confidence:.1%} model confidence.
The macro regime is classified as **{pred.macro_regime}** in a **{pred.macro_phase}** phase, with the
HMM statistical model confirming a **{pred.hmm_regime}** market structure state.

## 🎯 Key Price Levels
- **POC (Point of Control):** {pred.poc:,.2f}
- **Value Area High (VAH):** {pred.vah:,.2f}
- **Value Area Low (VAL):** {pred.val:,.2f}
- **Current Price:** {pred.current_price:,.2f}
- **ATR (14):** {pred.atr:.0f} pts

## ⚡ ML Targets
- Upper T1: {pred.upper_t1_lo:,.0f} – {pred.upper_t1_hi:,.0f} ({pred.upper_t1_label}) — Touch prob: {int(pred.upper_t1_conf*100)}%
- Upper T2: {pred.upper_t2_lo:,.0f} – {pred.upper_t2_hi:,.0f} ({pred.upper_t2_label}) — Touch prob: {int(pred.upper_t2_conf*100)}%
- Lower T1: {pred.lower_t1_lo:,.0f} – {pred.lower_t1_hi:,.0f} ({pred.lower_t1_label}) — Touch prob: {int(pred.lower_t1_conf*100)}%
- Lower T2: {pred.lower_t2_lo:,.0f} – {pred.lower_t2_hi:,.0f} ({pred.lower_t2_label}) — Touch prob: {int(pred.lower_t2_conf*100)}%

## ⚠️ Invalidation
{pred.invalidation_condition}

{pred.invalidation_flip_detail}

*To enable full AI chart analysis, please ensure a valid Gemini API key is configured in Streamlit secrets.*"""


def _render_analysis_html(analysis_text: str) -> str:
    """Convert the analysis markdown text into styled HTML for rendering."""
    import html as html_lib

    lines = analysis_text.split("\n")
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = html_lib.escape(stripped[3:])
            html_lines.append(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'font-weight:700;color:#00D4AA;letter-spacing:2px;text-transform:uppercase;'
                f'margin:18px 0 8px 0;padding-bottom:6px;border-bottom:1px solid #1F2937;">'
                f'{heading}</div>'
            )
        elif stripped.startswith("- "):
            content = stripped[2:]
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#F3F4F6;">\1</strong>', html_lib.escape(content))
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#F3F4F6;">\1</strong>', content)
            html_lines.append(
                f'<div style="display:flex;gap:8px;margin-bottom:5px;">'
                f'<span style="color:#00D4AA;flex-shrink:0;">▸</span>'
                f'<span style="font-family:\'Inter\',sans-serif;font-size:12px;'
                f'color:#D1D5DB;line-height:1.6;">{content}</span></div>'
            )
        elif stripped.startswith("*Note:") or stripped.startswith("*To enable"):
            note = re.sub(r'\*(.+?)\*', r'\1', stripped)
            html_lines.append(
                f'<div style="font-family:\'Inter\',sans-serif;font-size:11px;'
                f'color:#6B7280;font-style:italic;margin-bottom:4px;">{html_lib.escape(note)}</div>'
            )
        elif stripped == "":
            html_lines.append('<div style="height:6px;"></div>')
        else:
            escaped = html_lib.escape(stripped)
            escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#F3F4F6;">\1</strong>', escaped)
            html_lines.append(
                f'<div style="font-family:\'Inter\',sans-serif;font-size:12px;'
                f'color:#D1D5DB;line-height:1.7;margin-bottom:4px;">{escaped}</div>'
            )

    return "\n".join(html_lines)


def page_chart_analysis(pred: Prediction):
    """Page 5: AI Chart Analysis — upload a screenshot, get AI + ML analysis."""

    st.markdown("""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 26px;margin-bottom:16px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:10px;">🤖 AI CHART ANALYSIS — VISION + ML CROSS-VALIDATION</div>
  <div style="font-family:'Inter',sans-serif;font-size:13px;color:#D1D5DB;line-height:1.7;">
    Upload any <strong style="color:#F3F4F6;">TradingView screenshot</strong> — chart, setup, or
    order flow view. The AI analyst will identify market structure, key levels, ICT/SMC concepts,
    and cross-validate the visual setup against the live ML model signals.
  </div>
  <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                 border-radius:4px;padding:3px 10px;color:#00D4AA;">✓ Market Structure</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                 border-radius:4px;padding:3px 10px;color:#00D4AA;">✓ ICT / SMC Concepts</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                 border-radius:4px;padding:3px 10px;color:#00D4AA;">✓ ML Cross-Validation</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                 border-radius:4px;padding:3px 10px;color:#00D4AA;">✓ Precise Entry / SL / TP</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                 border-radius:4px;padding:3px 10px;color:#00D4AA;">✓ Risk Assessment</span>
  </div>
</div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    bias_color = "#00D4AA" if pred.bias=="BULLISH" else "#FF4B4B" if pred.bias=="BEARISH" else "#FFB800"
    with m1:
        st.metric("ML Bias", pred.bias,
                  help="Current ML ensemble directional bias used for cross-validation.")
    with m2:
        st.metric("Confidence", f"{int(pred.confidence*100)}%",
                  help="ML model confidence score (lower std across models = higher confidence).")
    with m3:
        st.metric("RSI(14)", f"{pred.rsi:.0f}",
                  help="14-period RSI on the daily timeframe.")
    with m4:
        st.metric("HMM Regime", pred.hmm_regime,
                  help="Hidden Markov Model statistical regime: Trend, Chop, or Transitioning.")
    with m5:
        st.metric("CVD Flow", pred.cvd_trend,
                  help="Cumulative Volume Delta proxy — net order flow direction.")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.4], gap="large")

    with left_col:
        st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
  📎 UPLOAD CHART SCREENSHOT</div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your TradingView chart here",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            help="Supported: PNG, JPG, JPEG, WEBP. Best results with full TradingView screenshots showing at least 50–100 candles.",
        )

        _secret_key = ""
        try:
            _s = st.secrets
            if hasattr(_s, "GEMINI_API_KEY"):
                _secret_key = str(_s.GEMINI_API_KEY).strip()
            elif "GEMINI_API_KEY" in _s:
                _secret_key = str(_s["GEMINI_API_KEY"]).strip()
            elif hasattr(_s, "get") and callable(_s.get):
                _val = _s.get("GEMINI_API_KEY", "")
                if _val:
                    _secret_key = str(_val).strip()
        except Exception:
            pass

        if _secret_key:
            api_key = _secret_key
            st.markdown("""
<div style="background:rgba(0,212,170,0.07);border:1px solid rgba(0,212,170,0.25);
            border-radius:8px;padding:10px 14px;margin-bottom:8px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
               color:#00D4AA;letter-spacing:2px;">🤖 AI VISION POWERED · GEMINI 2.5 FLASH ✓</span>
</div>""", unsafe_allow_html=True)
        else:
            api_key = ""
            st.markdown("""
<div style="background:rgba(255,75,75,0.07);border:1px solid rgba(255,75,75,0.25);
            border-radius:8px;padding:10px 14px;margin-bottom:8px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
               color:#FF4B4B;letter-spacing:2px;">⚠ AI KEY NOT CONFIGURED IN SECRETS</span>
</div>""", unsafe_allow_html=True)

        analyze_btn = st.button(
            "🔍 Analyze Chart",
            use_container_width=True,
            disabled=(uploaded is None),
            help="Run the full AI vision + ML cross-validation analysis on the uploaded chart.",
        )

        st.markdown("""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:12px;
            padding:16px 18px;margin-top:14px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
              color:#9CA3AF;letter-spacing:2px;margin-bottom:10px;">💡 TIPS FOR BEST RESULTS</div>
  <div style="font-family:'Inter',sans-serif;font-size:11px;color:#6B7280;line-height:1.8;">
    ▸ Use full-screen TradingView screenshots<br>
    ▸ Include at least 50–100 candles for context<br>
    ▸ Show the price scale clearly on the right<br>
    ▸ Any timeframe works (1m – Daily)<br>
    ▸ Include indicators if visible (EMA, VWAP, etc.)<br>
    ▸ MNQ/NQ charts give the most relevant analysis
  </div>
</div>""", unsafe_allow_html=True)

    with right_col:
        if uploaded is not None:
            st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
  📸 UPLOADED CHART</div>""", unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)

            file_size_kb = len(uploaded.getvalue()) / 1024
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                f'color:#4B5563;text-align:right;margin-top:4px;">'
                f'{uploaded.name} · {file_size_kb:.0f} KB · {uploaded.type}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
<div style="background:#0D0D18;border:2px dashed #2A2A3E;border-radius:16px;
            padding:60px 20px;text-align:center;">
  <div style="font-size:48px;margin-bottom:16px;opacity:0.4;">📊</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
              color:#4B5563;letter-spacing:2px;">NO CHART UPLOADED YET</div>
  <div style="font-family:'Inter',sans-serif;font-size:12px;
              color:#2A2A3E;margin-top:8px;">Upload a TradingView screenshot to begin</div>
</div>""", unsafe_allow_html=True)

    if analyze_btn and uploaded is not None:
        import base64

        img_bytes   = uploaded.getvalue()
        img_b64     = base64.b64encode(img_bytes).decode("utf-8")
        media_type  = uploaded.type if uploaded.type in ["image/png", "image/jpeg", "image/webp"] else "image/png"

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        _progress_steps = [
            (0,  "⚙ INITIALISING AI ANALYSIS ENGINE..."),
            (12, "🖼 DECODING CHART IMAGE..."),
            (28, "🔍 IDENTIFYING MARKET STRUCTURE..."),
            (44, "📐 MAPPING KEY LEVELS & ORDER BLOCKS..."),
            (60, "🤖 CROSS-VALIDATING WITH ML SIGNALS..."),
            (78, "✍ GENERATING TRADING PLAN..."),
            (92, "🔄 FINALISING RISK ASSESSMENT..."),
        ]
        _prog_bar   = st.progress(0)
        _prog_label = st.empty()

        def _set_progress(pct: int, label: str):
            _prog_bar.progress(pct)
            _prog_label.markdown(
                f'<div style="background:rgba(0,212,170,0.06);'
                f'border:1px solid rgba(0,212,170,0.2);border-radius:10px;'
                f'padding:11px 18px;margin-bottom:4px;">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                f'color:#00D4AA;letter-spacing:2px;">{label}</span></div>',
                unsafe_allow_html=True,
            )

        for _pct, _lbl in _progress_steps:
            _set_progress(_pct, _lbl)
            time.sleep(0.25)

        analysis_text = _call_gemini_vision(img_b64, media_type, pred, api_key.strip())

        _set_progress(100, "✅ ANALYSIS COMPLETE")
        time.sleep(0.4)
        _prog_bar.empty()
        _prog_label.empty()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        analysis_html = _render_analysis_html(analysis_text)

        st.markdown(f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:26px 28px;margin-top:4px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;">
      🤖 AI ANALYSIS RESULT</div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                 border-radius:4px;padding:3px 10px;color:#00D4AA;">
      ML BIAS: {pred.bias} · RSI: {pred.rsi:.0f} · {pred.macro_phase.upper()}</span>
  </div>
  {analysis_html}
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(_targets_panel(pred), unsafe_allow_html=True)
        with c2:
            st.markdown(_invalidation_panel(pred), unsafe_allow_html=True)

    elif uploaded is not None and not analyze_btn:
        st.markdown("""
<div style="background:rgba(255,184,0,0.06);border:1px solid rgba(255,184,0,0.2);
            border-radius:12px;padding:14px 20px;margin-top:10px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
               color:#FFB800;letter-spacing:2px;">
    ↑ Click "Analyze Chart" above to run the AI analysis</span>
</div>""", unsafe_allow_html=True)


def _call_gemini_vision(image_b64: str, media_type: str,
                        pred: Prediction, api_key: str) -> str:
    """
    Call Google Gemini Vision API.

    FIX v6.1: gemini-1.5-flash was fully RETIRED by Google and now returns HTTP 404
    for every request, which silently fell through to the ML-only fallback ("Vision
    API Unavailable"). This version:
      1. Uses the current gemini-2.5-flash model with an automatic fallback chain.
      2. Authenticates via the x-goog-api-key header (not key-in-URL).
      3. Treats 404 / empty-text 200 / 5xx as "try next model".
      4. Surfaces 401/403/400/429 cleanly. Falls back to ML-only only if exhausted.
    Docs: https://ai.google.dev/gemini-api/docs
    """
    if not api_key:
        return _generate_ml_only_analysis(pred)

    full_prompt = f"""You are an elite intraday futures trader and quantitative analyst specializing in NQ/MNQ Nasdaq-100 futures.
You have deep expertise in ICT (Inner Circle Trader) and SMC (Smart Money Concepts), classical technical analysis, volume analysis, and multi-timeframe analysis.
Your PRIMARY focus is INTRADAY price action — what matters is what happens TODAY, not what happened over the last 2 weeks.

Current ML Model Context (intraday-weighted signals — cross-validate your visual analysis):
- ML Intraday Bias: {pred.bias} | Bull Prob: {pred.prob_bull:.1%} | Bear Prob: {pred.prob_bear:.1%}
- Model Confidence: {pred.confidence:.1%}
- Current Price: {pred.current_price:,.2f} | ATR(14): {pred.atr:.0f} pts
- Micro Bias (intraday EMA/RSI): {pred.micro_bias}
- Regime: {pred.macro_regime} / Phase: {pred.macro_phase}
- RSI(14): {pred.rsi:.1f} | CVD Flow: {pred.cvd_trend} | HMM State: {pred.hmm_regime}
- VIX: {pred.vix:.1f}
- Smart Targets UP: T1 {pred.upper_t1_lo:,.0f}–{pred.upper_t1_hi:,.0f} | T2 {pred.upper_t2_lo:,.0f}–{pred.upper_t2_hi:,.0f}
- Smart Targets DOWN: T1 {pred.lower_t1_lo:,.0f}–{pred.lower_t1_hi:,.0f} | T2 {pred.lower_t2_lo:,.0f}–{pred.lower_t2_hi:,.0f}
- Bias Invalidation: {pred.invalidation_condition}

Analyze the attached TradingView chart image for an INTRADAY NQ/MNQ trade. Structure your response EXACTLY as:

## 📊 Market Structure
Current intraday structure: trend direction, CHoCH/BOS levels, recent swing highs/lows on THIS chart. Is price in a higher-TF draw or range?

## 🎯 Key Price Levels
List SPECIFIC price levels visible on the chart:
- Resistance zones / sell-side liquidity
- Support zones / buy-side liquidity
- Order Blocks (OB) or Fair Value Gaps (FVG) visible
- VWAP / EMA confluences if visible
- Today's high / low / previous day high / low if visible

## ✅ Setup Evaluation — WHAT SPEAKS FOR A TRADE
- List concrete reasons supporting a trade in the bias direction ({pred.bias})
- Confluences, structure, momentum, liquidity targets
- What pattern/setup is forming (e.g. OTE, displacement + FVG fill, liquidity sweep + reversal)

## ❌ Setup Evaluation — WHAT SPEAKS AGAINST
- What are the risks and counter-arguments?
- What is missing for a high-probability setup?
- Are there signs of chop, distribution, or opposing momentum?

## 📈 ML Cross-Validation
Does the visual chart CONFIRM or CONTRADICT the ML intraday bias of {pred.bias}? How do visible levels relate to the ML targets above?

## 🎯 Intraday Trading Plan
- Directional bias from chart
- Precise entry zone (price level + trigger)
- Stop loss level with reasoning (invalidation)
- Target 1 (conservative, e.g. next liquidity level)
- Target 2 (extended, e.g. daily range target)
- Estimated Risk/Reward ratio
- Best entry time window (NY session context)

## ⚠️ Risk Factors & Invalidation
What price action would invalidate this setup? What to avoid?
If no clean setup is visible, say so honestly and describe what you WOULD wait for.

Be specific with price levels. Respond in English."""

    # Current models with automatic fallback. 1.5-flash removed (retired -> 404).
    model_chain = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-2.0-flash",
    ]
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers  = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": image_b64,
                        }
                    },
                    {
                        "text": full_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 4096,
            "topP": 0.9,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    def _parse_text(data: dict) -> Tuple[str, str]:
        """Extract all text parts from a Gemini response; also surface finishReason."""
        try:
            cands = data.get("candidates") or []
            if not cands:
                return "", "NO_CANDIDATES"
            cand = cands[0]
            finish = cand.get("finishReason", "")
            parts = (cand.get("content") or {}).get("parts") or []
            text_chunks = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
            return ("".join(text_chunks).strip(), finish)
        except Exception as e:
            return "", f"PARSE_ERROR:{e}"

    last_diag = ""
    for model_id in model_chain:
        url = base_url.format(model=model_id)
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
        except requests.exceptions.Timeout:
            last_diag = f"{model_id}: timeout"
            logger.warning(last_diag)
            continue
        except Exception as e:
            last_diag = f"{model_id}: request error {e}"
            logger.warning(last_diag)
            continue

        sc = response.status_code

        # ── 200 OK ────────────────────────────────────────────
        if sc == 200:
            try:
                data = response.json()
            except Exception as e:
                last_diag = f"{model_id}: 200 but non-JSON ({e})"
                logger.warning(last_diag)
                continue
            text, finish = _parse_text(data)
            if text:
                return text
            # Empty text (safety block / MAX_TOKENS) — try next model
            last_diag = f"{model_id}: empty response (finishReason={finish or 'N/A'})"
            logger.warning(last_diag)
            continue

        # ── 404: model retired/unknown — try next ─────────────
        if sc == 404:
            last_diag = f"{model_id}: 404 not found — trying next model"
            logger.warning(last_diag)
            continue

        # ── 401/403: bad key — stop, report clearly ───────────
        if sc in (401, 403):
            return (
                "## ❌ Invalid or Unauthorized API Key\n\n"
                f"Gemini returned HTTP {sc}.\n\n"
                "- Get a free key at: https://aistudio.google.com/app/apikey\n"
                "- Keys start with 'AIza...'\n"
                "- Make sure the Generative Language API is enabled\n"
                "- Set it in Streamlit secrets as GEMINI_API_KEY\n\n"
                + _generate_ml_only_analysis(pred)
            )

        # ── 400: bad request — log, try next ──────────────────
        if sc == 400:
            try:
                err = response.json().get("error", {}).get("message", "Bad request")[:200]
            except Exception:
                err = response.text[:200]
            last_diag = f"{model_id}: 400 — {err}"
            logger.warning(last_diag)
            if "API_KEY" in err.upper() or "api key" in err.lower():
                return (
                    "## ❌ Invalid API Key\n\n"
                    "The Gemini API key appears to be invalid or expired.\n\n"
                    "- Get a free key at: https://aistudio.google.com/app/apikey\n"
                    "- Keys start with 'AIza...'\n\n"
                    + _generate_ml_only_analysis(pred)
                )
            continue

        # ── 429: rate limit — stop, report ────────────────────
        if sc == 429:
            logger.warning("Gemini rate limit hit")
            return (
                "## ⏱ Rate Limit Reached\n\n"
                "The Gemini free tier limit was hit (requests/minute or /day).\n"
                "Please wait 60 seconds and try again.\n\n"
                + _generate_ml_only_analysis(pred)
            )

        # ── 5xx: server error — try next ──────────────────────
        if 500 <= sc < 600:
            last_diag = f"{model_id}: server error {sc} — trying next model"
            logger.warning(last_diag)
            continue

        # ── Any other status — log, try next ──────────────────
        last_diag = f"{model_id}: unexpected HTTP {sc}: {response.text[:200]}"
        logger.warning(last_diag)
        continue

    # All models exhausted
    return (
        _generate_ml_only_analysis(pred)
        + f"\n\n*Diagnostic: all Gemini vision models failed. Last: {last_diag}*"
    )


# ════════════════════════════════════════════════════════════════
#  ENGINE CACHE  (persists across Streamlit reruns / refreshes)
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_engine() -> PredictionEngine:
    """Create & initialise the ML engine once per server process."""
    eng = PredictionEngine()
    eng.initialize()
    return eng


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="MNQ/NQ ML Suite v6",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    page, show_proj, auto_refresh = render_sidebar()

    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

    now_utc = datetime.utcnow()
    try:
        import zoneinfo
        from datetime import timezone as _tzmod
        _tz_ny = zoneinfo.ZoneInfo("America/New_York")
        now_ny = datetime.now(_tzmod.utc).astimezone(_tz_ny).replace(tzinfo=None)
    except Exception:
        _edt = 3 <= now_utc.month <= 11
        now_ny = now_utc - timedelta(hours=4 if _edt else 5)
    st.markdown(_header_html(now_ny, page), unsafe_allow_html=True)

    ny_hour   = now_ny.hour
    ny_minute = now_ny.minute
    market_open = (9 < ny_hour < 17) or (ny_hour == 9 and ny_minute >= 30)

    if "selected_tf" not in st.session_state:
        st.session_state["selected_tf"] = "1H"
    selected_tf = st.session_state["selected_tf"]

    _init_placeholder = st.empty()
    _init_placeholder.markdown(
        '<div style="background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);'
        'border-radius:12px;padding:14px 20px;margin-bottom:8px;">'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#00D4AA;'
        'letter-spacing:2px;">🔧 LOADING ML ENGINE v6...</span></div>',
        unsafe_allow_html=True,
    )
    engine: PredictionEngine = _get_engine()
    _init_placeholder.empty()

    if not engine.ready:
        st.error(
            "❌ Engine initialization failed. yfinance may be rate-limited or "
            "no intraday data available. Try 🔄 Retrain Model or wait 60s and refresh."
        )
        if st.button("🔄 Force Retry"):
            st.cache_resource.clear()
            st.rerun()
        return

    with st.spinner("📡 Computing v6 prediction…"):
        pred = engine.predict()

    if pred is None:
        st.error("❌ Prediction failed. Try retraining via the sidebar.")
        return

    with st.spinner(f"Loading {selected_tf} chart data…"):
        df_chart = DataPipeline.fetch_for_timeframe(selected_tf)

    news = fetch_market_news()

    sentiment, score       = NewsSentimentAnalyzer.score(news)
    pred.news_sentiment    = sentiment
    pred.news_sentiment_score = score

    if page == "Dashboard":
        page_dashboard(pred, show_proj, df_chart, selected_tf)
    elif page == "Charts & Orderflow":
        page_charts(pred, show_proj, df_chart, selected_tf)
    elif page == "Validation & Internals":
        page_validation(pred)
    elif page == "News & Outlook":
        page_news(pred, news)
    elif page == "Chart Analysis (AI)":
        page_chart_analysis(pred)

    st.markdown("""
<div style="text-align:center;padding:28px 0 8px 0;color:#2A2A3E;
            font-size:10px;border-top:1px solid #1F2937;margin-top:28px;
            font-family:'JetBrains Mono',monospace;letter-spacing:2px;">
  MNQ/NQ ML PREDICTOR v6 · AI EXPLAINABILITY · CVD ORDERFLOW · HMM REGIME ·
  PROBABILISTIC ONLY · NOT FINANCIAL ADVICE
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
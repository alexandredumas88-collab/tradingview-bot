#!/usr/bin/env python3
"""
Multi-strategy backtesting framework.
Assets : SP500/SPY, NAS100/QQQ, EURUSD, USDJPY, XAUUSD
Strategies: BB+RSI, Supertrend, Ichimoku, EMA Crossover, OPR Breakout
Targets : 40+ trades/year, PF > 2.0, Max DD < 8%
"""

import warnings
warnings.filterwarnings('ignore')

import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = {
    'SP500':  'SPY',
    'NAS100': 'QQQ',
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'XAUUSD': 'GC=F',
}

INITIAL_CAPITAL = 100_000
COMMISSION      = 0.0001   # 0.01 %
TRADE_SIZE      = 0.99     # fraction of equity per trade

TARGETS = dict(min_trades_yr=40, min_pf=2.0, max_dd_pct=8.0)

DATA_CACHE = os.path.join(os.path.dirname(__file__), 'pydata')
REPORT_OUT  = os.path.join(os.path.dirname(__file__), 'pybacktest-report.html')
RESULTS_OUT = os.path.join(os.path.dirname(__file__), 'pybacktest-results.json')

os.makedirs(DATA_CACHE, exist_ok=True)

# ── Data Download ─────────────────────────────────────────────────────────────

def download_data(name, ticker):
    cache = os.path.join(DATA_CACHE, f'{name}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f'  {name}: {len(df)} bars (cached)')
        return df

    print(f'  Downloading {name} ({ticker}) …')
    df = yf.download(ticker, period='730d', interval='1h', progress=False)
    if df.empty:
        print(f'  [WARN] No data for {name}')
        return None

    # yfinance 1.3+ returns MultiIndex columns even for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # backtesting.py needs tz-naive index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    df.to_csv(cache)
    print(f'  {name}: {len(df)} bars  ({df.index[0].date()} to {df.index[-1].date()})')
    return df

# ── Indicator helpers (return full numpy arrays for self.I()) ─────────────────

def _sma(arr, n):
    return pd.Series(arr).rolling(n, min_periods=n).mean().values

def _ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values

def _rsi(arr, n):
    s     = pd.Series(arr, dtype=float)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

def _bb_upper(arr, n, std=2.0):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() + std * s.rolling(n, min_periods=n).std(ddof=0)).values

def _bb_mid(arr, n):
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().values

def _bb_lower(arr, n, std=2.0):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() - std * s.rolling(n, min_periods=n).std(ddof=0)).values

def _atr(high, low, close, n=14):
    h, l, c = (pd.Series(x, dtype=float) for x in (high, low, close))
    pc  = c.shift(1)
    tr  = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean().values

def _supertrend_trend(high, low, close, n=10, mult=3.0):
    """Returns trend array: 1 = bullish, -1 = bearish."""
    atr = pd.Series(_atr(high, low, close, n))
    hl2 = (pd.Series(high, dtype=float) + pd.Series(low, dtype=float)) / 2
    raw_up = (hl2 + mult * atr).values
    raw_dn = (hl2 - mult * atr).values
    c      = np.array(close, dtype=float)

    up  = raw_up.copy()
    dn  = raw_dn.copy()
    trend = np.zeros(len(c))

    for i in range(1, len(c)):
        dn[i] = max(raw_dn[i], dn[i-1]) if c[i-1] > dn[i-1] else raw_dn[i]
        up[i] = min(raw_up[i], up[i-1]) if c[i-1] < up[i-1] else raw_up[i]
        if   c[i] > up[i-1]: trend[i] =  1
        elif c[i] < dn[i-1]: trend[i] = -1
        else:                  trend[i] = trend[i-1]

    return trend

def _ichimoku_span_a(high, low, t=9, k=26, d=26):
    h, l = pd.Series(high, dtype=float), pd.Series(low, dtype=float)
    tenkan = (h.rolling(t).max() + l.rolling(t).min()) / 2
    kijun  = (h.rolling(k).max() + l.rolling(k).min()) / 2
    return ((tenkan + kijun) / 2).shift(d).values

def _ichimoku_span_b(high, low, s=52, d=26):
    h, l = pd.Series(high, dtype=float), pd.Series(low, dtype=float)
    return ((h.rolling(s).max() + l.rolling(s).min()) / 2).shift(d).values

def _ichimoku_tenkan(high, low, t=9):
    h, l = pd.Series(high, dtype=float), pd.Series(low, dtype=float)
    return ((h.rolling(t).max() + l.rolling(t).min()) / 2).values

def _ichimoku_kijun(high, low, k=26):
    h, l = pd.Series(high, dtype=float), pd.Series(low, dtype=float)
    return ((h.rolling(k).max() + l.rolling(k).min()) / 2).values

# ── Strategies ────────────────────────────────────────────────────────────────

class BBRSIStrategy(Strategy):
    """Mean reversion: buy oversold bounce off lower Bollinger Band."""
    bb_period  = 30
    rsi_period = 7
    rsi_lo     = 35

    def init(self):
        c = self.data.Close
        self.bb_up  = self.I(_bb_upper, c, self.bb_period)
        self.bb_mid = self.I(_bb_mid,   c, self.bb_period)
        self.bb_lo  = self.I(_bb_lower, c, self.bb_period)
        self.rsi    = self.I(_rsi,      c, self.rsi_period)

    def next(self):
        if np.isnan(self.bb_lo[-1]) or np.isnan(self.rsi[-1]):
            return
        # Entry: RSI crosses up through threshold while price < lower BB
        rsi_cross = self.rsi[-2] < self.rsi_lo <= self.rsi[-1]
        if not self.position and self.data.Close[-1] < self.bb_lo[-1] and rsi_cross:
            self.buy(size=TRADE_SIZE)
        elif self.position.is_long and self.data.Close[-1] > self.bb_mid[-1]:
            self.position.close()


class SupertrendStrategy(Strategy):
    """Trend-following with Supertrend (ATR-based)."""
    atr_period = 10
    multiplier = 3.0

    def init(self):
        self.trend = self.I(
            _supertrend_trend,
            self.data.High, self.data.Low, self.data.Close,
            self.atr_period, self.multiplier,
        )

    def next(self):
        if np.isnan(self.trend[-1]) or self.trend[-2] == 0:
            return
        if self.trend[-2] < 0 < self.trend[-1]:      # bearish→bullish flip
            if self.position.is_short: self.position.close()
            if not self.position: self.buy(size=TRADE_SIZE)
        elif self.trend[-2] > 0 > self.trend[-1]:     # bullish→bearish flip
            if self.position.is_long: self.position.close()
            if not self.position: self.sell(size=TRADE_SIZE)


class IchimokuStrategy(Strategy):
    """Cloud breakout with Tenkan/Kijun trend filter."""

    def init(self):
        self.span_a  = self.I(_ichimoku_span_a,  self.data.High, self.data.Low)
        self.span_b  = self.I(_ichimoku_span_b,  self.data.High, self.data.Low)
        self.tenkan  = self.I(_ichimoku_tenkan,  self.data.High, self.data.Low)
        self.kijun   = self.I(_ichimoku_kijun,   self.data.High, self.data.Low)

    def next(self):
        sa, sb = self.span_a[-1], self.span_b[-1]
        if np.isnan(sa) or np.isnan(sb): return
        c   = self.data.Close[-1]
        top = max(sa, sb)
        bot = min(sa, sb)

        if not self.position:
            if c > top and self.tenkan[-1] > self.kijun[-1]:
                self.buy(size=TRADE_SIZE)
            elif c < bot and self.tenkan[-1] < self.kijun[-1]:
                self.sell(size=TRADE_SIZE)
        elif self.position.is_long  and c < bot:
            self.position.close()
        elif self.position.is_short and c > top:
            self.position.close()


class EMACrossStrategy(Strategy):
    """Classic fast/slow EMA crossover."""
    fast = 9
    slow = 21

    def init(self):
        c = self.data.Close
        self.ema_fast = self.I(_ema, c, self.fast)
        self.ema_slow = self.I(_ema, c, self.slow)

    def next(self):
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short: self.position.close()
            if not self.position: self.buy(size=TRADE_SIZE)
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long: self.position.close()
            if not self.position: self.sell(size=TRADE_SIZE)


class OPRBreakoutStrategy(Strategy):
    """
    Opening Price Range breakout.
    First bar of each calendar day = reference range.
    Enter on close that exceeds the range high/low.
    """
    sl_pct = 0.005   # 0.5 % stop loss from entry

    def init(self):
        self._opr_h  = np.nan
        self._opr_l  = np.nan
        self._cur_day = None

    def next(self):
        cur_day = self.data.index[-1].date()

        if cur_day != self._cur_day:          # new trading day
            self._cur_day = cur_day
            self._opr_h   = self.data.High[-1]
            self._opr_l   = self.data.Low[-1]
            if self.position: self.position.close()
            return

        if np.isnan(self._opr_h): return
        c = self.data.Close[-1]

        if not self.position:
            if c > self._opr_h:
                sl = c * (1 - self.sl_pct)
                self.buy(size=TRADE_SIZE, sl=sl)
            elif c < self._opr_l:
                sl = c * (1 + self.sl_pct)
                self.sell(size=TRADE_SIZE, sl=sl)

# ── Runner ────────────────────────────────────────────────────────────────────

STRATEGY_CLASSES = {
    'BB+RSI':        BBRSIStrategy,
    'Supertrend':    SupertrendStrategy,
    'Ichimoku':      IchimokuStrategy,
    'EMA Crossover': EMACrossStrategy,
    'OPR Breakout':  OPRBreakoutStrategy,
}

def run_combo(asset_name, df, strategy_name, StratClass):
    try:
        bt    = Backtest(df, StratClass, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()

        # ── pull stats ────────────────────────────────────────────────────────
        def g(key, default=np.nan):
            try: return stats[key]
            except Exception: return default

        trades    = int(g('# Trades', 0))
        pf_raw    = g('Profit Factor', np.nan)
        pf        = float(pf_raw) if not (isinstance(pf_raw, float) and np.isinf(pf_raw)) else 99.0
        wr        = float(g('Win Rate [%]', np.nan))
        dd        = abs(float(g('Max. Drawdown [%]', np.nan)))
        ret       = float(g('Return [%]', np.nan))
        sharpe    = float(g('Sharpe Ratio', np.nan))
        avg_trade = float(g('Avg. Trade [%]', np.nan))

        # years covered (at least 1 to avoid divide-by-zero)
        years = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
        tpy   = trades / years

        meets_t  = tpy   >= TARGETS['min_trades_yr']
        meets_pf = pf    >= TARGETS['min_pf']
        meets_dd = dd    <  TARGETS['max_dd_pct']
        qualifies = meets_t and meets_pf and meets_dd

        status = '✓' if qualifies else '✗'
        flags  = []
        if not meets_t:  flags.append(f'{tpy:.0f}/yr')
        if not meets_pf: flags.append(f'PF {pf:.2f}')
        if not meets_dd: flags.append(f'DD {dd:.1f}%')
        flag_str = ('  [' + ', '.join(flags) + ']') if flags else ''
        print(f'  {status} {asset_name:8s} {strategy_name:15s}  Trades: {trades:4d} ({tpy:.0f}/yr)  PF: {pf:.3f}  DD: {dd:.1f}%  Ret: {ret:.1f}%{flag_str}')

        return dict(
            asset=asset_name, strategy=strategy_name,
            trades=trades, trades_per_year=round(tpy, 1),
            profit_factor=round(pf, 4), win_rate=round(wr, 2),
            max_dd_pct=round(dd, 2), return_pct=round(ret, 2),
            sharpe=round(sharpe, 3), avg_trade_pct=round(avg_trade, 3),
            qualifies=qualifies, meets_trades=meets_t, meets_pf=meets_pf, meets_dd=meets_dd,
        )
    except Exception as e:
        print(f'  ✗ {asset_name:8s} {strategy_name:15s}  ERROR: {e}')
        return dict(asset=asset_name, strategy=strategy_name, error=str(e), qualifies=False)

# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(results):
    assets     = list(SYMBOLS.keys())
    strategies = list(STRATEGY_CLASSES.keys())
    qualified  = [r for r in results if r.get('qualifies')]

    # Build lookup: (asset, strategy) → result
    lookup = {(r['asset'], r['strategy']): r for r in results if not r.get('error')}

    def cell_class(r):
        if not r: return 'na'
        if r.get('error'): return 'err'
        if r.get('qualifies'): return 'hit'
        return 'miss'

    def cell_html(r):
        if not r: return '<td class="na">—</td>'
        if r.get('error'): return f'<td class="err" title="{r["error"]}">ERR</td>'
        pf  = r.get('profit_factor', 0)
        tpy = r.get('trades_per_year', 0)
        dd  = r.get('max_dd_pct', 0)
        ret = r.get('return_pct', 0)
        cls = 'hit' if r.get('qualifies') else 'miss'
        return (f'<td class="{cls}">'
                f'<div class="cval">PF {pf:.2f}</div>'
                f'<div class="csub">{tpy:.0f}/yr · DD {dd:.1f}%</div>'
                f'<div class="csub ret{"pos" if ret >= 0 else "neg"}">{ret:+.1f}%</div>'
                f'</td>')

    # Header row
    thead = '<tr><th>Asset \\ Strategy</th>' + ''.join(f'<th>{s}</th>' for s in strategies) + '</tr>'
    tbody_rows = []
    for asset in assets:
        row = f'<tr><th class="asset-th">{asset}</th>'
        for strat in strategies:
            r = lookup.get((asset, strat))
            row += cell_html(r)
        tbody_rows.append(row + '</tr>')

    # Best performers list
    valid = [r for r in results if not r.get('error')]
    ranked = sorted(valid, key=lambda r: (r.get('qualifies', False), r.get('profit_factor', 0)), reverse=True)

    COLORS = ['#f0b429','#4caf50','#2196f3','#9c27b0','#ff5722',
              '#00bcd4','#e91e63','#8bc34a','#ff9800','#3f51b5']

    top_cards = ''
    for i, r in enumerate(ranked[:8]):
        c = COLORS[i % len(COLORS)]
        cls = 'hit' if r.get('qualifies') else 'miss'
        banner = '<div class="qual-banner">✓ FTMO</div>' if r.get('qualifies') else ''
        top_cards += f'''<div class="card" style="border-top:3px solid {c}">
  {banner}
  <div class="card-rank" style="color:{c}">#{i+1}</div>
  <div class="card-name">{r["asset"]} · {r["strategy"]}</div>
  <div class="metric-grid">
    <div class="ms"><div class="mv">{r.get("profit_factor","?")}</div><div class="ml">PF</div></div>
    <div class="ms"><div class="mv">{r.get("win_rate","?")}%</div><div class="ml">Win Rate</div></div>
    <div class="ms"><div class="mv">{r.get("trades",0)} ({r.get("trades_per_year","?")}⁄yr)</div><div class="ml">Trades</div></div>
    <div class="ms"><div class="mv" style="color:{"#4caf50" if r.get("return_pct",0)>=0 else "#f44336"}">{r.get("return_pct","?")}%</div><div class="ml">Return</div></div>
    <div class="ms"><div class="mv" style="color:#f44336">{r.get("max_dd_pct","?")}%</div><div class="ml">Max DD</div></div>
    <div class="ms"><div class="mv">{r.get("sharpe","?")}</div><div class="ml">Sharpe</div></div>
  </div>
  <div class="card-badges">
    {"<span class='badge green'>40+/yr ✓</span>" if r.get("meets_trades") else "<span class='badge red'>freq ✗</span>"}
    {"<span class='badge green'>PF ✓</span>" if r.get("meets_pf") else "<span class='badge red'>PF ✗</span>"}
    {"<span class='badge green'>DD ✓</span>" if r.get("meets_dd") else "<span class='badge red'>DD ✗</span>"}
  </div>
</div>'''

    # Strategy column summary
    strat_summary = ''
    for s in strategies:
        s_results = [r for r in valid if r['strategy'] == s]
        q_count   = sum(1 for r in s_results if r.get('qualifies'))
        avg_pf    = np.nanmean([r.get('profit_factor', np.nan) for r in s_results]) if s_results else np.nan
        avg_tpy   = np.nanmean([r.get('trades_per_year', np.nan) for r in s_results]) if s_results else np.nan
        strat_summary += f'<tr><td>{s}</td><td>{q_count}/{len(assets)}</td><td>{avg_pf:.2f}</td><td>{avg_tpy:.0f}</td></tr>'

    # Rec box
    if qualified:
        best = sorted(qualified, key=lambda r: r.get('profit_factor', 0), reverse=True)[0]
        rec = f'<div class="rec good"><b>Top qualifier:</b> {best["asset"]} · {best["strategy"]} — PF {best["profit_factor"]} · {best["trades_per_year"]}/yr trades · Max DD {best["max_dd_pct"]}% · Return {best["return_pct"]}%</div>'
    else:
        best_pf  = max(valid, key=lambda r: r.get('profit_factor', 0), default=None)
        best_tpy = max(valid, key=lambda r: r.get('trades_per_year', 0), default=None)
        rec = f'''<div class="rec warn">
  <b>No combo meets all 3 FTMO targets (40+/yr · PF ≥ 2.0 · DD &lt; 8%).</b><br><br>
  Highest PF: <b>{best_pf["asset"]} · {best_pf["strategy"]}</b> (PF {best_pf.get("profit_factor","?")} · {best_pf.get("trades_per_year","?")} trd/yr)<br>
  Most active: <b>{best_tpy["asset"]} · {best_tpy["strategy"]}</b> ({best_tpy.get("trades_per_year","?")} trd/yr · PF {best_tpy.get("profit_factor","?")})<br><br>
  Suggestions: tune BB period / RSI threshold on BB+RSI, try 30min TF data, or combine EMA trend filter with BB mean reversion.
</div>'''

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi-Strategy Backtest Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:26px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:28px}}
h2{{font-size:17px;color:#fff;margin:32px 0 14px;border-left:3px solid #f0b429;padding-left:10px}}

/* stats bar */
.sbar{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}}
.sstat{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:130px}}
.sv{{display:block;font-size:20px;font-weight:bold;color:#f0b429}}
.sl{{display:block;font-size:11px;color:#666;margin-top:3px}}

/* grid table */
.grid-wrap{{overflow-x:auto;margin-bottom:28px}}
table.grid{{border-collapse:collapse;width:100%;min-width:900px}}
table.grid th{{background:#141414;color:#aaa;padding:10px 14px;border:1px solid #2a2a2a;font-size:12px;white-space:nowrap;text-align:center}}
table.grid .asset-th{{text-align:left;color:#f0b429;font-weight:bold}}
td.hit{{background:#0e2e0e;border:1px solid #2a4a2a;text-align:center;padding:8px 6px}}
td.miss{{background:#1a1a1a;border:1px solid #2a2a2a;text-align:center;padding:8px 6px}}
td.na{{background:#111;border:1px solid #1a1a1a;text-align:center;color:#444}}
td.err{{background:#2e0e0e;border:1px solid #4a2a2a;text-align:center;color:#f44336;cursor:help}}
.cval{{font-size:13px;font-weight:bold;color:#fff}}
.csub{{font-size:10px;color:#888;margin-top:2px}}
.csubpos{{color:#4caf50}}
.csubneg{{color:#f44336}}

/* cards */
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px}}
@media(max-width:900px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;position:relative;overflow:hidden}}
.qual-banner{{position:absolute;top:0;right:0;background:#1a3a1a;color:#4caf50;font-size:10px;padding:3px 8px;border-radius:0 12px 0 6px;font-weight:bold}}
.card-rank{{font-size:28px;font-weight:bold;margin-bottom:2px}}
.card-name{{font-size:12px;font-weight:bold;color:#fff;margin-bottom:10px}}
.metric-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px}}
.ms{{background:#111;border-radius:6px;padding:6px 8px;text-align:center}}
.mv{{font-size:13px;font-weight:bold;color:#f0b429}}
.ml{{font-size:9px;color:#555;margin-top:1px}}
.card-badges{{display:flex;gap:5px;flex-wrap:wrap}}
.badge{{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:bold}}
.badge.green{{background:#1a3a1a;color:#4caf50}}
.badge.red{{background:#3a1a1a;color:#f44336}}

/* strategy summary table */
.sum-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:28px}}
table.sum{{width:100%;border-collapse:collapse;font-size:13px}}
table.sum th{{color:#666;padding:8px 12px;text-align:left;border-bottom:1px solid #2a2a2a}}
table.sum td{{padding:8px 12px;border-bottom:1px solid #161616}}

.rec{{border-radius:10px;padding:16px 20px;margin-bottom:28px;font-size:13px;line-height:1.7}}
.rec.good{{background:#1a3a1a;border:1px solid #4caf50;color:#c8e6c9}}
.rec.warn{{background:#3a2a1a;border:1px solid #f0b429;color:#fff3cc}}
</style>
</head>
<body>
<h1>Multi-Strategy Backtest Report</h1>
<p class="sub">yfinance 2Y hourly · 5 assets · 5 strategies · Commission 0.01% · Generated {now}</p>

<div class="sbar">
  <div class="sstat"><span class="sv">{len(results)}</span><span class="sl">Total Backtests</span></div>
  <div class="sstat"><span class="sv">{len(qualified)}</span><span class="sl">Meet All Targets</span></div>
  <div class="sstat"><span class="sv">40+/yr</span><span class="sl">Trade Freq Target</span></div>
  <div class="sstat"><span class="sv">PF ≥ 2.0</span><span class="sl">Profit Factor Target</span></div>
  <div class="sstat"><span class="sv">DD &lt; 8%</span><span class="sl">Max Drawdown Target</span></div>
  <div class="sstat"><span class="sv">5 Assets</span><span class="sl">SP500 NAS100 EURUSD USDJPY XAUUSD</span></div>
</div>

<h2>Results Grid (green = all 3 targets met)</h2>
<div class="grid-wrap">
<table class="grid">
  <thead>{thead}</thead>
  <tbody>{''.join(tbody_rows)}</tbody>
</table>
</div>

<h2>Top 8 Combos (ranked by PF)</h2>
<div class="cards">{top_cards}</div>

<h2>Strategy Summary (avg across all assets)</h2>
<div class="sum-wrap">
<table class="sum">
  <thead><tr><th>Strategy</th><th>Qualifiers</th><th>Avg PF</th><th>Avg Trades/yr</th></tr></thead>
  <tbody>{strat_summary}</tbody>
</table>
</div>

<h2>Recommendation</h2>
{rec}
</body>
</html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] HTML report → {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Multi-Strategy Backtest  |  yfinance + backtesting.py')
    print('=' * 60)

    # Download all data
    print('\n[1/3] Downloading data …')
    datasets = {}
    for name, ticker in SYMBOLS.items():
        df = download_data(name, ticker)
        if df is not None and len(df) >= 100:
            datasets[name] = df

    print(f'\n[2/3] Running {len(datasets)} assets × {len(STRATEGY_CLASSES)} strategies = {len(datasets) * len(STRATEGY_CLASSES)} backtests …\n')

    results = []
    for asset_name, df in datasets.items():
        for strat_name, StratClass in STRATEGY_CLASSES.items():
            r = run_combo(asset_name, df, strat_name, StratClass)
            results.append(r)

    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\n[3/3] Generating report …')
    generate_report(results)

    # Console summary
    qualified = [r for r in results if r.get('qualifies')]
    print(f'\n{"="*60}')
    print(f'  QUALIFIED (all 3 targets): {len(qualified)}/{len(results)}')
    for r in sorted(qualified, key=lambda x: x.get('profit_factor', 0), reverse=True):
        print(f'  ✓ {r["asset"]:8s} {r["strategy"]:15s}  PF {r["profit_factor"]:.3f}  {r["trades_per_year"]:.0f}/yr  DD {r["max_dd_pct"]:.1f}%')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()

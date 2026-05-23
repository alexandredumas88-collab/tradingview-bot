#!/usr/bin/env python3
"""
Focused tuning run:
  A) XAUUSD Supertrend  mult = 1.5 and 2.0   (ATR period 10)
  B) BB+RSI level entry (rsi < threshold, not crossover)
       assets: XAUUSD, EURUSD, SP500, NAS100
       BB lengths: 10, 20, 30   RSI period: 7   threshold: 35
Target: 40+ trades/yr  PF > 2.0  Max DD < 8%
"""

import warnings
warnings.filterwarnings('ignore')

import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = {
    'SP500':  'SPY',
    'NAS100': 'QQQ',
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'XAUUSD': 'GC=F',
}
INITIAL_CAPITAL = 100_000
COMMISSION      = 0.0001   # 0.01 %
TRADE_SIZE      = 0.99
TARGETS         = dict(min_trades_yr=40, min_pf=2.0, max_dd_pct=8.0)

_here       = os.path.dirname(__file__)
DATA_CACHE  = os.path.join(_here, 'pydata')
REPORT_OUT  = os.path.join(_here, 'pytuned-report.html')
RESULTS_OUT = os.path.join(_here, 'pytuned-results.json')
os.makedirs(DATA_CACHE, exist_ok=True)

# ── Indicators ────────────────────────────────────────────────────────────────

def _bb_upper(arr, n, std=2.0):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() + std * s.rolling(n, min_periods=n).std(ddof=0)).values

def _bb_mid(arr, n):
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().values

def _bb_lower(arr, n, std=2.0):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() - std * s.rolling(n, min_periods=n).std(ddof=0)).values

def _rsi(arr, n):
    s     = pd.Series(arr, dtype=float)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

def _atr(high, low, close, n=14):
    h, l, c = (pd.Series(x, dtype=float) for x in (high, low, close))
    pc  = c.shift(1)
    tr  = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean().values

def _supertrend_trend(high, low, close, n=10, mult=3.0):
    atr    = pd.Series(_atr(high, low, close, n))
    hl2    = (pd.Series(high, dtype=float) + pd.Series(low, dtype=float)) / 2
    raw_up = (hl2 + mult * atr).values
    raw_dn = (hl2 - mult * atr).values
    c      = np.array(close, dtype=float)
    up, dn = raw_up.copy(), raw_dn.copy()
    trend  = np.zeros(len(c))
    for i in range(1, len(c)):
        dn[i] = max(raw_dn[i], dn[i-1]) if c[i-1] > dn[i-1] else raw_dn[i]
        up[i] = min(raw_up[i], up[i-1]) if c[i-1] < up[i-1] else raw_up[i]
        if   c[i] > up[i-1]: trend[i] =  1
        elif c[i] < dn[i-1]: trend[i] = -1
        else:                  trend[i] = trend[i-1]
    return trend

# ── Strategies ────────────────────────────────────────────────────────────────

class BBRSILevelStrategy(Strategy):
    """
    Mean reversion: enter when close < BB_lower AND rsi < threshold (level, not crossover).
    Exit when close crosses above BB_mid.
    Long-only.
    """
    bb_period  = 20
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
        below_lo = self.data.Close[-1] < self.bb_lo[-1]
        rsi_os   = self.rsi[-1] < self.rsi_lo   # level, not crossover

        if not self.position and below_lo and rsi_os:
            self.buy(size=TRADE_SIZE)
        elif self.position.is_long and self.data.Close[-1] > self.bb_mid[-1]:
            self.position.close()


class SupertrendStrategy(Strategy):
    """Bidirectional Supertrend trend-follower."""
    atr_period = 10
    multiplier = 3.0

    def init(self):
        self.trend = self.I(
            _supertrend_trend,
            self.data.High, self.data.Low, self.data.Close,
            self.atr_period, self.multiplier,
        )

    def next(self):
        if self.trend[-2] == 0:
            return
        if self.trend[-2] < 0 < self.trend[-1]:      # flip to bullish
            if self.position.is_short: self.position.close()
            if not self.position: self.buy(size=TRADE_SIZE)
        elif self.trend[-2] > 0 > self.trend[-1]:     # flip to bearish
            if self.position.is_long: self.position.close()
            if not self.position: self.sell(size=TRADE_SIZE)

# ── Combo list ────────────────────────────────────────────────────────────────

# Each entry: (asset_key, label, StratClass, {class_params})
COMBOS = []

# A) Supertrend tuned
for mult in [2.0, 1.5]:
    COMBOS.append(('XAUUSD', f'Supertrend mult={mult}', SupertrendStrategy, {'atr_period': 10, 'multiplier': mult}))

# B) BB+RSI level entry  –  4 assets × 3 BB lengths
for asset in ['XAUUSD', 'EURUSD', 'SP500', 'NAS100']:
    for bb in [10, 20, 30]:
        label = f'BB{bb} RSI<35 (level)'
        COMBOS.append((asset, label, BBRSILevelStrategy, {'bb_period': bb, 'rsi_period': 7, 'rsi_lo': 35}))

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(name):
    cache = os.path.join(DATA_CACHE, f'{name}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df
    ticker = SYMBOLS[name]
    print(f'  Downloading {name} ({ticker}) ...')
    df = yf.download(ticker, period='730d', interval='1h', progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    df.to_csv(cache)
    return df

# ── Runner ────────────────────────────────────────────────────────────────────

def run_combo(asset_name, label, StratClass, params):
    df = load_data(asset_name)
    if df is None or len(df) < 200:
        print(f'  [SKIP] {asset_name} no data')
        return None

    # Apply class-level params (sequential execution — no race condition)
    for k, v in params.items():
        setattr(StratClass, k, v)

    try:
        bt    = Backtest(df, StratClass, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()

        def g(key, default=np.nan):
            try:    return stats[key]
            except: return default

        trades = int(g('# Trades', 0))
        pf_raw = g('Profit Factor', np.nan)
        if isinstance(pf_raw, float) and np.isinf(pf_raw):
            pf = 99.0
        elif isinstance(pf_raw, float) and np.isnan(pf_raw):
            pf = 0.0
        else:
            pf = float(pf_raw)

        wr        = float(g('Win Rate [%]',    np.nan))
        dd        = abs(float(g('Max. Drawdown [%]', np.nan)))
        ret       = float(g('Return [%]',       np.nan))
        sharpe    = float(g('Sharpe Ratio',     np.nan))
        avg_trade = float(g('Avg. Trade [%]',   np.nan))

        years = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
        tpy   = trades / years

        mt  = tpy  >= TARGETS['min_trades_yr']
        mp  = pf   >= TARGETS['min_pf']
        md  = dd   <  TARGETS['max_dd_pct']
        ok  = mt and mp and md

        flags = []
        if not mt: flags.append(f'{tpy:.0f}/yr')
        if not mp: flags.append(f'PF {pf:.2f}')
        if not md: flags.append(f'DD {dd:.1f}%')
        flag_s = ('  [' + ', '.join(flags) + ']') if flags else ''
        sym    = 'OK' if ok else 'X'
        print(f'  {sym}  {asset_name:7s} {label:28s} {trades:4d}tr ({tpy:.0f}/yr)  PF {pf:.3f}  DD {dd:.1f}%  Ret {ret:+.1f}%{flag_s}')

        return dict(
            asset=asset_name, label=label, strategy=StratClass.__name__, params=params,
            trades=trades, trades_per_year=round(tpy, 1),
            profit_factor=round(pf, 4), win_rate=round(wr, 2),
            max_dd_pct=round(dd, 2), return_pct=round(ret, 2),
            sharpe=round(sharpe, 3), avg_trade_pct=round(avg_trade, 3),
            qualifies=ok, meets_trades=mt, meets_pf=mp, meets_dd=md,
        )
    except Exception as e:
        print(f'  X  {asset_name:7s} {label:28s}  ERROR: {e}')
        return dict(asset=asset_name, label=label, error=str(e), qualifies=False)

# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(results):
    valid     = [r for r in results if not r.get('error')]
    qualified = [r for r in valid if r.get('qualifies')]
    ranked    = sorted(valid, key=lambda r: (r.get('qualifies', False), r.get('profit_factor', 0)), reverse=True)

    COLORS = ['#f0b429','#4caf50','#2196f3','#9c27b0','#ff5722',
              '#00bcd4','#e91e63','#8bc34a','#ff9800','#3f51b5',
              '#cddc39','#607d8b','#795548','#9e9e9e']

    def badge(ok, yes_txt, no_txt):
        cls = 'green' if ok else 'red'
        txt = yes_txt if ok else no_txt
        return f'<span class="badge {cls}">{txt}</span>'

    # Cards
    cards_html = ''
    for i, r in enumerate(ranked):
        c      = COLORS[i % len(COLORS)]
        ok     = r.get('qualifies', False)
        banner = '<div class="qual-banner">FTMO</div>' if ok else ''
        dd_col = '#4caf50' if r.get('meets_dd') else '#f44336'
        np_col = '#4caf50' if r.get('return_pct', 0) >= 0 else '#f44336'
        cards_html += f'''<div class="card" style="border-top:3px solid {c}">
  {banner}
  <div class="rank" style="color:{c}">#{i+1}</div>
  <div class="cname">{r["asset"]}</div>
  <div class="clabel">{r["label"]}</div>
  <div class="mgrid">
    <div class="ms"><div class="mv">{r.get("profit_factor","?")}</div><div class="ml">PF</div></div>
    <div class="ms"><div class="mv">{r.get("win_rate","?")}%</div><div class="ml">Win Rate</div></div>
    <div class="ms"><div class="mv">{r.get("trades",0)}</div><div class="ml">Trades 2Y</div></div>
    <div class="ms"><div class="mv">{r.get("trades_per_year","?")}⁄yr</div><div class="ml">Frequency</div></div>
    <div class="ms"><div class="mv" style="color:{np_col}">{r.get("return_pct","?")}%</div><div class="ml">Return</div></div>
    <div class="ms"><div class="mv" style="color:{dd_col}">{r.get("max_dd_pct","?")}%</div><div class="ml">Max DD</div></div>
  </div>
  <div class="badges">
    {badge(r.get("meets_trades"), "40+/yr", f'{r.get("trades_per_year","?")} trd/yr')}
    {badge(r.get("meets_pf"),     "PF OK",  f'PF {r.get("profit_factor","?")} ')}
    {badge(r.get("meets_dd"),     "DD OK",  f'DD {r.get("max_dd_pct","?")}%')}
  </div>
</div>'''

    # Comparison table
    table_rows = ''
    for i, r in enumerate(ranked):
        c   = COLORS[i % len(COLORS)]
        ok  = r.get('qualifies', False)
        row_bg = 'background:#0e2e0e' if ok else ''
        table_rows += f'''<tr style="{row_bg}">
  <td style="color:{c};font-weight:bold">{r["asset"]}</td>
  <td>{r["label"]}</td>
  <td>{"FTMO" if ok else "-"}</td>
  <td style="color:{"#4caf50" if r.get("meets_trades") else "#f44336"}">{r.get("trades_per_year","?")} ({r.get("trades","?")} total)</td>
  <td style="color:{"#4caf50" if r.get("meets_pf") else "#f44336"}">{r.get("profit_factor","?")}</td>
  <td style="color:{"#4caf50" if r.get("meets_dd") else "#f44336"}">{r.get("max_dd_pct","?")}%</td>
  <td style="color:{"#4caf50" if r.get("return_pct",0)>=0 else "#f44336"}">{r.get("return_pct","?")}%</td>
  <td>{r.get("win_rate","?")}%</td>
  <td>{r.get("sharpe","?")}</td>
</tr>'''

    # Supertrend comparison (A group)
    st_rows = [r for r in results if r.get('strategy') == 'SupertrendStrategy' and not r.get('error')]
    st_tbl = ''
    for r in sorted(st_rows, key=lambda x: x.get('profit_factor', 0), reverse=True):
        p = r.get('params', {})
        ok = r.get('qualifies', False)
        bg = 'background:#0e2e0e' if ok else ''
        st_tbl += f'<tr style="{bg}"><td>{p.get("multiplier","?")}</td><td>{p.get("atr_period","?")}</td><td style="color:{"#4caf50" if r.get("meets_trades") else "#f44336"}">{r.get("trades_per_year","?")} ({r.get("trades","?")} total)</td><td style="color:{"#4caf50" if r.get("meets_pf") else "#f44336"}">{r.get("profit_factor","?")}</td><td style="color:{"#4caf50" if r.get("meets_dd") else "#f44336"}">{r.get("max_dd_pct","?")}%</td><td>{r.get("return_pct","?")}%</td></tr>'

    # BB+RSI heatmap — rows=asset, cols=BB length
    bb_assets = ['XAUUSD', 'EURUSD', 'SP500', 'NAS100']
    bb_lens   = [10, 20, 30]
    bb_lookup = {(r['asset'], r.get('params', {}).get('bb_period')): r
                 for r in results if r.get('strategy') == 'BBRSILevelStrategy' and not r.get('error')}

    heat_head = '<tr><th>Asset \\ BB</th>' + ''.join(f'<th>BB({n})</th>' for n in bb_lens) + '</tr>'
    heat_body = ''
    for asset in bb_assets:
        heat_body += f'<tr><th class="asset-th">{asset}</th>'
        for bb in bb_lens:
            r = bb_lookup.get((asset, bb))
            if not r:
                heat_body += '<td class="na">—</td>'
            else:
                ok  = r.get('qualifies', False)
                cls = 'hit' if ok else 'miss'
                heat_body += (f'<td class="{cls}">'
                              f'<div class="cval">PF {r["profit_factor"]:.2f}</div>'
                              f'<div class="csub">{r["trades_per_year"]:.0f}/yr</div>'
                              f'<div class="csub">DD {r["max_dd_pct"]:.1f}%</div>'
                              f'<div class="csub" style="color:{"#4caf50" if r.get("return_pct",0)>=0 else "#f44336"}">{r.get("return_pct","?")}%</div>'
                              f'</td>')
        heat_body += '</tr>'

    # Recommendation
    if qualified:
        best = sorted(qualified, key=lambda r: r['profit_factor'], reverse=True)[0]
        rec = (f'<div class="rec good"><b>Recommended for FTMO: {best["asset"]} + {best["label"]}</b><br>'
               f'PF {best["profit_factor"]} &nbsp;·&nbsp; {best["trades_per_year"]}/yr trades &nbsp;·&nbsp; '
               f'Max DD {best["max_dd_pct"]}% &nbsp;·&nbsp; Return {best["return_pct"]}% &nbsp;·&nbsp; '
               f'Win Rate {best["win_rate"]}% &nbsp;·&nbsp; Sharpe {best["sharpe"]}</div>')
    else:
        best_pf  = max(valid, key=lambda r: r.get('profit_factor', 0), default={})
        best_tpy = max(valid, key=lambda r: r.get('trades_per_year', 0), default={})
        rec = f'''<div class="rec warn">
<b>No combo meets all 3 targets yet.</b><br><br>
Highest PF: <b>{best_pf.get("asset")} {best_pf.get("label")}</b>
  — PF {best_pf.get("profit_factor","?")} · {best_pf.get("trades_per_year","?")}/yr · DD {best_pf.get("max_dd_pct","?")}%<br>
Most trades: <b>{best_tpy.get("asset")} {best_tpy.get("label")}</b>
  — {best_tpy.get("trades_per_year","?")}/yr · PF {best_tpy.get("profit_factor","?")} · DD {best_tpy.get("max_dd_pct","?")}%<br><br>
Next steps: Try adding a trend filter (e.g., only enter when 50-EMA slope is positive) to raise PF, or widen BB std-dev (2.5/3.0) to reduce trade frequency but improve entry quality.
</div>'''

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tuned Strategy Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:26px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:28px}}
h2{{font-size:17px;color:#fff;margin:32px 0 14px;border-left:3px solid #f0b429;padding-left:10px}}
/* stats bar */
.sbar{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:120px}}
.sv{{display:block;font-size:20px;font-weight:bold;color:#f0b429}}
.sl{{display:block;font-size:11px;color:#666;margin-top:3px}}
/* cards */
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:28px}}
@media(max-width:1000px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;position:relative}}
.qual-banner{{position:absolute;top:0;right:0;background:#1a3a1a;color:#4caf50;font-size:10px;padding:3px 9px;border-radius:0 12px 0 6px;font-weight:bold}}
.rank{{font-size:26px;font-weight:bold;margin-bottom:2px}}
.cname{{font-size:15px;font-weight:bold;color:#fff}}
.clabel{{font-size:11px;color:#888;margin-bottom:10px}}
.mgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px}}
.ms{{background:#111;border-radius:6px;padding:7px 8px;text-align:center}}
.mv{{font-size:13px;font-weight:bold;color:#f0b429}}
.ml{{font-size:9px;color:#555;margin-top:1px}}
.badges{{display:flex;gap:5px;flex-wrap:wrap}}
.badge{{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:bold}}
.badge.green{{background:#1a3a1a;color:#4caf50}}
.badge.red{{background:#3a1a1a;color:#f44336}}
/* tables */
.twrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:28px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#888;padding:8px 12px;text-align:left;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
td{{padding:9px 12px;border-bottom:1px solid #161616;vertical-align:middle}}
tr:hover td{{background:#1e1e1e}}
/* heatmap */
.hmap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:28px;overflow-x:auto}}
table.heat{{border-collapse:collapse;width:100%}}
table.heat th{{background:#141414;color:#aaa;padding:10px 14px;border:1px solid #2a2a2a;text-align:center;font-size:12px}}
.asset-th{{text-align:left!important;color:#f0b429;font-weight:bold}}
td.hit{{background:#0e2e0e;border:1px solid #2a4a2a;text-align:center;padding:10px 8px}}
td.miss{{background:#1a1a1a;border:1px solid #2a2a2a;text-align:center;padding:10px 8px}}
td.na{{background:#111;border:1px solid #1a1a1a;text-align:center;color:#444}}
.cval{{font-size:13px;font-weight:bold;color:#fff}}
.csub{{font-size:10px;color:#888;margin-top:2px}}
/* rec */
.rec{{border-radius:10px;padding:16px 20px;margin-bottom:28px;font-size:13px;line-height:1.7}}
.rec.good{{background:#1a3a1a;border:1px solid #4caf50;color:#c8e6c9}}
.rec.warn{{background:#3a2a1a;border:1px solid #f0b429;color:#fff3cc}}
</style>
</head>
<body>
<h1>Tuned Strategy Backtests</h1>
<p class="sub">yfinance 2Y hourly · Commission 0.01% · BB+RSI level entry (rsi &lt; 35) · Supertrend ATR(10) mult 1.5/2.0 · Generated {now}</p>

<div class="sbar">
  <div class="ss"><span class="sv">{len(results)}</span><span class="sl">Combos Run</span></div>
  <div class="ss"><span class="sv">{len(qualified)}</span><span class="sl">FTMO Qualified</span></div>
  <div class="ss"><span class="sv">40+/yr</span><span class="sl">Trade Freq Target</span></div>
  <div class="ss"><span class="sv">PF &gt; 2.0</span><span class="sl">Profit Factor</span></div>
  <div class="ss"><span class="sv">DD &lt; 8%</span><span class="sl">Max Drawdown</span></div>
</div>

<h2>All Results — Ranked by Profit Factor</h2>
<div class="cards">{cards_html}</div>

<h2>Head-to-Head Table</h2>
<div class="twrap">
<table>
  <thead><tr><th>Asset</th><th>Config</th><th>FTMO</th><th>Trades/yr</th><th>PF</th><th>Max DD</th><th>Return</th><th>Win Rate</th><th>Sharpe</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>
</div>

<h2>A) XAUUSD Supertrend — Multiplier Comparison (ATR 10)</h2>
<div class="twrap">
<table>
  <thead><tr><th>Multiplier</th><th>ATR Period</th><th>Trades/yr</th><th>PF</th><th>Max DD</th><th>Return</th></tr></thead>
  <tbody>{st_tbl}</tbody>
</table>
</div>

<h2>B) BB+RSI Level Entry — Heatmap (PF / trades/yr / Max DD)</h2>
<div class="hmap">
<table class="heat">
  <thead>{heat_head}</thead>
  <tbody>{heat_body}</tbody>
</table>
</div>

<h2>Recommendation</h2>
{rec}
</body>
</html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] Report saved: {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('  Tuned Backtests: Supertrend x2  +  BB+RSI level x12')
    print('=' * 65)
    print(f'\n  Combos: {len(COMBOS)}  |  Targets: 40+/yr  PF>2.0  DD<8%\n')

    results = []
    for asset, label, StratClass, params in COMBOS:
        r = run_combo(asset, label, StratClass, params)
        if r:
            results.append(r)

    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    generate_report(results)

    qualified = [r for r in results if r.get('qualifies')]
    print(f'\n{"="*65}')
    print(f'  QUALIFIED: {len(qualified)}/{len(results)}')
    for r in sorted(qualified, key=lambda x: x.get('profit_factor', 0), reverse=True):
        print(f'  OK  {r["asset"]:7s} {r["label"]:30s} PF {r["profit_factor"]:.3f}  {r["trades_per_year"]:.0f}/yr  DD {r["max_dd_pct"]:.1f}%')
    if not qualified:
        valid = [r for r in results if not r.get('error')]
        best  = max(valid, key=lambda r: r.get('profit_factor', 0), default=None)
        if best:
            print(f'  Closest: {best["asset"]} {best["label"]}  PF {best["profit_factor"]}  {best["trades_per_year"]}/yr  DD {best["max_dd_pct"]}%')
    print('=' * 65)

if __name__ == '__main__':
    main()

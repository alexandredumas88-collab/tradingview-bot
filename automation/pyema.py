#!/usr/bin/env python3
"""
EMA trend-filter sweep.
Strategy: BB+RSI level entry (rsi < 35) ONLY when close > EMA(n)  [uptrend gate].
Exit: close > BB_mid.
Grid: 2 assets x 3 BB lengths x 2 EMA periods = 12 combos.
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

SYMBOLS  = {'SP500': 'SPY', 'EURUSD': 'EURUSD=X'}
ASSETS   = ['SP500', 'EURUSD']
BB_LENS  = [10, 20, 30]
EMA_PERS = [50, 200]

INITIAL_CAPITAL = 100_000
COMMISSION      = 0.0001
TRADE_SIZE      = 0.99
TARGETS         = dict(min_trades_yr=40, min_pf=2.0, max_dd_pct=8.0)

_here       = os.path.dirname(__file__)
DATA_CACHE  = os.path.join(_here, 'pydata')
REPORT_OUT  = os.path.join(_here, 'pyema-report.html')
RESULTS_OUT = os.path.join(_here, 'pyema-results.json')
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

def _ema(arr, n):
    return pd.Series(arr, dtype=float).ewm(span=n, adjust=False).mean().values

# ── Strategy ──────────────────────────────────────────────────────────────────

class BBRSIEMAFilter(Strategy):
    """
    Long only when:
      1. close < BB_lower   (oversold price)
      2. rsi   < rsi_lo     (oversold momentum)
      3. close > EMA(ema_period)  (uptrend filter)
    Exit when close > BB_mid.
    """
    bb_period  = 20
    rsi_period = 7
    rsi_lo     = 35
    ema_period = 50

    def init(self):
        c = self.data.Close
        self.bb_up  = self.I(_bb_upper, c, self.bb_period)
        self.bb_mid = self.I(_bb_mid,   c, self.bb_period)
        self.bb_lo  = self.I(_bb_lower, c, self.bb_period)
        self.rsi    = self.I(_rsi,      c, self.rsi_period)
        self.ema    = self.I(_ema,      c, self.ema_period)

    def next(self):
        if np.isnan(self.bb_lo[-1]) or np.isnan(self.rsi[-1]) or np.isnan(self.ema[-1]):
            return
        c = self.data.Close[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < self.rsi_lo and c > self.ema[-1]:
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(name):
    cache = os.path.join(DATA_CACHE, f'{name}.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
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

def run_one(asset, df, bb, ema_p):
    BBRSIEMAFilter.bb_period  = bb
    BBRSIEMAFilter.ema_period = ema_p

    try:
        bt    = Backtest(df, BBRSIEMAFilter, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()

        def g(key, default=np.nan):
            try:    return stats[key]
            except: return default

        trades = int(g('# Trades', 0))
        pf_raw = g('Profit Factor', np.nan)
        if   isinstance(pf_raw, float) and np.isinf(pf_raw):  pf = 99.0
        elif isinstance(pf_raw, float) and np.isnan(pf_raw):  pf = 0.0
        else:                                                    pf = float(pf_raw)

        wr     = float(g('Win Rate [%]',       np.nan))
        dd     = abs(float(g('Max. Drawdown [%]', np.nan)))
        ret    = float(g('Return [%]',          np.nan))
        sharpe = float(g('Sharpe Ratio',        np.nan))
        avg_t  = float(g('Avg. Trade [%]',      np.nan))

        years = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
        tpy   = trades / years

        mt = tpy >= TARGETS['min_trades_yr']
        mp = pf  >= TARGETS['min_pf']
        md = dd  <  TARGETS['max_dd_pct']
        ok = mt and mp and md

        flags = []
        if not mt: flags.append(f'{tpy:.0f}/yr')
        if not mp: flags.append(f'PF {pf:.2f}')
        if not md: flags.append(f'DD {dd:.1f}%')
        flag_s = ('  [' + ', '.join(flags) + ']') if flags else ''
        sym    = 'OK' if ok else 'X '
        print(f'  {sym}  {asset:7s} BB{bb:2d}  EMA{ema_p:3d}   '
              f'{trades:4d}tr ({tpy:.0f}/yr)  PF {pf:.3f}  DD {dd:.1f}%  Ret {ret:+.1f}%{flag_s}')

        return dict(
            asset=asset, bb=bb, ema_p=ema_p,
            trades=trades, tpy=round(tpy, 1),
            pf=round(pf, 4), wr=round(wr, 2),
            dd=round(dd, 2), ret=round(ret, 2),
            sharpe=round(sharpe, 3), avg_t=round(avg_t, 3),
            ok=ok, mt=mt, mp=mp, md=md,
        )
    except Exception as e:
        print(f'  X   {asset:7s} BB{bb:2d}  EMA{ema_p:3d}   ERROR: {e}')
        return dict(asset=asset, bb=bb, ema_p=ema_p, error=str(e), ok=False)

# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(results, baseline):
    valid     = [r for r in results if not r.get('error')]
    qualified = [r for r in valid  if r['ok']]
    ranked    = sorted(valid, key=lambda r: (r['ok'], r['pf']), reverse=True)

    COLORS = ['#f0b429','#4caf50','#2196f3','#9c27b0','#ff5722','#00bcd4',
              '#e91e63','#8bc34a','#ff9800','#3f51b5','#cddc39','#607d8b']

    def badge(ok, yes_txt, no_txt):
        return f'<span class="badge {"green" if ok else "red"}">{yes_txt if ok else no_txt}</span>'

    # ── Cards ─────────────────────────────────────────────────────────────────
    cards_html = ''
    for i, r in enumerate(ranked):
        c      = COLORS[i % len(COLORS)]
        banner = '<div class="qual-banner">FTMO</div>' if r['ok'] else ''
        dd_col = '#4caf50' if r['md'] else '#f44336'
        rp_col = '#4caf50' if r['ret'] >= 0 else '#f44336'
        cards_html += f'''<div class="card" style="border-top:3px solid {c}">
  {banner}
  <div class="rank" style="color:{c}">#{i+1}</div>
  <div class="cname">{r["asset"]}</div>
  <div class="clabel">BB({r["bb"]}) · EMA({r["ema_p"]})</div>
  <div class="mgrid">
    <div class="ms"><div class="mv">{r["pf"]}</div><div class="ml">PF</div></div>
    <div class="ms"><div class="mv">{r["wr"]}%</div><div class="ml">Win Rate</div></div>
    <div class="ms"><div class="mv">{r["tpy"]}/yr</div><div class="ml">Freq</div></div>
    <div class="ms"><div class="mv">{r["trades"]}</div><div class="ml">Trades 2Y</div></div>
    <div class="ms"><div class="mv" style="color:{rp_col}">{r["ret"]}%</div><div class="ml">Return</div></div>
    <div class="ms"><div class="mv" style="color:{dd_col}">{r["dd"]}%</div><div class="ml">Max DD</div></div>
  </div>
  <div class="badges">
    {badge(r["mt"], "40+/yr",  f'{r["tpy"]}/yr')}
    {badge(r["mp"], "PF OK",   f'PF {r["pf"]}')}
    {badge(r["md"], "DD OK",   f'DD {r["dd"]}%')}
  </div>
</div>'''

    # ── Heatmaps: one per asset, rows = EMA, cols = BB ────────────────────────
    def heatmap(asset):
        lookup = {(r['bb'], r['ema_p']): r for r in valid if r['asset'] == asset}
        head   = '<tr><th>EMA filter \\ BB</th>' + ''.join(f'<th>BB({b})</th>' for b in BB_LENS) + '</tr>'
        body   = ''
        for ema_p in EMA_PERS:
            body += f'<tr><th class="sl-th">EMA({ema_p})</th>'
            for bb in BB_LENS:
                r = lookup.get((bb, ema_p))
                if not r:
                    body += '<td class="na">—</td>'
                    continue
                cls  = 'hit' if r['ok'] else 'miss'
                pf_c = '#4caf50' if r['mp'] else '#f44336'
                dd_c = '#4caf50' if r['md'] else '#f44336'
                fr_c = '#4caf50' if r['mt'] else '#f44336'
                body += (f'<td class="{cls}">'
                         f'<div class="cv" style="color:{pf_c}">PF {r["pf"]:.2f}</div>'
                         f'<div class="cs" style="color:{fr_c}">{r["tpy"]:.0f}/yr</div>'
                         f'<div class="cs" style="color:{dd_c}">DD {r["dd"]:.1f}%</div>'
                         f'<div class="cs">WR {r["wr"]:.0f}%</div>'
                         f'<div class="cs">Ret {r["ret"]:+.1f}%</div>'
                         f'</td>')
            body += '</tr>'
        return f'<table class="heat"><thead>{head}</thead><tbody>{body}</tbody></table>'

    # ── Baseline comparison table ──────────────────────────────────────────────
    base_rows = ''
    for b in baseline:
        vs = [r for r in valid if r['asset'] == b['asset'] and r['bb'] == b['bb']]
        delta_pf = ''
        delta_dd = ''
        best_v   = max(vs, key=lambda r: r['pf'], default=None) if vs else None
        if best_v:
            dpf = best_v['pf'] - b['pf']
            ddd = best_v['dd'] - b['dd']
            delta_pf = f'{"+" if dpf >= 0 else ""}{dpf:.3f}'
            delta_dd = f'{"+" if ddd >= 0 else ""}{ddd:.1f}%'
            dpf_c = '#4caf50' if dpf > 0 else '#f44336'
            ddd_c = '#4caf50' if ddd < 0 else '#f44336'
            best_cfg = f'EMA({best_v["ema_p"]})'
        else:
            dpf_c = ddd_c = '#888'
            best_cfg = '—'
        base_rows += (f'<tr>'
                      f'<td>{b["asset"]}</td><td>BB({b["bb"]})</td>'
                      f'<td>No filter</td>'
                      f'<td>{b["tpy"]}/yr</td><td>{b["pf"]}</td><td>{b["dd"]}%</td>'
                      f'<td>{best_cfg}</td>'
                      f'<td style="color:{dpf_c}">{delta_pf}</td>'
                      f'<td style="color:{ddd_c}">{delta_dd}</td>'
                      f'</tr>')

    # ── Full ranked table ──────────────────────────────────────────────────────
    tbl_rows = ''
    for i, r in enumerate(ranked):
        c  = COLORS[i % len(COLORS)]
        bg = 'background:#0e2e0e' if r['ok'] else ''
        tbl_rows += (f'<tr style="{bg}">'
                     f'<td style="color:{c};font-weight:bold">{r["asset"]}</td>'
                     f'<td>BB({r["bb"]})</td>'
                     f'<td>EMA({r["ema_p"]})</td>'
                     f'<td>{"FTMO" if r["ok"] else "—"}</td>'
                     f'<td style="color:{"#4caf50" if r["mt"] else "#f44336"}">{r["tpy"]} ({r["trades"]})</td>'
                     f'<td style="color:{"#4caf50" if r["mp"] else "#f44336"}">{r["pf"]}</td>'
                     f'<td style="color:{"#4caf50" if r["md"] else "#f44336"}">{r["dd"]}%</td>'
                     f'<td style="color:{"#4caf50" if r["ret"]>=0 else "#f44336"}">{r["ret"]:+.1f}%</td>'
                     f'<td>{r["wr"]}%</td>'
                     f'<td>{r["sharpe"]}</td>'
                     f'</tr>')

    # ── Recommendation ────────────────────────────────────────────────────────
    if qualified:
        best = sorted(qualified, key=lambda r: r['pf'], reverse=True)[0]
        rec = (f'<div class="rec good">'
               f'<b>Recommended for FTMO: {best["asset"]} BB({best["bb"]}) + EMA({best["ema_p"]}) filter</b><br>'
               f'PF {best["pf"]} &nbsp;·&nbsp; {best["tpy"]}/yr &nbsp;·&nbsp; '
               f'DD {best["dd"]}% &nbsp;·&nbsp; Return {best["ret"]}% &nbsp;·&nbsp; '
               f'Win Rate {best["wr"]}% &nbsp;·&nbsp; Sharpe {best["sharpe"]}'
               f'</div>')
    else:
        def gap(r):
            return (max(0, TARGETS['min_trades_yr'] - r['tpy']) / TARGETS['min_trades_yr'] +
                    max(0, TARGETS['min_pf']        - r['pf'])  / TARGETS['min_pf'] +
                    max(0, r['dd'] - TARGETS['max_dd_pct'])     / TARGETS['max_dd_pct'])
        closest  = min(valid, key=gap,          default=None)
        best_pf  = max(valid, key=lambda r: r['pf'], default=None)
        best_tpy = max(valid, key=lambda r: r['tpy'], default=None)
        c_txt = (f'{closest["asset"]} BB({closest["bb"]}) EMA({closest["ema_p"]}) — '
                 f'PF {closest["pf"]} · {closest["tpy"]}/yr · DD {closest["dd"]}%') if closest else '—'
        rec = (f'<div class="rec warn">'
               f'<b>No combo qualifies yet.</b><br><br>'
               f'Closest overall: <b>{c_txt}</b><br>'
               f'Highest PF: <b>{best_pf["asset"]} BB({best_pf["bb"]}) EMA({best_pf["ema_p"]})</b> — PF {best_pf["pf"]} · {best_pf["tpy"]}/yr · DD {best_pf["dd"]}%<br>'
               f'Most trades: <b>{best_tpy["asset"]} BB({best_tpy["bb"]}) EMA({best_tpy["ema_p"]})</b> — {best_tpy["tpy"]}/yr · PF {best_tpy["pf"]} · DD {best_tpy["dd"]}%<br><br>'
               f'If PF is still the blocker at sufficient frequency, the next lever is entry timing: require <code>rsi crossover</code> (recovery confirmation) rather than level, or add a second EMA (e.g., EMA20 slope positive) as a short-term momentum gate.'
               f'</div>')

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EMA Filter Sweep — SP500 &amp; EURUSD</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:26px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:28px}}
h2{{font-size:17px;color:#fff;margin:32px 0 14px;border-left:3px solid #f0b429;padding-left:10px}}
.sbar{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:120px}}
.sv{{display:block;font-size:20px;font-weight:bold;color:#f0b429}}
.sl{{display:block;font-size:11px;color:#666;margin-top:3px}}
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
.twrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:28px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#888;padding:8px 12px;text-align:left;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
td{{padding:9px 12px;border-bottom:1px solid #161616;vertical-align:middle}}
tr:hover td{{background:#1e1e1e}}
code{{background:#111;padding:1px 5px;border-radius:3px;color:#f0b429;font-size:12px}}
.hmap-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:28px;overflow-x:auto}}
table.heat{{border-collapse:collapse;min-width:480px}}
table.heat th{{background:#141414;color:#aaa;padding:10px 18px;border:1px solid #2a2a2a;text-align:center;font-size:12px}}
.sl-th{{text-align:left!important;color:#f0b429;font-weight:bold;white-space:nowrap}}
td.hit{{background:#0e2e0e;border:1px solid #2a4a2a;text-align:center;padding:12px 16px;min-width:120px}}
td.miss{{background:#1a1a1a;border:1px solid #2a2a2a;text-align:center;padding:12px 16px;min-width:120px}}
td.na{{background:#111;border:1px solid #1a1a1a;text-align:center;color:#444}}
.cv{{font-size:14px;font-weight:bold}}
.cs{{font-size:11px;color:#aaa;margin-top:3px}}
.rec{{border-radius:10px;padding:16px 20px;margin-bottom:28px;font-size:13px;line-height:1.7}}
.rec.good{{background:#1a3a1a;border:1px solid #4caf50;color:#c8e6c9}}
.rec.warn{{background:#3a2a1a;border:1px solid #f0b429;color:#fff3cc}}
</style>
</head>
<body>
<h1>EMA Trend-Filter Sweep</h1>
<p class="sub">BB+RSI level entry (rsi &lt; 35) · Only enter when close &gt; EMA(n) · BB 10/20/30 · EMA 50/200 · 2Y hourly · Commission 0.01% · Generated {now}</p>

<div class="sbar">
  <div class="ss"><span class="sv">{len(results)}</span><span class="sl">Combos Run</span></div>
  <div class="ss"><span class="sv">{len(qualified)}</span><span class="sl">FTMO Qualified</span></div>
  <div class="ss"><span class="sv">40+/yr</span><span class="sl">Trade Freq</span></div>
  <div class="ss"><span class="sv">PF &gt; 2.0</span><span class="sl">Profit Factor</span></div>
  <div class="ss"><span class="sv">DD &lt; 8%</span><span class="sl">Max Drawdown</span></div>
</div>

<h2>All Combos — Ranked by Profit Factor</h2>
<div class="cards">{cards_html}</div>

<h2>SP500 — Heatmap (EMA filter vs BB length)</h2>
<div class="hmap-wrap">{heatmap('SP500')}</div>

<h2>EURUSD — Heatmap</h2>
<div class="hmap-wrap">{heatmap('EURUSD')}</div>

<h2>vs. No-Filter Baseline (best EMA config per combo)</h2>
<div class="twrap">
<table>
  <thead><tr><th>Asset</th><th>BB</th><th>Filter</th><th>Trades/yr</th><th>PF (base)</th><th>DD (base)</th><th>Best EMA</th><th>PF delta</th><th>DD delta</th></tr></thead>
  <tbody>{base_rows}</tbody>
</table>
</div>

<h2>Full Ranked Table</h2>
<div class="twrap">
<table>
  <thead><tr><th>Asset</th><th>BB</th><th>EMA</th><th>FTMO</th><th>Trades/yr</th><th>PF</th><th>Max DD</th><th>Return</th><th>Win Rate</th><th>Sharpe</th></tr></thead>
  <tbody>{tbl_rows}</tbody>
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

# No-filter baselines from pytuned.py run (for the delta comparison table)
BASELINE = [
    dict(asset='SP500',  bb=10, tpy=45.0, pf=1.648, dd=9.8),
    dict(asset='SP500',  bb=20, tpy=30.0, pf=2.040, dd=10.4),
    dict(asset='SP500',  bb=30, tpy=21.0, pf=2.038, dd=10.9),
    dict(asset='EURUSD', bb=10, tpy=175.0, pf=1.004, dd=3.9),
    dict(asset='EURUSD', bb=20, tpy=127.0, pf=1.179, dd=4.6),
    dict(asset='EURUSD', bb=30, tpy=84.0,  pf=1.064, dd=5.2),
]

def main():
    n_combos = len(ASSETS) * len(BB_LENS) * len(EMA_PERS)
    print('=' * 65)
    print('  EMA Filter Sweep: SP500 + EURUSD · BB 10/20/30 · EMA 50/200')
    print('=' * 65)
    print(f'\n  {len(ASSETS)} assets x {len(BB_LENS)} BB x {len(EMA_PERS)} EMA = {n_combos} combos\n')

    datasets = {}
    for name in ASSETS:
        df = load_data(name)
        if df is not None and len(df) >= 200:
            datasets[name] = df
            print(f'  {name}: {len(df)} bars  ({df.index[0].date()} to {df.index[-1].date()})')

    print()
    results = []
    for asset in ASSETS:
        if asset not in datasets:
            continue
        df = datasets[asset]
        for bb in BB_LENS:
            for ema_p in EMA_PERS:
                r = run_one(asset, df, bb, ema_p)
                if r:
                    results.append(r)

    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    generate_report(results, BASELINE)

    qualified = [r for r in results if r.get('ok')]
    print(f'\n{"="*65}')
    print(f'  QUALIFIED: {len(qualified)}/{len(results)}')
    for r in sorted(qualified, key=lambda x: x['pf'], reverse=True):
        print(f'  OK  {r["asset"]:7s} BB({r["bb"]}) EMA({r["ema_p"]})  '
              f'PF {r["pf"]:.3f}  {r["tpy"]:.0f}/yr  DD {r["dd"]:.1f}%  Ret {r["ret"]:+.1f}%')
    if not qualified:
        valid = [r for r in results if not r.get('error')]
        def gap(r):
            return (max(0, TARGETS['min_trades_yr'] - r['tpy']) / TARGETS['min_trades_yr'] +
                    max(0, TARGETS['min_pf']        - r['pf'])  / TARGETS['min_pf'] +
                    max(0, r['dd'] - TARGETS['max_dd_pct'])     / TARGETS['max_dd_pct'])
        closest = min(valid, key=gap, default=None)
        if closest:
            print(f'  Closest: {closest["asset"]} BB({closest["bb"]}) EMA({closest["ema_p"]})  '
                  f'PF {closest["pf"]}  {closest["tpy"]}/yr  DD {closest["dd"]}%')
    print('=' * 65)

if __name__ == '__main__':
    main()

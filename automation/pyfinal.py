#!/usr/bin/env python3
"""
Final qualifier sweep.
Assets  : EURUSD, SP500
BB      : 10, 20
EMA     : 100, 200
Entry   : level (rsi < 35) vs crossover (rsi crosses up through 35)
Exit    : close > BB_mid
Total   : 2 x 2 x 2 x 2 = 16 combos
Target  : 40+ trades/yr  PF > 2.0  Max DD < 8%
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
ASSETS   = ['EURUSD', 'SP500']
BB_LENS  = [10, 20]
EMA_PERS = [100, 200]
ENTRIES  = ['level', 'crossover']

INITIAL_CAPITAL = 100_000
COMMISSION      = 0.0001
TRADE_SIZE      = 0.99
TARGETS         = dict(min_trades_yr=40, min_pf=2.0, max_dd_pct=8.0)

_here       = os.path.dirname(__file__)
DATA_CACHE  = os.path.join(_here, 'pydata')
REPORT_OUT  = os.path.join(_here, 'pyfinal-report.html')
RESULTS_OUT = os.path.join(_here, 'pyfinal-results.json')
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

class BBRSIFiltered(Strategy):
    """
    Long only when:
      close < BB_lower  AND  rsi < rsi_lo (level) or rsi crosses up (crossover)
      AND  close > EMA(ema_period)
    Exit: close > BB_mid
    use_crossover=True  → entry only on RSI recovery signal
    use_crossover=False → entry any bar RSI is in oversold zone
    """
    bb_period    = 20
    rsi_period   = 7
    rsi_lo       = 35
    ema_period   = 200
    use_crossover = False   # set to True for crossover entry

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

        if self.use_crossover:
            rsi_signal = self.rsi[-2] < self.rsi_lo <= self.rsi[-1]
        else:
            rsi_signal = self.rsi[-1] < self.rsi_lo

        if not self.position:
            if c < self.bb_lo[-1] and rsi_signal and c > self.ema[-1]:
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

def run_one(asset, df, bb, ema_p, entry):
    BBRSIFiltered.bb_period     = bb
    BBRSIFiltered.ema_period    = ema_p
    BBRSIFiltered.use_crossover = (entry == 'crossover')

    try:
        bt    = Backtest(df, BBRSIFiltered, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
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

        flags  = []
        if not mt: flags.append(f'{tpy:.0f}/yr')
        if not mp: flags.append(f'PF {pf:.2f}')
        if not md: flags.append(f'DD {dd:.1f}%')
        flag_s = ('  [' + ', '.join(flags) + ']') if flags else ''
        sym    = 'OK' if ok else 'X '
        print(f'  {sym}  {asset:7s} BB{bb:2d} EMA{ema_p:3d} {entry:10s}  '
              f'{trades:4d}tr ({tpy:.0f}/yr)  PF {pf:.3f}  DD {dd:.1f}%  Ret {ret:+.1f}%{flag_s}')

        return dict(
            asset=asset, bb=bb, ema_p=ema_p, entry=entry,
            trades=trades, tpy=round(tpy, 1),
            pf=round(pf, 4), wr=round(wr, 2),
            dd=round(dd, 2), ret=round(ret, 2),
            sharpe=round(sharpe, 3), avg_t=round(avg_t, 3),
            ok=ok, mt=mt, mp=mp, md=md,
        )
    except Exception as e:
        print(f'  X   {asset:7s} BB{bb:2d} EMA{ema_p:3d} {entry:10s}  ERROR: {e}')
        return dict(asset=asset, bb=bb, ema_p=ema_p, entry=entry, error=str(e), ok=False)

# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(results):
    valid     = [r for r in results if not r.get('error')]
    qualified = [r for r in valid  if r['ok']]
    ranked    = sorted(valid, key=lambda r: (r['ok'], r['pf']), reverse=True)

    COLORS = ['#f0b429','#4caf50','#2196f3','#9c27b0','#ff5722','#00bcd4',
              '#e91e63','#8bc34a','#ff9800','#3f51b5','#cddc39','#607d8b',
              '#795548','#9e9e9e','#ff7043','#26c6da']

    def badge(ok, yes_txt, no_txt):
        return f'<span class="badge {"green" if ok else "red"}">{yes_txt if ok else no_txt}</span>'

    def pf_color(v):   return '#4caf50' if v >= TARGETS['min_pf']        else '#f44336'
    def tpy_color(v):  return '#4caf50' if v >= TARGETS['min_trades_yr'] else '#f44336'
    def dd_color(v):   return '#4caf50' if v <  TARGETS['max_dd_pct']    else '#f44336'
    def ret_color(v):  return '#4caf50' if v >= 0 else '#f44336'

    # ── Cards ─────────────────────────────────────────────────────────────────
    cards_html = ''
    for i, r in enumerate(ranked):
        c      = COLORS[i % len(COLORS)]
        banner = '<div class="qual-banner">FTMO</div>' if r['ok'] else ''
        cards_html += f'''<div class="card" style="border-top:3px solid {c}">
  {banner}
  <div class="rank" style="color:{c}">#{i+1}</div>
  <div class="cname">{r["asset"]}</div>
  <div class="clabel">BB({r["bb"]}) · EMA({r["ema_p"]}) · {r["entry"]}</div>
  <div class="mgrid">
    <div class="ms"><div class="mv" style="color:{pf_color(r["pf"])}">{r["pf"]}</div><div class="ml">PF</div></div>
    <div class="ms"><div class="mv">{r["wr"]}%</div><div class="ml">Win Rate</div></div>
    <div class="ms"><div class="mv" style="color:{tpy_color(r["tpy"])}">{r["tpy"]}/yr</div><div class="ml">Freq</div></div>
    <div class="ms"><div class="mv">{r["trades"]}</div><div class="ml">Trades 2Y</div></div>
    <div class="ms"><div class="mv" style="color:{ret_color(r["ret"])}">{r["ret"]}%</div><div class="ml">Return</div></div>
    <div class="ms"><div class="mv" style="color:{dd_color(r["dd"])}">{r["dd"]}%</div><div class="ml">Max DD</div></div>
  </div>
  <div class="badges">
    {badge(r["mt"], "40+/yr",  f'{r["tpy"]}/yr')}
    {badge(r["mp"], "PF OK",   f'PF {r["pf"]}')}
    {badge(r["md"], "DD OK",   f'DD {r["dd"]}%')}
  </div>
</div>'''

    # ── Per-asset heatmaps: rows=entry×EMA, cols=BB ───────────────────────────
    def heatmap(asset):
        lookup = {(r['bb'], r['ema_p'], r['entry']): r
                  for r in valid if r['asset'] == asset}
        head = '<tr><th>Entry · EMA</th>' + ''.join(f'<th>BB({b})</th>' for b in BB_LENS) + '</tr>'
        body = ''
        for entry in ENTRIES:
            for ema_p in EMA_PERS:
                row_lbl = f'{entry.capitalize()} · EMA({ema_p})'
                body += f'<tr><th class="row-th">{row_lbl}</th>'
                for bb in BB_LENS:
                    r = lookup.get((bb, ema_p, entry))
                    if not r:
                        body += '<td class="na">—</td>'
                        continue
                    cls = 'hit' if r['ok'] else 'miss'
                    body += (f'<td class="{cls}">'
                             f'<div class="cv" style="color:{pf_color(r["pf"])}">PF {r["pf"]:.2f}</div>'
                             f'<div class="cs" style="color:{tpy_color(r["tpy"])}">{r["tpy"]:.0f}/yr</div>'
                             f'<div class="cs" style="color:{dd_color(r["dd"])}">DD {r["dd"]:.1f}%</div>'
                             f'<div class="cs">WR {r["wr"]:.0f}%  Ret {r["ret"]:+.1f}%</div>'
                             f'</td>')
                body += '</tr>'
        return f'<table class="heat"><thead>{head}</thead><tbody>{body}</tbody></table>'

    # ── Level vs crossover comparison (grouped by asset+BB+EMA) ──────────────
    cmp_rows = ''
    for asset in ASSETS:
        for bb in BB_LENS:
            for ema_p in EMA_PERS:
                rl = next((r for r in valid if r['asset']==asset and r['bb']==bb
                           and r['ema_p']==ema_p and r['entry']=='level'), None)
                rc = next((r for r in valid if r['asset']==asset and r['bb']==bb
                           and r['ema_p']==ema_p and r['entry']=='crossover'), None)
                if not rl or not rc: continue

                winner = 'level' if rl['pf'] >= rc['pf'] else 'crossover'
                cmp_rows += (
                    f'<tr>'
                    f'<td>{asset}</td><td>BB({bb})</td><td>EMA({ema_p})</td>'
                    f'<td style="color:{pf_color(rl["pf"])}">{rl["pf"]}</td>'
                    f'<td style="color:{tpy_color(rl["tpy"])}">{rl["tpy"]}/yr</td>'
                    f'<td style="color:{dd_color(rl["dd"])}">{rl["dd"]}%</td>'
                    f'<td style="color:{pf_color(rc["pf"])}">{rc["pf"]}</td>'
                    f'<td style="color:{tpy_color(rc["tpy"])}">{rc["tpy"]}/yr</td>'
                    f'<td style="color:{dd_color(rc["dd"])}">{rc["dd"]}%</td>'
                    f'<td style="font-weight:bold;color:#f0b429">{winner}</td>'
                    f'</tr>'
                )

    # ── Full ranked table ──────────────────────────────────────────────────────
    tbl_rows = ''
    for i, r in enumerate(ranked):
        c  = COLORS[i % len(COLORS)]
        bg = 'background:#0e2e0e' if r['ok'] else ''
        tbl_rows += (f'<tr style="{bg}">'
                     f'<td style="color:{c};font-weight:bold">{r["asset"]}</td>'
                     f'<td>BB({r["bb"]})</td><td>EMA({r["ema_p"]})</td><td>{r["entry"]}</td>'
                     f'<td>{"FTMO" if r["ok"] else "—"}</td>'
                     f'<td style="color:{tpy_color(r["tpy"])}">{r["tpy"]} ({r["trades"]})</td>'
                     f'<td style="color:{pf_color(r["pf"])}">{r["pf"]}</td>'
                     f'<td style="color:{dd_color(r["dd"])}">{r["dd"]}%</td>'
                     f'<td style="color:{ret_color(r["ret"])}">{r["ret"]:+.1f}%</td>'
                     f'<td>{r["wr"]}%</td><td>{r["sharpe"]}</td>'
                     f'</tr>')

    # ── Recommendation ────────────────────────────────────────────────────────
    if qualified:
        best = sorted(qualified, key=lambda r: r['pf'], reverse=True)[0]
        rec = (f'<div class="rec good">'
               f'<b>FTMO Qualifier Found: {best["asset"]} BB({best["bb"]}) + EMA({best["ema_p"]}) + {best["entry"]} entry</b><br><br>'
               f'PF <b>{best["pf"]}</b> &nbsp;·&nbsp; '
               f'{best["tpy"]}/yr ({best["trades"]} trades over 2Y) &nbsp;·&nbsp; '
               f'Max DD <b>{best["dd"]}%</b> &nbsp;·&nbsp; '
               f'Return <b>{best["ret"]}%</b> &nbsp;·&nbsp; '
               f'Win Rate {best["wr"]}% &nbsp;·&nbsp; Sharpe {best["sharpe"]}'
               f'</div>')
    else:
        def gap(r):
            return (max(0, TARGETS['min_trades_yr'] - r['tpy']) / TARGETS['min_trades_yr'] +
                    max(0, TARGETS['min_pf']        - r['pf'])  / TARGETS['min_pf'] +
                    max(0, r['dd'] - TARGETS['max_dd_pct'])     / TARGETS['max_dd_pct'])
        closest = min(valid, key=gap, default=None)
        best_pf = max(valid, key=lambda r: r['pf'], default=None)
        rec = (f'<div class="rec warn">'
               f'<b>No combo qualifies yet.</b><br><br>'
               f'Closest: <b>{closest["asset"]} BB({closest["bb"]}) EMA({closest["ema_p"]}) {closest["entry"]}</b>'
               f' — PF {closest["pf"]} · {closest["tpy"]}/yr · DD {closest["dd"]}%<br>'
               f'Highest PF: <b>{best_pf["asset"]} BB({best_pf["bb"]}) EMA({best_pf["ema_p"]}) {best_pf["entry"]}</b>'
               f' — PF {best_pf["pf"]} · {best_pf["tpy"]}/yr · DD {best_pf["dd"]}%'
               f'</div>')

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Final Qualifier Sweep</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1500px;margin:0 auto}}
h1{{font-size:26px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:28px}}
h2{{font-size:17px;color:#fff;margin:32px 0 14px;border-left:3px solid #f0b429;padding-left:10px}}
.sbar{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:120px}}
.sv{{display:block;font-size:20px;font-weight:bold;color:#f0b429}}
.sl{{display:block;font-size:11px;color:#666;margin-top:3px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:28px}}
@media(max-width:1100px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;position:relative}}
.qual-banner{{position:absolute;top:0;right:0;background:#1a3a1a;color:#4caf50;font-size:10px;padding:3px 9px;border-radius:0 12px 0 6px;font-weight:bold;letter-spacing:1px}}
.rank{{font-size:26px;font-weight:bold;margin-bottom:2px}}
.cname{{font-size:15px;font-weight:bold;color:#fff}}
.clabel{{font-size:11px;color:#888;margin-bottom:10px}}
.mgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px}}
.ms{{background:#111;border-radius:6px;padding:7px 8px;text-align:center}}
.mv{{font-size:13px;font-weight:bold}}
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
.hmap-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:28px;overflow-x:auto}}
table.heat{{border-collapse:collapse;min-width:420px}}
table.heat th{{background:#141414;color:#aaa;padding:10px 18px;border:1px solid #2a2a2a;text-align:center;font-size:12px}}
.row-th{{text-align:left!important;color:#f0b429;font-weight:bold;white-space:nowrap;min-width:180px}}
td.hit{{background:#0e2e0e;border:1px solid #2a4a2a;text-align:center;padding:12px 18px;min-width:130px}}
td.miss{{background:#1a1a1a;border:1px solid #2a2a2a;text-align:center;padding:12px 18px;min-width:130px}}
td.na{{background:#111;border:1px solid #1a1a1a;text-align:center;color:#444}}
.cv{{font-size:14px;font-weight:bold}}
.cs{{font-size:11px;color:#aaa;margin-top:3px}}
.rec{{border-radius:10px;padding:20px 24px;margin-bottom:28px;font-size:14px;line-height:1.8}}
.rec.good{{background:#1a3a1a;border:1px solid #4caf50;color:#c8e6c9}}
.rec.warn{{background:#3a2a1a;border:1px solid #f0b429;color:#fff3cc}}
</style>
</head>
<body>
<h1>Final Qualifier Sweep</h1>
<p class="sub">EURUSD + SP500 · BB 10/20 · EMA 100/200 · Level vs Crossover RSI entry · 2Y hourly · Commission 0.01% · Generated {now}</p>

<div class="sbar">
  <div class="ss"><span class="sv">{len(results)}</span><span class="sl">Combos Tested</span></div>
  <div class="ss"><span class="sv">{len(qualified)}</span><span class="sl">FTMO Qualified</span></div>
  <div class="ss"><span class="sv">40+/yr</span><span class="sl">Trade Freq Target</span></div>
  <div class="ss"><span class="sv">PF &gt; 2.0</span><span class="sl">Profit Factor</span></div>
  <div class="ss"><span class="sv">DD &lt; 8%</span><span class="sl">Max Drawdown</span></div>
</div>

<h2>All 16 Combos — Ranked by Profit Factor</h2>
<div class="cards">{cards_html}</div>

<h2>EURUSD — Heatmap</h2>
<div class="hmap-wrap">{heatmap('EURUSD')}</div>

<h2>SP500 — Heatmap</h2>
<div class="hmap-wrap">{heatmap('SP500')}</div>

<h2>Level vs Crossover — Head to Head</h2>
<div class="twrap">
<table>
  <thead>
    <tr>
      <th>Asset</th><th>BB</th><th>EMA</th>
      <th>Level PF</th><th>Level /yr</th><th>Level DD</th>
      <th>Cross PF</th><th>Cross /yr</th><th>Cross DD</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>{cmp_rows}</tbody>
</table>
</div>

<h2>Full Ranked Table</h2>
<div class="twrap">
<table>
  <thead><tr><th>Asset</th><th>BB</th><th>EMA</th><th>Entry</th><th>FTMO</th><th>Trades/yr</th><th>PF</th><th>Max DD</th><th>Return</th><th>Win Rate</th><th>Sharpe</th></tr></thead>
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

def main():
    n = len(ASSETS) * len(BB_LENS) * len(EMA_PERS) * len(ENTRIES)
    print('=' * 65)
    print('  Final Qualifier Sweep — 16 combos')
    print('=' * 65)
    print(f'\n  EURUSD + SP500  ·  BB 10/20  ·  EMA 100/200  ·  level + crossover\n')

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
                for entry in ENTRIES:
                    r = run_one(asset, df, bb, ema_p, entry)
                    if r:
                        results.append(r)
                        if r.get('ok'):
                            print(f'\n  *** FIRST QUALIFIER FOUND: {r["asset"]} BB({r["bb"]}) EMA({r["ema_p"]}) {r["entry"]} ***\n')

    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    generate_report(results)

    qualified = [r for r in results if r.get('ok')]
    print(f'\n{"="*65}')
    print(f'  QUALIFIED: {len(qualified)}/{len(results)}')
    for r in sorted(qualified, key=lambda x: x['pf'], reverse=True):
        print(f'  OK  {r["asset"]:7s} BB({r["bb"]}) EMA({r["ema_p"]}) {r["entry"]:10s}  '
              f'PF {r["pf"]:.3f}  {r["tpy"]:.0f}/yr  DD {r["dd"]:.1f}%  Ret {r["ret"]:+.1f}%')
    if not qualified:
        valid = [r for r in results if not r.get('error')]
        def gap(r):
            return (max(0, TARGETS['min_trades_yr'] - r['tpy']) / TARGETS['min_trades_yr'] +
                    max(0, TARGETS['min_pf']        - r['pf'])  / TARGETS['min_pf'] +
                    max(0, r['dd'] - TARGETS['max_dd_pct'])     / TARGETS['max_dd_pct'])
        closest = min(valid, key=gap, default=None)
        if closest:
            print(f'  Closest: {closest["asset"]} BB({closest["bb"]}) EMA({closest["ema_p"]}) {closest["entry"]}  '
                  f'PF {closest["pf"]}  {closest["tpy"]}/yr  DD {closest["dd"]}%')
    print('=' * 65)

if __name__ == '__main__':
    main()

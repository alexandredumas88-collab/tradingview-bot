#!/usr/bin/env python3
"""
Two-part test:
  A) BB Bandwidth filter on EURUSD — cull low-volatility noise entries.
     BB10/20 x EMA100/200 x bandwidth thresholds 0.004/0.006/0.008/0.010 = 16 combos
  B) Combined portfolio — EURUSD + SP500 trading simultaneously on pooled $100K equity.
     Several config pairs; report aggregated PF, trades/yr, max DD.
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

SYMBOLS = {'SP500': 'SPY', 'EURUSD': 'EURUSD=X'}
INITIAL_CAPITAL = 100_000
COMMISSION      = 0.0001
TRADE_SIZE      = 0.99
TARGETS         = dict(min_trades_yr=40, min_pf=2.0, max_dd_pct=8.0)

# Part A grid
A_BB_LENS    = [10, 20]
A_EMA_PERS   = [100, 200]
A_BW_THRESHS = [0.004, 0.006, 0.008, 0.010]   # (BB_upper - BB_lower) / BB_mid

# Part B portfolio pairs  — each is (EURUSD params, SP500 params)
# Using best standalone configs from prior runs
PORTFOLIO_PAIRS = [
    dict(label='EURUSD BB10 EMA100  +  SP500 BB20 EMA100',
         eu=dict(bb=10, ema_p=100, bw=0.0),
         sp=dict(bb=20, ema_p=100)),
    dict(label='EURUSD BB10 EMA200  +  SP500 BB20 EMA100',
         eu=dict(bb=10, ema_p=200, bw=0.0),
         sp=dict(bb=20, ema_p=100)),
    dict(label='EURUSD BB20 EMA200  +  SP500 BB10 EMA200',
         eu=dict(bb=20, ema_p=200, bw=0.0),
         sp=dict(bb=10, ema_p=200)),
]

_here       = os.path.dirname(__file__)
DATA_CACHE  = os.path.join(_here, 'pydata')
REPORT_OUT  = os.path.join(_here, 'pycombined-report.html')
RESULTS_OUT = os.path.join(_here, 'pycombined-results.json')
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

# ── Part A: EURUSD bandwidth-filtered strategy ────────────────────────────────

class BBRSIBandwidth(Strategy):
    """Level entry gated by EMA uptrend AND minimum BB bandwidth."""
    bb_period  = 10
    rsi_period = 7
    rsi_lo     = 35
    ema_period = 100
    bw_min     = 0.006    # (upper-lower)/mid must exceed this

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
        c  = self.data.Close[-1]
        bw = (self.bb_up[-1] - self.bb_lo[-1]) / self.bb_mid[-1] if self.bb_mid[-1] else 0
        if not self.position:
            if (c < self.bb_lo[-1]
                    and self.rsi[-1] < self.rsi_lo
                    and c > self.ema[-1]
                    and bw >= self.bw_min):
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()

# ── Part B: SP500 strategy (no bandwidth filter needed) ──────────────────────

class BBRSIFiltered(Strategy):
    """Level entry gated by EMA uptrend only."""
    bb_period  = 20
    rsi_period = 7
    rsi_lo     = 35
    ema_period = 100

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

# ── Shared stats extractor ────────────────────────────────────────────────────

def extract(stats, df, cash):
    def g(key, default=np.nan):
        try:    return stats[key]
        except: return default

    trades = int(g('# Trades', 0))
    pf_raw = g('Profit Factor', np.nan)
    if   isinstance(pf_raw, float) and np.isinf(pf_raw):  pf = 99.0
    elif isinstance(pf_raw, float) and np.isnan(pf_raw):  pf = 0.0
    else:                                                    pf = float(pf_raw)

    years  = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
    return dict(
        trades=trades, tpy=round(trades / years, 1),
        pf=round(pf, 4),
        wr=round(float(g('Win Rate [%]', np.nan)), 2),
        dd=round(abs(float(g('Max. Drawdown [%]', np.nan))), 2),
        ret=round(float(g('Return [%]', np.nan)), 2),
        sharpe=round(float(g('Sharpe Ratio', np.nan)), 3),
        eq_curve=stats._equity_curve['Equity'] if hasattr(stats, '_equity_curve') else None,
        trade_pnl=(stats._trades['PnL'].tolist()
                   if hasattr(stats, '_trades') and len(stats._trades) else []),
        years=years,
        cash=cash,
    )

# ── Part A runner ─────────────────────────────────────────────────────────────

def run_bw(df_eu, bb, ema_p, bw):
    BBRSIBandwidth.bb_period  = bb
    BBRSIBandwidth.ema_period = ema_p
    BBRSIBandwidth.bw_min     = bw
    try:
        bt    = Backtest(df_eu, BBRSIBandwidth, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()
        s = extract(stats, df_eu, INITIAL_CAPITAL)
        mt = s['tpy'] >= TARGETS['min_trades_yr']
        mp = s['pf']  >= TARGETS['min_pf']
        md = s['dd']  <  TARGETS['max_dd_pct']
        ok = mt and mp and md
        flags  = []
        if not mt: flags.append(f'{s["tpy"]:.0f}/yr')
        if not mp: flags.append(f'PF {s["pf"]:.2f}')
        if not md: flags.append(f'DD {s["dd"]:.1f}%')
        flag_s = ('  [' + ', '.join(flags) + ']') if flags else ''
        sym    = 'OK' if ok else 'X '
        print(f'  {sym}  EURUSD BB{bb:2d} EMA{ema_p:3d} BW{bw:.3f}   '
              f'{s["trades"]:4d}tr ({s["tpy"]:.0f}/yr)  PF {s["pf"]:.3f}  DD {s["dd"]:.1f}%  Ret {s["ret"]:+.1f}%{flag_s}')
        return dict(bb=bb, ema_p=ema_p, bw=bw, ok=ok, mt=mt, mp=mp, md=md, **{k: v for k, v in s.items() if k not in ('eq_curve','trade_pnl')})
    except Exception as e:
        print(f'  X   EURUSD BB{bb:2d} EMA{ema_p:3d} BW{bw:.3f}   ERROR: {e}')
        return dict(bb=bb, ema_p=ema_p, bw=bw, error=str(e), ok=False)

# ── Part B: combined portfolio maths ─────────────────────────────────────────

def combine_portfolio(s_eu, s_sp, label):
    """Merge two strategy results on pooled $100K equity (50/50 split)."""
    half = INITIAL_CAPITAL / 2

    # Scale PnL to the half-capital allocation
    scale_eu = half / s_eu['cash']
    scale_sp = half / s_sp['cash']

    all_pnl = ([p * scale_eu for p in s_eu['trade_pnl']] +
               [p * scale_sp for p in s_sp['trade_pnl']])
    gross_profit = sum(p for p in all_pnl if p > 0)
    gross_loss   = abs(sum(p for p in all_pnl if p < 0))
    pf = round(gross_profit / gross_loss, 4) if gross_loss > 0 else 99.0

    total_trades = s_eu['trades'] + s_sp['trades']
    years        = max(s_eu['years'], s_sp['years'])
    tpy          = round(total_trades / years, 1)

    # Combined equity curve
    eq_eu = s_eu['eq_curve']
    eq_sp = s_sp['eq_curve']
    max_dd, combined_ret = np.nan, np.nan
    if eq_eu is not None and eq_sp is not None:
        # Align to a common index
        idx     = eq_eu.index.union(eq_sp.index)
        eu_r    = eq_eu.reindex(idx, method='ffill')
        sp_r    = eq_sp.reindex(idx, method='ffill')
        # Scale each to half-capital
        eu_r    = eu_r * scale_eu
        sp_r    = sp_r * scale_sp
        combined = eu_r + sp_r
        peak     = combined.cummax()
        dd_pct   = (peak - combined) / peak * 100
        max_dd   = round(float(dd_pct.max()), 2)
        combined_ret = round(float((combined.iloc[-1] / combined.iloc[0] - 1) * 100), 2)

    wr = round(sum(1 for p in all_pnl if p > 0) / len(all_pnl) * 100, 1) if all_pnl else 0.0

    mt = tpy   >= TARGETS['min_trades_yr']
    mp = pf    >= TARGETS['min_pf']
    md = (not np.isnan(max_dd)) and max_dd < TARGETS['max_dd_pct']
    ok = mt and mp and md

    flags = []
    if not mt: flags.append(f'{tpy:.0f}/yr')
    if not mp: flags.append(f'PF {pf:.2f}')
    if not md: flags.append(f'DD {max_dd:.1f}%')
    flag_s = ('  [' + ', '.join(flags) + ']') if flags else ''
    sym    = 'OK' if ok else 'X '
    print(f'  {sym}  {label}')
    print(f'       {total_trades}tr ({tpy:.0f}/yr)  PF {pf:.3f}  DD {max_dd:.1f}%  Ret {combined_ret:+.1f}%  WR {wr:.0f}%{flag_s}')

    return dict(label=label, trades=total_trades, tpy=tpy, pf=pf, wr=wr,
                dd=max_dd, ret=combined_ret, ok=ok, mt=mt, mp=mp, md=md,
                eu_tpy=s_eu['tpy'], eu_pf=s_eu['pf'], eu_dd=s_eu['dd'],
                sp_tpy=s_sp['tpy'], sp_pf=s_sp['pf'], sp_dd=s_sp['dd'])

def run_portfolio(df_eu, df_sp, eu_cfg, sp_cfg, label):
    # EURUSD leg
    BBRSIBandwidth.bb_period  = eu_cfg['bb']
    BBRSIBandwidth.ema_period = eu_cfg['ema_p']
    BBRSIBandwidth.bw_min     = eu_cfg.get('bw', 0.0)
    bt_eu = Backtest(df_eu, BBRSIBandwidth, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
    s_eu  = extract(bt_eu.run(), df_eu, INITIAL_CAPITAL)

    # SP500 leg
    BBRSIFiltered.bb_period  = sp_cfg['bb']
    BBRSIFiltered.ema_period = sp_cfg['ema_p']
    bt_sp = Backtest(df_sp, BBRSIFiltered, cash=INITIAL_CAPITAL, commission=COMMISSION, exclusive_orders=True)
    s_sp  = extract(bt_sp.run(), df_sp, INITIAL_CAPITAL)

    return combine_portfolio(s_eu, s_sp, label)

# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(bw_results, port_results):
    def pf_c(v):  return '#4caf50' if v >= TARGETS['min_pf']        else '#f44336'
    def tpy_c(v): return '#4caf50' if v >= TARGETS['min_trades_yr'] else '#f44336'
    def dd_c(v):  return '#4caf50' if v <  TARGETS['max_dd_pct']    else '#f44336'
    def ret_c(v): return '#4caf50' if v >= 0                        else '#f44336'

    def badge(ok, yes, no):
        return f'<span class="badge {"green" if ok else "red"}">{yes if ok else no}</span>'

    all_qualified = [r for r in bw_results + port_results if r.get('ok')]
    COLORS = ['#f0b429','#4caf50','#2196f3','#9c27b0','#ff5722','#00bcd4',
              '#e91e63','#8bc34a','#ff9800','#3f51b5']

    # ── Part A: bandwidth heatmaps ────────────────────────────────────────────
    def bw_heatmap(bb, ema_p):
        lookup = {r['bw']: r for r in bw_results
                  if not r.get('error') and r['bb']==bb and r['ema_p']==ema_p}
        # baseline (bw=0 = no bandwidth filter, from pyema results)
        baselines = {
            (10,100): dict(tpy=51.0, pf=1.501, dd=1.5),
            (10,200): dict(tpy=64.0, pf=1.162, dd=1.8),
            (20,100): dict(tpy=31.0, pf=1.690, dd=1.4),
            (20,200): dict(tpy=43.0, pf=1.599, dd=1.6),
        }
        base = baselines.get((bb, ema_p), {})
        rows = ''
        # baseline row
        rows += (f'<tr><th class="row-th">No filter (baseline)</th>'
                 f'<td style="color:{pf_c(base.get("pf",0))}">{base.get("pf","?")} PF</td>'
                 f'<td style="color:{tpy_c(base.get("tpy",0))}">{base.get("tpy","?")} /yr</td>'
                 f'<td style="color:{dd_c(base.get("dd",99))}">{base.get("dd","?")}% DD</td>'
                 f'<td>—</td><td>—</td></tr>')
        for bw in A_BW_THRESHS:
            r = lookup.get(bw)
            if not r:
                rows += f'<tr><th class="row-th">BW &gt; {bw:.3f}</th><td colspan="5" class="na">—</td></tr>'
                continue
            cls    = 'ok-row' if r['ok'] else ''
            dpf    = r['pf'] - base.get('pf', r['pf'])
            dtpy   = r['tpy'] - base.get('tpy', r['tpy'])
            dpf_c  = '#4caf50' if dpf >= 0 else '#f44336'
            dtpy_c = '#f44336' if dtpy < 0 else '#4caf50'
            rows += (f'<tr class="{cls}">'
                     f'<th class="row-th">BW &gt; {bw:.3f}</th>'
                     f'<td style="color:{pf_c(r["pf"])}">{r["pf"]:.3f}</td>'
                     f'<td style="color:{tpy_c(r["tpy"])}">{r["tpy"]:.0f}/yr  ({r["trades"]} total)</td>'
                     f'<td style="color:{dd_c(r["dd"])}">{r["dd"]:.1f}%</td>'
                     f'<td style="color:{dpf_c}">{"+" if dpf>=0 else ""}{dpf:.3f} PF</td>'
                     f'<td style="color:{dtpy_c}">{"+" if dtpy>=0 else ""}{dtpy:.0f}/yr</td>'
                     f'</tr>')
        head = '<tr><th>Bandwidth min</th><th>PF</th><th>Trades/yr</th><th>Max DD</th><th>vs baseline PF</th><th>vs baseline /yr</th></tr>'
        return f'<table class="bw-tbl"><thead>{head}</thead><tbody>{rows}</tbody></table>'

    # ── Part A best cards ─────────────────────────────────────────────────────
    valid_bw = [r for r in bw_results if not r.get('error')]
    ranked_bw = sorted(valid_bw, key=lambda r: (r['ok'], r['pf']), reverse=True)
    bw_cards = ''
    for i, r in enumerate(ranked_bw[:8]):
        c      = COLORS[i % len(COLORS)]
        banner = '<div class="qual-banner">FTMO</div>' if r['ok'] else ''
        bw_cards += f'''<div class="card" style="border-top:3px solid {c}">
  {banner}
  <div class="rank" style="color:{c}">#{i+1}</div>
  <div class="cname">EURUSD</div>
  <div class="clabel">BB({r["bb"]}) · EMA({r["ema_p"]}) · BW&gt;{r["bw"]:.3f}</div>
  <div class="mgrid">
    <div class="ms"><div class="mv" style="color:{pf_c(r["pf"])}">{r["pf"]}</div><div class="ml">PF</div></div>
    <div class="ms"><div class="mv">{r["wr"]}%</div><div class="ml">Win Rate</div></div>
    <div class="ms"><div class="mv" style="color:{tpy_c(r["tpy"])}">{r["tpy"]}/yr</div><div class="ml">Freq</div></div>
    <div class="ms"><div class="mv">{r["trades"]}</div><div class="ml">Trades 2Y</div></div>
    <div class="ms"><div class="mv" style="color:{ret_c(r["ret"])}">{r["ret"]}%</div><div class="ml">Return</div></div>
    <div class="ms"><div class="mv" style="color:{dd_c(r["dd"])}">{r["dd"]}%</div><div class="ml">Max DD</div></div>
  </div>
  <div class="badges">
    {badge(r["mt"], "40+/yr",  f'{r["tpy"]}/yr')}
    {badge(r["mp"], "PF OK",   f'PF {r["pf"]}')}
    {badge(r["md"], "DD OK",   f'DD {r["dd"]}%')}
  </div>
</div>'''

    # ── Part B portfolio cards ────────────────────────────────────────────────
    port_cards = ''
    for i, r in enumerate(port_results):
        c      = COLORS[i % len(COLORS)]
        banner = '<div class="qual-banner">FTMO</div>' if r['ok'] else ''
        port_cards += f'''<div class="port-card" style="border-left:4px solid {c}">
  {banner}
  <div class="port-title" style="color:{c}">{r["label"]}</div>
  <div class="port-grid">
    <div class="ms"><div class="mv" style="color:{pf_c(r["pf"])}">{r["pf"]}</div><div class="ml">Combined PF</div></div>
    <div class="ms"><div class="mv" style="color:{tpy_c(r["tpy"])}">{r["tpy"]}/yr</div><div class="ml">Total Freq</div></div>
    <div class="ms"><div class="mv" style="color:{dd_c(r["dd"] if not np.isnan(r["dd"]) else 99)}">{r["dd"]}%</div><div class="ml">Combined DD</div></div>
    <div class="ms"><div class="mv">{r["trades"]}</div><div class="ml">Total Trades</div></div>
    <div class="ms"><div class="mv" style="color:{ret_c(r["ret"] if r["ret"] else 0)}">{r["ret"]}%</div><div class="ml">Combined Ret</div></div>
    <div class="ms"><div class="mv">{r["wr"]}%</div><div class="ml">Blended WR</div></div>
  </div>
  <div class="port-breakdown">
    <span>EURUSD: {r["eu_tpy"]}/yr · PF {r["eu_pf"]} · DD {r["eu_dd"]}%</span>
    <span>SP500: {r["sp_tpy"]}/yr · PF {r["sp_pf"]} · DD {r["sp_dd"]}%</span>
  </div>
  <div class="badges" style="margin-top:8px">
    {badge(r["mt"], "40+/yr",  f'{r["tpy"]}/yr')}
    {badge(r["mp"], "PF OK",   f'PF {r["pf"]}')}
    {badge(r["md"], "DD OK",   f'DD {r["dd"]}%')}
  </div>
</div>'''

    # ── Recommendation ────────────────────────────────────────────────────────
    if all_qualified:
        best = sorted(all_qualified, key=lambda r: r.get('pf', 0), reverse=True)[0]
        is_port = 'label' in best
        if is_port:
            rec = (f'<div class="rec good"><b>FTMO Qualifier — Combined Portfolio: {best["label"]}</b><br><br>'
                   f'PF <b>{best["pf"]}</b> &nbsp;·&nbsp; {best["tpy"]}/yr ({best["trades"]} trades) &nbsp;·&nbsp; '
                   f'Max DD <b>{best["dd"]}%</b> &nbsp;·&nbsp; Return {best["ret"]}% &nbsp;·&nbsp; Win Rate {best["wr"]}%<br><br>'
                   f'<b>Allocation:</b> $50,000 EURUSD leg + $50,000 SP500 leg on shared $100,000 equity base.</div>')
        else:
            rec = (f'<div class="rec good"><b>FTMO Qualifier — EURUSD BB({best["bb"]}) EMA({best["ema_p"]}) BW&gt;{best["bw"]:.3f}</b><br><br>'
                   f'PF <b>{best["pf"]}</b> &nbsp;·&nbsp; {best["tpy"]}/yr &nbsp;·&nbsp; Max DD <b>{best["dd"]}%</b> &nbsp;·&nbsp; '
                   f'Return {best["ret"]}% &nbsp;·&nbsp; Win Rate {best["wr"]}%</div>')
    else:
        all_valid = [r for r in bw_results + port_results if not r.get('error')]
        def gap(r):
            dd_v = r.get('dd', 99)
            if isinstance(dd_v, float) and np.isnan(dd_v): dd_v = 99
            return (max(0, TARGETS['min_trades_yr'] - r.get('tpy',0)) / TARGETS['min_trades_yr'] +
                    max(0, TARGETS['min_pf']        - r.get('pf', 0)) / TARGETS['min_pf'] +
                    max(0, dd_v - TARGETS['max_dd_pct'])               / TARGETS['max_dd_pct'])
        closest = min(all_valid, key=gap, default=None)
        lbl = (closest.get('label') or
               f'EURUSD BB({closest["bb"]}) EMA({closest["ema_p"]}) BW>{closest.get("bw",0):.3f}') if closest else '—'
        rec = (f'<div class="rec warn"><b>No combo qualifies yet.</b><br><br>'
               f'Closest: <b>{lbl}</b> — PF {closest.get("pf","?")} · {closest.get("tpy","?")} trd/yr · DD {closest.get("dd","?")}%'
               f'</div>')

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Combined Portfolio + Bandwidth Filter</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1500px;margin:0 auto}}
h1{{font-size:26px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:28px}}
h2{{font-size:17px;color:#fff;margin:32px 0 14px;border-left:3px solid #f0b429;padding-left:10px}}
h3{{font-size:14px;color:#aaa;margin:20px 0 10px}}
.sbar{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:120px}}
.sv{{display:block;font-size:20px;font-weight:bold;color:#f0b429}}
.sl{{display:block;font-size:11px;color:#666;margin-top:3px}}
/* cards */
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:28px}}
@media(max-width:1100px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;position:relative}}
.port-card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:16px;position:relative}}
.qual-banner{{position:absolute;top:0;right:0;background:#1a3a1a;color:#4caf50;font-size:10px;padding:3px 9px;border-radius:0 12px 0 6px;font-weight:bold;letter-spacing:1px}}
.rank{{font-size:26px;font-weight:bold;margin-bottom:2px}}
.cname{{font-size:15px;font-weight:bold;color:#fff}}
.clabel{{font-size:11px;color:#888;margin-bottom:10px}}
.port-title{{font-size:14px;font-weight:bold;margin-bottom:12px}}
.mgrid,.port-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px}}
.port-grid{{grid-template-columns:repeat(6,1fr)}}
@media(max-width:1000px){{.port-grid{{grid-template-columns:repeat(3,1fr)}}}}
.ms{{background:#111;border-radius:6px;padding:7px 8px;text-align:center}}
.mv{{font-size:13px;font-weight:bold}}
.ml{{font-size:9px;color:#555;margin-top:1px}}
.port-breakdown{{display:flex;gap:24px;font-size:11px;color:#888;margin-top:8px;flex-wrap:wrap}}
.badges{{display:flex;gap:5px;flex-wrap:wrap}}
.badge{{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:bold}}
.badge.green{{background:#1a3a1a;color:#4caf50}}
.badge.red{{background:#3a1a1a;color:#f44336}}
/* BW table */
.bw-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:20px;overflow-x:auto}}
table.bw-tbl{{width:100%;border-collapse:collapse;font-size:12px}}
table.bw-tbl th{{color:#888;padding:8px 12px;text-align:left;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
table.bw-tbl td{{padding:9px 12px;border-bottom:1px solid #161616}}
table.bw-tbl tr.ok-row td,table.bw-tbl tr.ok-row th{{background:#0e2e0e}}
table.bw-tbl .row-th{{color:#f0b429}}
table.bw-tbl tr:hover td,table.bw-tbl tr:hover th{{background:#1e1e1e}}
.na{{color:#444;text-align:center}}
.rec{{border-radius:10px;padding:20px 24px;margin-bottom:28px;font-size:14px;line-height:1.8}}
.rec.good{{background:#1a3a1a;border:1px solid #4caf50;color:#c8e6c9}}
.rec.warn{{background:#3a2a1a;border:1px solid #f0b429;color:#fff3cc}}
</style>
</head>
<body>
<h1>Combined Portfolio + Bandwidth Filter</h1>
<p class="sub">Part A: EURUSD bandwidth filter sweep (16 combos) · Part B: EURUSD+SP500 combined portfolio (3 pairs) · Generated {now}</p>

<div class="sbar">
  <div class="ss"><span class="sv">{len(bw_results)+len(port_results)}</span><span class="sl">Total Tests</span></div>
  <div class="ss"><span class="sv">{len(all_qualified)}</span><span class="sl">FTMO Qualified</span></div>
  <div class="ss"><span class="sv">40+/yr</span><span class="sl">Freq Target</span></div>
  <div class="ss"><span class="sv">PF &gt; 2.0</span><span class="sl">PF Target</span></div>
  <div class="ss"><span class="sv">DD &lt; 8%</span><span class="sl">DD Target</span></div>
</div>

<h2>Part A — EURUSD Bandwidth Filter: Best 8</h2>
<div class="cards">{bw_cards}</div>

<h2>Part A — Bandwidth vs Baseline (EURUSD, by BB/EMA config)</h2>
{"".join(f'<h3>BB({bb}) · EMA({ema_p})</h3><div class="bw-wrap">{bw_heatmap(bb, ema_p)}</div>' for bb in A_BB_LENS for ema_p in A_EMA_PERS)}

<h2>Part B — Combined Portfolio ($50K EURUSD + $50K SP500)</h2>
{port_cards}

<h2>Recommendation</h2>
{rec}
</body>
</html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] Report saved: {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_bw   = len(A_BB_LENS) * len(A_EMA_PERS) * len(A_BW_THRESHS)
    n_port = len(PORTFOLIO_PAIRS)
    print('=' * 65)
    print(f'  Part A: EURUSD bandwidth filter — {n_bw} combos')
    print(f'  Part B: Combined portfolio       — {n_port} pairs')
    print('=' * 65)

    df_eu = load_data('EURUSD')
    df_sp = load_data('SP500')
    for name, df in [('EURUSD', df_eu), ('SP500', df_sp)]:
        if df is not None:
            print(f'  {name}: {len(df)} bars  ({df.index[0].date()} to {df.index[-1].date()})')

    # ── Part A ────────────────────────────────────────────────────────────────
    print(f'\n--- Part A: Bandwidth sweep ({n_bw} combos) ---\n')
    bw_results = []
    for bb in A_BB_LENS:
        for ema_p in A_EMA_PERS:
            for bw in A_BW_THRESHS:
                r = run_bw(df_eu, bb, ema_p, bw)
                if r:
                    bw_results.append(r)

    # ── Part B ────────────────────────────────────────────────────────────────
    print(f'\n--- Part B: Combined portfolio ({n_port} pairs) ---\n')
    port_results = []
    for pair in PORTFOLIO_PAIRS:
        try:
            r = run_portfolio(df_eu, df_sp, pair['eu'], pair['sp'], pair['label'])
            port_results.append(r)
        except Exception as e:
            print(f'  X   {pair["label"]}  ERROR: {e}')
            port_results.append(dict(label=pair['label'], error=str(e), ok=False))

    all_results = dict(bandwidth=bw_results, portfolio=port_results)
    with open(RESULTS_OUT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    generate_report(bw_results, port_results)

    all_qualified = [r for r in bw_results + port_results if r.get('ok')]
    print(f'\n{"="*65}')
    print(f'  QUALIFIED: {len(all_qualified)}/{n_bw + n_port}')
    for r in sorted(all_qualified, key=lambda x: x.get('pf', 0), reverse=True):
        lbl = r.get('label') or f'EURUSD BB({r["bb"]}) EMA({r["ema_p"]}) BW>{r["bw"]:.3f}'
        print(f'  OK  {lbl}  PF {r["pf"]:.3f}  {r["tpy"]:.0f}/yr  DD {r["dd"]:.1f}%')
    if not all_qualified:
        all_valid = [r for r in bw_results + port_results if not r.get('error')]
        def gap(r):
            dd_v = r.get('dd', 99)
            if isinstance(dd_v, float) and np.isnan(dd_v): dd_v = 99
            return (max(0, TARGETS['min_trades_yr'] - r.get('tpy',0)) / TARGETS['min_trades_yr'] +
                    max(0, TARGETS['min_pf']        - r.get('pf', 0)) / TARGETS['min_pf'] +
                    max(0, dd_v - TARGETS['max_dd_pct'])               / TARGETS['max_dd_pct'])
        closest = min(all_valid, key=gap, default=None)
        if closest:
            lbl = (closest.get('label') or
                   f'EURUSD BB({closest["bb"]}) EMA({closest["ema_p"]}) BW>{closest.get("bw",0):.3f}')
            print(f'  Closest: {lbl}  PF {closest.get("pf","?")}  {closest.get("tpy","?")} trd/yr  DD {closest.get("dd","?")}%')
    print('=' * 65)

if __name__ == '__main__':
    main()

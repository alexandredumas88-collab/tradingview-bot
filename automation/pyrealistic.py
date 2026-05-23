#!/usr/bin/env python3
"""
Realistic analysis: EURUSD BB10 EMA100 + SP500 BB20 EMA100
Commission: 0.01%/order  |  Slippage: 1 pip EURUSD, 2 ticks SPY
Outputs: trade stats, monthly distribution, FTMO Phase 1/2 pass probability
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, math
import numpy as np
import pandas as pd
from datetime import datetime
from backtesting import Backtest, Strategy

# ── Parameters ────────────────────────────────────────────────────────────────
ACCOUNT    = 100_000
HALF       = ACCOUNT / 2
TRADE_SIZE = 0.99
BASE_COMM  = 0.0001      # 0.01% per order
EU_TICK    = 0.0001      # 1 pip EURUSD
SPY_TICK   = 0.01        # 1 minimum SPY tick
N_SIM      = 10_000

_here      = os.path.dirname(__file__)
DATA_DIR   = os.path.join(_here, 'pydata')
REPORT_OUT = os.path.join(_here, 'pyrealistic-report.html')

FTMO_PHASES = {
    'Phase 1': dict(target_pct=10.0, max_dd_pct=10.0, max_daily_pct=5.0, cal_days=30, min_tdays=10),
    'Phase 2': dict(target_pct=5.0,  max_dd_pct=10.0, max_daily_pct=5.0, cal_days=60, min_tdays=10),
}

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
    s = pd.Series(arr, dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

def _ema(arr, n):
    return pd.Series(arr, dtype=float).ewm(span=n, adjust=False).mean().values

# ── Strategies ────────────────────────────────────────────────────────────────
class BBRSIEurUsd(Strategy):
    def init(self):
        c = self.data.Close
        self.bb_up  = self.I(_bb_upper, c, 10)
        self.bb_mid = self.I(_bb_mid,   c, 10)
        self.bb_lo  = self.I(_bb_lower, c, 10)
        self.rsi    = self.I(_rsi,      c,  7)
        self.ema    = self.I(_ema,      c, 100)

    def next(self):
        if np.isnan(self.bb_lo[-1]) or np.isnan(self.rsi[-1]) or np.isnan(self.ema[-1]):
            return
        c = self.data.Close[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < 35 and c > self.ema[-1]:
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()


class BBRSISp500(Strategy):
    def init(self):
        c = self.data.Close
        self.bb_up  = self.I(_bb_upper, c, 20)
        self.bb_mid = self.I(_bb_mid,   c, 20)
        self.bb_lo  = self.I(_bb_lower, c, 20)
        self.rsi    = self.I(_rsi,      c,  7)
        self.ema    = self.I(_ema,      c, 100)

    def next(self):
        if np.isnan(self.bb_lo[-1]) or np.isnan(self.rsi[-1]) or np.isnan(self.ema[-1]):
            return
        c = self.data.Close[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < 35 and c > self.ema[-1]:
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()

# ── Data ──────────────────────────────────────────────────────────────────────
def load_df(name):
    path = os.path.join(DATA_DIR, f'{name}.csv')
    return pd.read_csv(path, index_col=0, parse_dates=True)

# ── Backtest runner ───────────────────────────────────────────────────────────
def run_bt(df, cls, commission):
    bt    = Backtest(df, cls, cash=ACCOUNT, commission=commission, exclusive_orders=True)
    stats = bt.run()
    trades = stats._trades.copy() if hasattr(stats, '_trades') and len(stats._trades) else pd.DataFrame()
    eq     = stats._equity_curve['Equity'].copy() if hasattr(stats, '_equity_curve') else None
    n      = int(stats.get('# Trades', 0))
    yrs    = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
    pf_raw = stats.get('Profit Factor', float('nan'))
    if isinstance(pf_raw, float) and math.isinf(pf_raw):  pf = 99.0
    elif isinstance(pf_raw, float) and math.isnan(pf_raw): pf = 0.0
    else: pf = float(pf_raw)
    return dict(
        n=n, tpy=round(n/yrs, 1), pf=round(pf, 3),
        wr=round(float(stats.get('Win Rate [%]', 0)), 1),
        dd=round(abs(float(stats.get('Max. Drawdown [%]', 0))), 2),
        ret=round(float(stats.get('Return [%]', 0)), 2),
        trades=trades, eq=eq, years=yrs,
    )

# ── Portfolio combination ─────────────────────────────────────────────────────
def combine(leg_eu, leg_sp):
    scale  = HALF / ACCOUNT
    pnl_eu = (leg_eu['trades']['PnL'] * scale).tolist() if not leg_eu['trades'].empty else []
    pnl_sp = (leg_sp['trades']['PnL'] * scale).tolist() if not leg_sp['trades'].empty else []
    all_pnl = pnl_eu + pnl_sp

    gp = sum(p for p in all_pnl if p > 0)
    gl = abs(sum(p for p in all_pnl if p < 0))
    pf = round(gp / gl, 3) if gl > 0 else 99.0
    n  = len(all_pnl)
    wr = round(sum(1 for p in all_pnl if p > 0) / n * 100, 1) if n else 0.0
    yrs = max(leg_eu['years'], leg_sp['years'])
    tpy = round((leg_eu['n'] + leg_sp['n']) / yrs, 1)

    eq_eu, eq_sp = leg_eu['eq'], leg_sp['eq']
    max_dd = combined_ret = float('nan')
    if eq_eu is not None and eq_sp is not None:
        idx      = eq_eu.index.union(eq_sp.index)
        combined = eq_eu.reindex(idx, method='ffill') * scale + eq_sp.reindex(idx, method='ffill') * scale
        peak     = combined.cummax()
        max_dd   = round(float(((peak - combined) / peak * 100).max()), 2)
        combined_ret = round(float((combined.iloc[-1] / combined.iloc[0] - 1) * 100), 2)

    avg = round(float(np.mean(all_pnl)), 2) if all_pnl else 0.0
    return dict(
        n=leg_eu['n'] + leg_sp['n'], tpy=tpy, pf=pf, wr=wr,
        dd=max_dd, ret=combined_ret, years=yrs,
        all_pnl=all_pnl,
        avg=avg,
        med=round(float(np.median(all_pnl)), 2) if all_pnl else 0.0,
        std=round(float(np.std(all_pnl)), 2) if all_pnl else 0.0,
        best=round(float(max(all_pnl)), 2) if all_pnl else 0.0,
        worst=round(float(min(all_pnl)), 2) if all_pnl else 0.0,
        monthly_exp=round(avg * tpy / 12, 2),
        annual_exp=round(avg * (leg_eu['n'] + leg_sp['n']) / yrs, 0),
    )

# ── Monthly distribution ──────────────────────────────────────────────────────
def monthly_dist(trades_eu, trades_sp):
    scale = HALF / ACCOUNT
    rows  = []
    for tdf, asset in [(trades_eu, 'EURUSD'), (trades_sp, 'SP500')]:
        if tdf.empty or 'ExitTime' not in tdf.columns:
            continue
        for _, r in tdf.iterrows():
            rows.append({'date': r['ExitTime'], 'pnl': r['PnL'] * scale, 'asset': asset})
    if not rows:
        return pd.DataFrame(columns=['month', 'pnl', 'n_trades', 'month_str'])
    df = pd.DataFrame(rows)
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    grp = df.groupby('month')['pnl'].agg(['sum', 'count']).reset_index()
    grp.columns = ['month', 'pnl', 'n_trades']
    grp['month_str'] = grp['month'].astype(str)
    return grp

# ── Monte Carlo FTMO ──────────────────────────────────────────────────────────
def monte_carlo(all_pnl, trades_eu, trades_sp, phase_key, seed=42):
    phase   = FTMO_PHASES[phase_key]
    rng     = np.random.default_rng(seed)
    pnl_arr = np.array(all_pnl, dtype=float)

    target_usd  = ACCOUNT * phase['target_pct']  / 100.0
    max_dd_usd  = ACCOUNT * phase['max_dd_pct']  / 100.0
    max_day_usd = ACCOUNT * phase['max_daily_pct'] / 100.0

    # Lambda: trades per business day from actual data
    all_dates = []
    for tdf in [trades_eu, trades_sp]:
        if not tdf.empty and 'ExitTime' in tdf.columns:
            all_dates += pd.to_datetime(tdf['ExitTime']).tolist()
    if all_dates:
        bdays = pd.bdate_range(min(all_dates), max(all_dates))
        lam   = len(all_pnl) / max(len(bdays), 1)
    else:
        lam = 0.25

    tdays  = round(phase['cal_days'] * 252 / 365)
    min_td = phase['min_tdays']

    passed = brk_dd = brk_day = short_td = 0

    for _ in range(N_SIM):
        equity   = float(ACCOUNT)
        cum      = 0.0
        active_d = 0
        broke    = False

        for _ in range(tdays):
            n_today = int(rng.poisson(lam))
            if n_today > 0:
                active_d += 1
            day_pnl = 0.0
            for __ in range(n_today):
                p = float(pnl_arr[rng.integers(len(pnl_arr))])
                day_pnl += p
                cum     += p
                equity  += p

            # Daily loss breach (5% of initial)
            if day_pnl < -max_day_usd:
                broke = True; brk_day += 1; break
            # Max drawdown breach (10% of initial, absolute)
            if ACCOUNT - equity > max_dd_usd:
                broke = True; brk_dd += 1; break

        if broke:
            continue
        if cum >= target_usd:
            if active_d >= min_td:
                passed += 1
            else:
                short_td += 1

    exp_tdays = round((1 - math.exp(-lam)) * tdays, 1)

    return dict(
        phase=phase_key,
        pass_rate=round(passed   / N_SIM * 100, 1),
        breach_dd=round(brk_dd   / N_SIM * 100, 1),
        breach_daily=round(brk_day  / N_SIM * 100, 1),
        min_day_fail=round(short_td / N_SIM * 100, 1),
        uncond_pass=round((passed + short_td) / N_SIM * 100, 1),
        lam=round(lam, 4),
        tdays=tdays,
        min_td=min_td,
        exp_trades=round(lam * tdays, 1),
        exp_tdays=exp_tdays,
        target_usd=target_usd,
        max_dd_usd=max_dd_usd,
        max_day_usd=max_day_usd,
    )

# ── HTML helpers ──────────────────────────────────────────────────────────────
def pf_c(v):  return '#4caf50' if v >= 2.0 else ('#f0b429' if v >= 1.5 else '#f44336')
def dd_c(v):  return '#4caf50' if v < 4.0  else ('#f0b429' if v < 8.0  else '#f44336')
def tpy_c(v): return '#4caf50' if v >= 40  else '#f44336'

def ftmo_box(mc):
    pr    = mc['pass_rate']
    unc   = mc['uncond_pass']
    bdd   = mc['breach_dd']
    bday  = mc['breach_daily']
    mdf   = mc['min_day_fail']
    pc    = '#4caf50' if pr >= 50 else ('#f0b429' if pr >= 20 else '#f44336')

    alert = ''
    if mc['exp_tdays'] < mc['min_td']:
        alert = (f'<div class="alert-box"><b>Critical — Minimum Trading Days:</b> '
                 f'This strategy is expected to trade on only <b>{mc["exp_tdays"]:.1f}</b> distinct days '
                 f'in the {mc["tdays"]}-day period, well below FTMO\'s required <b>{mc["min_td"]}</b>. '
                 f'Even when the profit target is hit ({mc["uncond_pass"]}% of sims), the account fails '
                 f'the min-days rule {mc["min_day_fail"]}% of the time. '
                 f'<b>Action required:</b> add more liquid assets or reduce EMA period to generate more signals.</div>')

    def bar(pct, col):
        w = min(pct, 100)
        return (f'<div class="gauge-bar-wrap"><div class="gauge-bar" '
                f'style="width:{w}%;background:{col}"></div></div>')

    is_p1  = 'Phase 1' in mc['phase']
    target = '$10,000 (+10%)' if is_p1 else '$5,000 (+5%)'
    period = '30 calendar days' if is_p1 else '60 calendar days'

    return f'''<div class="ftmo-box">
  <div class="ftmo-title">{mc["phase"]} &mdash; Target {target}</div>
  <div class="stat-row"><span>Period</span><span>{period} (~{mc["tdays"]} trading days)</span></div>
  <div class="stat-row"><span>Profit target</span><span>${mc["target_usd"]:,.0f}</span></div>
  <div class="stat-row"><span>Max drawdown limit</span><span>${mc["max_dd_usd"]:,.0f}</span></div>
  <div class="stat-row"><span>Max daily loss limit</span><span>${mc["max_day_usd"]:,.0f}</span></div>
  <div class="stat-row"><span>Avg trades/day (lambda)</span><span>{mc["lam"]:.4f}</span></div>
  <div class="stat-row"><span>Expected trades in period</span><span>{mc["exp_trades"]:.1f}</span></div>
  <div class="stat-row"><span>Expected active days</span><span>{mc["exp_tdays"]:.1f} (need {mc["min_td"]})</span></div>
  <hr style="border-color:#2a2a2a;margin:12px 0">
  <div class="gauge-row">
    <span class="gauge-label">Full pass (all rules)</span>
    {bar(pr, pc)}
    <span class="gauge-pct" style="color:{pc}">{pr}%</span>
  </div>
  <div class="gauge-row">
    <span class="gauge-label">Pass (ignore min-days)</span>
    {bar(unc, '#2196f3')}
    <span class="gauge-pct" style="color:#2196f3">{unc}%</span>
  </div>
  <div class="gauge-row">
    <span class="gauge-label">Breach — max drawdown</span>
    {bar(bdd, '#f44336')}
    <span class="gauge-pct" style="color:#f44336">{bdd}%</span>
  </div>
  <div class="gauge-row">
    <span class="gauge-label">Breach — daily loss</span>
    {bar(bday, '#f44336')}
    <span class="gauge-pct" style="color:#f44336">{bday}%</span>
  </div>
  <div class="gauge-row">
    <span class="gauge-label">Failed min-days rule</span>
    {bar(mdf, '#ff9800')}
    <span class="gauge-pct" style="color:#ff9800">{mdf}%</span>
  </div>
  {alert}
</div>'''

# ── HTML Report ───────────────────────────────────────────────────────────────
def generate_report(gross_eu, gross_sp, real_eu, real_sp,
                    port_gross, port_real, monthly, mc1, mc2,
                    eu_comm_total, sp_comm_total, eu_avg_p, sp_avg_p):

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Chart data
    months_js  = json.dumps(monthly['month_str'].tolist() if not monthly.empty else [])
    pnls_js    = json.dumps([round(p, 2) for p in monthly['pnl'].tolist()] if not monthly.empty else [])
    mcolors_js = ('[' +
                  ','.join("'#4caf50'" if p >= 0 else "'#f44336'"
                           for p in monthly['pnl'].tolist()) +
                  ']') if not monthly.empty else '[]'

    pnl_arr = np.array(port_real['all_pnl'])
    if len(pnl_arr):
        hist, edges = np.histogram(pnl_arr, bins=20)
        hlabels = json.dumps([f'{edges[i]:.0f}' for i in range(len(hist))])
        hvals   = json.dumps(hist.tolist())
        hcolors = ('[' +
                   ','.join("'#4caf50'" if (edges[i]+edges[i+1])/2 >= 0 else "'#f44336'"
                            for i in range(len(hist))) +
                   ']')
    else:
        hlabels = hvals = '[]'; hcolors = '[]'

    ret_color = '#4caf50' if port_real['ret'] > 0 else '#f44336'
    avg_color = '#4caf50' if port_real['avg'] > 0 else '#f44336'

    eu_rt_pips   = round(eu_comm_total * 2 * 10000, 2)
    sp_rt_bps    = round(sp_comm_total * 2 * 10000, 3)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Realistic Portfolio Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:23px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:24px}}
h2{{font-size:15px;color:#fff;margin:26px 0 12px;border-left:3px solid #f0b429;padding-left:10px}}
.sbar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:13px 16px;text-align:center;flex:1;min-width:100px}}
.sv{{display:block;font-size:19px;font-weight:bold}}
.sl{{display:block;font-size:11px;color:#666;margin-top:3px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
.box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px}}
.box h3{{font-size:12px;color:#888;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#666;padding:7px 10px;text-align:left;border-bottom:1px solid #2a2a2a}}
td{{padding:7px 10px;border-bottom:1px solid #161616}}
tr:hover td{{background:#1e1e1e}}
.good{{color:#4caf50}} .warn{{color:#f0b429}} .bad{{color:#f44336}}
.chart-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:18px;margin-bottom:14px}}
.chart-wrap h3{{font-size:12px;color:#888;margin-bottom:12px}}
canvas{{max-height:240px}}
.ftmo-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}}
@media(max-width:900px){{.ftmo-grid{{grid-template-columns:1fr}}}}
.ftmo-box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:18px}}
.ftmo-title{{font-size:14px;font-weight:bold;color:#f0b429;margin-bottom:12px}}
.gauge-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.gauge-label{{font-size:11px;color:#888;width:170px;flex-shrink:0}}
.gauge-bar-wrap{{flex:1;background:#111;border-radius:4px;height:13px;overflow:hidden}}
.gauge-bar{{height:100%;border-radius:4px}}
.gauge-pct{{font-size:12px;font-weight:bold;width:48px;text-align:right}}
.stat-row{{display:flex;justify-content:space-between;font-size:12px;padding:5px 0;border-bottom:1px solid #161616;color:#aaa}}
.stat-row span:last-child{{color:#e0e0e0}}
.stat-row:last-child{{border-bottom:none}}
.alert-box{{background:#2a0e0e;border:1px solid #f44336;border-radius:8px;padding:12px 14px;margin-top:12px;font-size:12px;line-height:1.7;color:#ffcdd2}}
.info-box{{background:#0e1a2a;border:1px solid #2196f3;border-radius:8px;padding:12px 16px;margin-top:14px;font-size:12px;line-height:1.7;color:#bbdefb}}
</style>
</head>
<body>
<h1>Realistic Portfolio — EURUSD BB(10) EMA(100) + SP500 BB(20) EMA(100)</h1>
<p class="sub">Commission 0.01%/order &nbsp;|&nbsp; Slippage 1 pip EURUSD / 2 ticks SPY &nbsp;|&nbsp; $100K account, 50/50 split &nbsp;|&nbsp; Generated {now}</p>

<h2>Combined Portfolio Overview (Realistic Costs)</h2>
<div class="sbar">
  <div class="ss"><span class="sv" style="color:{pf_c(port_real['pf'])}">{port_real['pf']}</span><span class="sl">Combined PF</span></div>
  <div class="ss"><span class="sv" style="color:{tpy_c(port_real['tpy'])}">{port_real['tpy']}/yr</span><span class="sl">Trades/Year</span></div>
  <div class="ss"><span class="sv" style="color:{dd_c(port_real['dd'])}">{port_real['dd']}%</span><span class="sl">Max DD</span></div>
  <div class="ss"><span class="sv">{port_real['wr']}%</span><span class="sl">Win Rate</span></div>
  <div class="ss"><span class="sv" style="color:{ret_color}">{port_real['ret']}%</span><span class="sl">Return 2Y</span></div>
  <div class="ss"><span class="sv">{port_real['n']}</span><span class="sl">Total Trades</span></div>
</div>

<h2>Cost Impact: Gross vs Realistic</h2>
<div class="box" style="margin-bottom:14px">
  <h3>Per-Leg Performance Before and After Transaction Costs</h3>
  <table>
    <thead><tr>
      <th>Asset</th><th>Scenario</th><th>Cost/Order</th>
      <th>Trades/yr</th><th>PF</th><th>Win Rate</th><th>Max DD</th><th>Return 2Y</th>
    </tr></thead>
    <tbody>
      <tr>
        <td>EURUSD</td><td>Gross (no cost)</td><td>—</td>
        <td>{gross_eu['tpy']}</td>
        <td style="color:{pf_c(gross_eu['pf'])}">{gross_eu['pf']}</td>
        <td>{gross_eu['wr']}%</td>
        <td style="color:{dd_c(gross_eu['dd'])}">{gross_eu['dd']}%</td>
        <td>{gross_eu['ret']}%</td>
      </tr>
      <tr style="background:#131313">
        <td>EURUSD</td><td><b>Realistic</b></td>
        <td class="bad">{eu_comm_total*10000:.2f} pips</td>
        <td>{real_eu['tpy']}</td>
        <td style="color:{pf_c(real_eu['pf'])}">{real_eu['pf']}</td>
        <td>{real_eu['wr']}%</td>
        <td style="color:{dd_c(real_eu['dd'])}">{real_eu['dd']}%</td>
        <td>{real_eu['ret']}%</td>
      </tr>
      <tr>
        <td>SP500 (SPY)</td><td>Gross (no cost)</td><td>—</td>
        <td>{gross_sp['tpy']}</td>
        <td style="color:{pf_c(gross_sp['pf'])}">{gross_sp['pf']}</td>
        <td>{gross_sp['wr']}%</td>
        <td style="color:{dd_c(gross_sp['dd'])}">{gross_sp['dd']}%</td>
        <td>{gross_sp['ret']}%</td>
      </tr>
      <tr style="background:#131313">
        <td>SP500 (SPY)</td><td><b>Realistic</b></td>
        <td class="bad">{sp_comm_total*10000:.2f} bps</td>
        <td>{real_sp['tpy']}</td>
        <td style="color:{pf_c(real_sp['pf'])}">{real_sp['pf']}</td>
        <td>{real_sp['wr']}%</td>
        <td style="color:{dd_c(real_sp['dd'])}">{real_sp['dd']}%</td>
        <td>{real_sp['ret']}%</td>
      </tr>
      <tr style="background:#0d1a0d;font-weight:bold">
        <td colspan="2">Combined (realistic, $50K/leg)</td>
        <td>—</td>
        <td style="color:{tpy_c(port_real['tpy'])}">{port_real['tpy']}</td>
        <td style="color:{pf_c(port_real['pf'])}">{port_real['pf']}</td>
        <td>{port_real['wr']}%</td>
        <td style="color:{dd_c(port_real['dd'])}">{port_real['dd']}%</td>
        <td>{port_real['ret']}%</td>
      </tr>
    </tbody>
  </table>
</div>

<h2>Per-Trade Statistics (Scaled to $100K Account, 50/50 Split)</h2>
<div class="grid2">
  <div class="box">
    <h3>P&L Summary</h3>
    <table>
      <tr><td>Average profit / trade</td><td style="color:{avg_color}"><b>${port_real['avg']:,.2f}</b></td></tr>
      <tr><td>Median profit / trade</td><td>${port_real['med']:,.2f}</td></tr>
      <tr><td>Std deviation</td><td>${port_real['std']:,.2f}</td></tr>
      <tr><td>Best single trade</td><td class="good">${port_real['best']:,.2f}</td></tr>
      <tr><td>Worst single trade</td><td class="bad">${port_real['worst']:,.2f}</td></tr>
      <tr><td>Monthly expectancy</td><td><b>${port_real['monthly_exp']:,.2f}</b></td></tr>
      <tr><td>Annual expectancy</td><td><b>${port_real['annual_exp']:,.0f}</b></td></tr>
    </table>
  </div>
  <div class="box">
    <h3>Cost Breakdown (Per Round-Trip)</h3>
    <table>
      <tr><td>EURUSD avg price</td><td>{eu_avg_p:.5f}</td></tr>
      <tr><td>Commission</td><td>1.00 pip/order</td></tr>
      <tr><td>Slippage</td><td>1.00 pip/order</td></tr>
      <tr><td>Total EURUSD round-trip</td><td class="bad"><b>{eu_rt_pips:.1f} pips</b></td></tr>
      <tr><td>SPY avg price</td><td>${sp_avg_p:.2f}</td></tr>
      <tr><td>Commission</td><td>0.01%/order (~${sp_avg_p*BASE_COMM*100:.2f} per $100)</td></tr>
      <tr><td>Slippage</td><td>$0.02/share (2 ticks)</td></tr>
      <tr><td>Total SPY round-trip</td><td class="bad"><b>{sp_rt_bps:.1f} bps</b></td></tr>
    </table>
  </div>
</div>

<div class="chart-wrap">
  <h3>Monthly Profit Distribution — Combined Portfolio (scaled to $100K)</h3>
  <canvas id="monthlyChart"></canvas>
</div>

<div class="chart-wrap">
  <h3>Trade P&L Distribution — Histogram (all {port_real['n']} trades)</h3>
  <canvas id="histChart"></canvas>
</div>

<h2>FTMO Pass Probability — Monte Carlo ({N_SIM:,} simulations each)</h2>
<div class="ftmo-grid">
  {ftmo_box(mc1)}
  {ftmo_box(mc2)}
</div>

<div class="info-box">
  <b>Model assumptions:</b> Trades arrive as a Poisson process (lambda = {mc1['lam']:.4f}/trading day = {mc1['lam']*252:.0f}/year).
  PnL sampled with replacement from empirical distribution ({port_real['n']} trades, avg ${port_real['avg']:,.2f}, std ${port_real['std']:,.2f}).
  Daily loss checked on realized PnL only; unrealized intraday exposure not modeled (actual daily risk is higher for open positions held overnight).
  Drawdown limit: account equity cannot fall more than ${mc1['max_dd_usd']:,.0f} below initial balance.
</div>

<script>
const mCtx = document.getElementById('monthlyChart').getContext('2d');
new Chart(mCtx, {{
  type: 'bar',
  data: {{
    labels: {months_js},
    datasets: [{{
      data: {pnls_js},
      backgroundColor: {mcolors_js},
      borderRadius: 4
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: c => '$' + c.raw.toLocaleString('en', {{minimumFractionDigits: 2}}) }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#888', maxRotation: 45 }}, grid: {{ color: '#1a1a1a' }} }},
      y: {{ ticks: {{ color: '#888', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#222' }} }}
    }}
  }}
}});

const hCtx = document.getElementById('histChart').getContext('2d');
new Chart(hCtx, {{
  type: 'bar',
  data: {{
    labels: {hlabels},
    datasets: [{{
      data: {hvals},
      backgroundColor: {hcolors},
      borderRadius: 2
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#888', callback: (v, i, ticks) => '$' + hCtx.chart.data.labels[i] }}, grid: {{ color: '#1a1a1a' }} }},
      y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }}, title: {{ display: true, text: 'Trade Count', color: '#666' }} }}
    }}
  }}
}});
</script>
</body>
</html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] Report saved: {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading cached data...')
    df_eu = load_df('EURUSD')
    df_sp = load_df('SP500')
    print(f'  EURUSD: {len(df_eu)} bars  ({df_eu.index[0].date()} to {df_eu.index[-1].date()})')
    print(f'  SP500:  {len(df_sp)} bars  ({df_sp.index[0].date()} to {df_sp.index[-1].date()})')

    # Slippage fraction = (tick_size * n_ticks) / avg_price
    eu_avg_p   = float(df_eu['Close'].mean())
    sp_avg_p   = float(df_sp['Close'].mean())
    eu_slip    = EU_TICK  * 1 / eu_avg_p
    sp_slip    = SPY_TICK * 2 / sp_avg_p
    eu_comm    = BASE_COMM + eu_slip
    sp_comm    = BASE_COMM + sp_slip

    print(f'\nTransaction costs:')
    print(f'  EURUSD avg price {eu_avg_p:.5f}  |  slip {eu_slip*10000:.2f} pips  |  total {eu_comm*10000:.2f} pips/order')
    print(f'  SPY    avg price ${sp_avg_p:.2f}  |  slip {sp_slip*10000:.4f} bps  |  total {sp_comm*10000:.4f} bps/order')

    print('\nRunning backtests (4 runs)...')
    gross_eu = run_bt(df_eu, BBRSIEurUsd, 0.0)
    gross_sp = run_bt(df_sp, BBRSISp500,  0.0)
    real_eu  = run_bt(df_eu, BBRSIEurUsd, eu_comm)
    real_sp  = run_bt(df_sp, BBRSISp500,  sp_comm)

    port_gross = combine(gross_eu, gross_sp)
    port_real  = combine(real_eu,  real_sp)

    monthly = monthly_dist(real_eu['trades'], real_sp['trades'])

    print(f'\nMonte Carlo ({N_SIM:,} simulations)...')
    mc1 = monte_carlo(port_real['all_pnl'], real_eu['trades'], real_sp['trades'], 'Phase 1', seed=42)
    mc2 = monte_carlo(port_real['all_pnl'], real_eu['trades'], real_sp['trades'], 'Phase 2', seed=43)

    # ── Console output ────────────────────────────────────────────────────────
    sep = '=' * 62
    print(f'\n{sep}')
    print(f'  REALISTIC: EURUSD BB10 EMA100  +  SP500 BB20 EMA100')
    print(f'  Commission: 0.01%/order  |  Slippage: 1 pip EU / 2 tick SP')
    print(f'{sep}')

    def delta(a, b, label, higher_is_better=True):
        d = b - a
        arrow = ('up' if d > 0 else 'dn') if higher_is_better else ('dn' if d > 0 else 'up')
        sign  = '+' if d > 0 else ''
        tag   = 'OK' if (d >= 0) == higher_is_better else 'X '
        return f'  {tag}  {label}: {a:.3f} -> {b:.3f}  ({sign}{d:.3f})'

    print('\n  Cost impact (gross -> realistic):')
    print(delta(gross_eu['pf'], real_eu['pf'], 'EURUSD PF'))
    print(delta(gross_sp['pf'], real_sp['pf'], 'SP500  PF'))
    print(delta(port_gross['pf'], port_real['pf'], 'Combined PF'))

    print(f'\n  Combined portfolio (realistic):')
    print(f'    Trades:   {port_real["n"]} ({port_real["tpy"]}/yr)  PF: {port_real["pf"]}  WR: {port_real["wr"]}%  DD: {port_real["dd"]}%')

    print(f'\n  Per-trade stats (scaled to $100K, 50/50 split):')
    print(f'    Avg profit:      ${port_real["avg"]:>8,.2f}')
    print(f'    Median:          ${port_real["med"]:>8,.2f}')
    print(f'    Std deviation:   ${port_real["std"]:>8,.2f}')
    print(f'    Best trade:      ${port_real["best"]:>8,.2f}')
    print(f'    Worst trade:     ${port_real["worst"]:>8,.2f}')
    print(f'    Monthly expect.: ${port_real["monthly_exp"]:>8,.2f}')
    print(f'    Annual expect.:  ${port_real["annual_exp"]:>8,.0f}')

    if not monthly.empty:
        print(f'\n  Monthly profit distribution:')
        for _, row in monthly.iterrows():
            bar  = '#' * max(1, int(abs(row['pnl']) / 30))
            sign = '+' if row['pnl'] >= 0 else '-'
            flag = ' <-- negative' if row['pnl'] < 0 else ''
            print(f'    {row["month_str"]}  {sign}${abs(row["pnl"]):>8,.2f}  ({int(row["n_trades"])} trades)  {bar}{flag}')

    print(f'\n  FTMO Phase 1  (30 days, +$10,000 target):')
    print(f'    Expected trades:     {mc1["exp_trades"]:.1f}')
    print(f'    Expected trade days: {mc1["exp_tdays"]:.1f}  (need {mc1["min_td"]})')
    print(f'    Pass rate (all):     {mc1["pass_rate"]}%')
    print(f'    Pass (no min-days):  {mc1["uncond_pass"]}%')
    print(f'    Breach (DD):         {mc1["breach_dd"]}%')
    print(f'    Breach (daily):      {mc1["breach_daily"]}%')
    print(f'    Failed min-days:     {mc1["min_day_fail"]}%')

    print(f'\n  FTMO Phase 2  (60 days, +$5,000 target):')
    print(f'    Expected trades:     {mc2["exp_trades"]:.1f}')
    print(f'    Expected trade days: {mc2["exp_tdays"]:.1f}  (need {mc2["min_td"]})')
    print(f'    Pass rate (all):     {mc2["pass_rate"]}%')
    print(f'    Pass (no min-days):  {mc2["uncond_pass"]}%')
    print(f'    Breach (DD):         {mc2["breach_dd"]}%')
    print(f'    Breach (daily):      {mc2["breach_daily"]}%')
    print(f'    Failed min-days:     {mc2["min_day_fail"]}%')

    print(f'\n{sep}')

    generate_report(gross_eu, gross_sp, real_eu, real_sp,
                    port_gross, port_real, monthly, mc1, mc2,
                    eu_comm, sp_comm, eu_avg_p, sp_avg_p)

if __name__ == '__main__':
    main()

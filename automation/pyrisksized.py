#!/usr/bin/env python3
"""
Risk-based position sizing: EURUSD BB10 EMA100 + SP500 BB20 EMA100
ATR(14) stop-loss at 2x ATR | 1% and 2% risk per trade
Leverage: 50:1 EURUSD, 10:1 SPY | Commission + slippage
$100k account, 50/50 split, $50k per leg
Outputs: avg $/trade, monthly expectancy, time to FTMO P1/P2 targets
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, math
import numpy as np
import pandas as pd
from datetime import datetime
from backtesting import Backtest, Strategy

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT      = 100_000
LEG_CAPITAL  = ACCOUNT / 2         # $50k per leg — actual allocation
BASE_COMM    = 0.0001
EU_TICK      = 0.0001              # 1 pip EURUSD
SPY_TICK     = 0.01                # 1 tick SPY
ATR_PERIOD   = 14
ATR_MULT     = 2.0                 # stop = entry − 2×ATR
EU_LEVERAGE  = 50
SP_LEVERAGE  = 10
N_SIM        = 10_000

_here      = os.path.dirname(__file__)
DATA_DIR   = os.path.join(_here, 'pydata')
REPORT_OUT = os.path.join(_here, 'pyrisksized-report.html')

SCENARIOS = [
    dict(label='1% risk / trade', risk_pct=0.01, risk_dollars=LEG_CAPITAL * 0.01),
    dict(label='2% risk / trade', risk_pct=0.02, risk_dollars=LEG_CAPITAL * 0.02),
]
FTMO = dict(
    p1=dict(name='Phase 1', target=10_000, max_dd=10_000, max_daily=5_000, cal_days=30, tdays=22, min_td=10),
    p2=dict(name='Phase 2', target=5_000,  max_dd=10_000, max_daily=5_000, cal_days=60, tdays=43, min_td=10),
)

# ── Indicators ────────────────────────────────────────────────────────────────
def _bb_upper(arr, n):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() + 2 * s.rolling(n, min_periods=n).std(ddof=0)).values

def _bb_mid(arr, n):
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().values

def _bb_lower(arr, n):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() - 2 * s.rolling(n, min_periods=n).std(ddof=0)).values

def _rsi(arr, n):
    s     = pd.Series(arr, dtype=float)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

def _ema(arr, n):
    return pd.Series(arr, dtype=float).ewm(span=n, adjust=False).mean().values

def _atr(high, low, close, n):
    h, l, c = (pd.Series(x, dtype=float) for x in (high, low, close))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().values

# ── Strategies ────────────────────────────────────────────────────────────────
class BBRSIRiskEU(Strategy):
    """EURUSD: BB(10) + RSI(7)<35 + EMA(100) + ATR stop + risk-based size."""
    risk_pct    = 0.01
    max_lev     = EU_LEVERAGE

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.bb_up  = self.I(_bb_upper, c, 10)
        self.bb_mid = self.I(_bb_mid,   c, 10)
        self.bb_lo  = self.I(_bb_lower, c, 10)
        self.rsi    = self.I(_rsi,      c,  7)
        self.ema    = self.I(_ema,      c, 100)
        self.atr    = self.I(_atr,      h, l, c, ATR_PERIOD)

    def next(self):
        if any(np.isnan(x[-1]) for x in (self.bb_lo, self.rsi, self.ema, self.atr)):
            return
        c   = self.data.Close[-1]
        atr = self.atr[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < 35 and c > self.ema[-1]:
                stop      = c - ATR_MULT * atr
                risk_dist = max(c - stop, c * 0.0001)   # floor at 1 pip equivalent
                risk_usd  = self.equity * self.risk_pct
                size      = int(min(risk_usd / risk_dist,
                                    self.equity * self.max_lev / c))
                if size >= 1:
                    self.buy(size=size, sl=stop)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()


class BBRSIRiskSP(Strategy):
    """SP500 (SPY): BB(20) + RSI(7)<35 + EMA(100) + ATR stop + risk-based size."""
    risk_pct    = 0.01
    max_lev     = SP_LEVERAGE

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.bb_up  = self.I(_bb_upper, c, 20)
        self.bb_mid = self.I(_bb_mid,   c, 20)
        self.bb_lo  = self.I(_bb_lower, c, 20)
        self.rsi    = self.I(_rsi,      c,  7)
        self.ema    = self.I(_ema,      c, 100)
        self.atr    = self.I(_atr,      h, l, c, ATR_PERIOD)

    def next(self):
        if any(np.isnan(x[-1]) for x in (self.bb_lo, self.rsi, self.ema, self.atr)):
            return
        c   = self.data.Close[-1]
        atr = self.atr[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < 35 and c > self.ema[-1]:
                stop      = c - ATR_MULT * atr
                risk_dist = max(c - stop, c * 0.001)
                risk_usd  = self.equity * self.risk_pct
                size      = int(min(risk_usd / risk_dist,
                                    self.equity * self.max_lev / c))
                if size >= 1:
                    self.buy(size=size, sl=stop)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()

# ── Data ──────────────────────────────────────────────────────────────────────
def load_df(name):
    return pd.read_csv(os.path.join(DATA_DIR, f'{name}.csv'), index_col=0, parse_dates=True)

# ── Backtest runner ───────────────────────────────────────────────────────────
def run_bt(df, cls, commission, margin, risk_pct):
    cls.risk_pct = risk_pct
    bt    = Backtest(df, cls, cash=LEG_CAPITAL, commission=commission,
                     margin=margin, exclusive_orders=True)
    stats = bt.run()
    trades = stats._trades.copy() if hasattr(stats, '_trades') and len(stats._trades) else pd.DataFrame()
    eq     = stats._equity_curve['Equity'].copy() if hasattr(stats, '_equity_curve') else None
    n      = int(stats.get('# Trades', 0))
    yrs    = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
    pf_raw = stats.get('Profit Factor', float('nan'))
    if isinstance(pf_raw, float) and math.isinf(pf_raw):  pf = 99.0
    elif isinstance(pf_raw, float) and math.isnan(pf_raw): pf = 0.0
    else: pf = float(pf_raw)

    avg_w = avg_l = 0.0
    if not trades.empty and 'PnL' in trades.columns:
        wins   = trades.loc[trades['PnL'] > 0, 'PnL']
        losses = trades.loc[trades['PnL'] < 0, 'PnL']
        avg_w  = float(wins.mean())   if len(wins)   else 0.0
        avg_l  = float(losses.mean()) if len(losses) else 0.0

    return dict(
        n=n, tpy=round(n/yrs, 1), pf=round(pf, 3),
        wr=round(float(stats.get('Win Rate [%]', 0)), 1),
        dd=round(abs(float(stats.get('Max. Drawdown [%]', 0))), 2),
        ret=round(float(stats.get('Return [%]', 0)), 2),
        avg_w=round(avg_w, 2), avg_l=round(avg_l, 2),
        trades=trades, eq=eq, years=yrs,
    )

# ── Portfolio combination ─────────────────────────────────────────────────────
def combine(leg_eu, leg_sp):
    """Merge two $50k legs into a $100k portfolio — no extra scaling needed."""
    pnl_eu = leg_eu['trades']['PnL'].tolist() if not leg_eu['trades'].empty else []
    pnl_sp = leg_sp['trades']['PnL'].tolist() if not leg_sp['trades'].empty else []
    all_pnl = pnl_eu + pnl_sp

    gp  = sum(p for p in all_pnl if p > 0)
    gl  = abs(sum(p for p in all_pnl if p < 0))
    pf  = round(gp / gl, 3) if gl > 0 else 99.0
    n   = len(all_pnl)
    wr  = round(sum(1 for p in all_pnl if p > 0) / n * 100, 1) if n else 0.0
    yrs = max(leg_eu['years'], leg_sp['years'])
    tpy = round((leg_eu['n'] + leg_sp['n']) / yrs, 1)
    avg = round(float(np.mean(all_pnl)), 2) if all_pnl else 0.0

    # Combined equity curve (sum of two $50k legs = $100k portfolio)
    eq_eu, eq_sp = leg_eu['eq'], leg_sp['eq']
    max_dd = combined_ret = float('nan')
    if eq_eu is not None and eq_sp is not None:
        idx      = eq_eu.index.union(eq_sp.index)
        combined = (eq_eu.reindex(idx, method='ffill') +
                    eq_sp.reindex(idx, method='ffill'))
        peak     = combined.cummax()
        max_dd   = round(float(((peak - combined) / peak * 100).max()), 2)
        combined_ret = round(float((combined.iloc[-1] / combined.iloc[0] - 1) * 100), 2)

    wins   = [p for p in all_pnl if p > 0]
    losses = [p for p in all_pnl if p < 0]
    return dict(
        n=n, tpy=tpy, pf=pf, wr=wr, dd=max_dd, ret=combined_ret, years=yrs,
        all_pnl=all_pnl,
        avg=avg,
        med=round(float(np.median(all_pnl)), 2) if all_pnl else 0.0,
        std=round(float(np.std(all_pnl)), 2) if all_pnl else 0.0,
        best=round(float(max(all_pnl)), 2) if all_pnl else 0.0,
        worst=round(float(min(all_pnl)), 2) if all_pnl else 0.0,
        avg_w=round(float(np.mean(wins)), 2) if wins else 0.0,
        avg_l=round(float(np.mean(losses)), 2) if losses else 0.0,
        monthly_exp=round(avg * tpy / 12, 2),
        annual_exp=round(avg * n / yrs, 0),
    )

# ── Monthly distribution ──────────────────────────────────────────────────────
def monthly_dist(trades_eu, trades_sp):
    rows = []
    for tdf, asset in [(trades_eu, 'EURUSD'), (trades_sp, 'SP500')]:
        if tdf.empty or 'ExitTime' not in tdf.columns:
            continue
        for _, r in tdf.iterrows():
            rows.append({'date': r['ExitTime'], 'pnl': r['PnL'], 'asset': asset})
    if not rows:
        return pd.DataFrame(columns=['month', 'pnl', 'n_trades', 'month_str'])
    df = pd.DataFrame(rows)
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    grp = df.groupby('month')['pnl'].agg(['sum', 'count']).reset_index()
    grp.columns = ['month', 'pnl', 'n_trades']
    grp['month_str'] = grp['month'].astype(str)
    return grp

# ── Time-to-target Monte Carlo ────────────────────────────────────────────────
def time_to_target(all_pnl, trades_eu, trades_sp, target_usd,
                   max_dd_usd, max_daily_usd, min_tdays,
                   max_sim_days=730, n_sim=N_SIM, seed=42):
    """
    Simulate how many calendar days until cumulative PnL >= target_usd.
    Tracks FTMO breach conditions: total drawdown and daily loss.
    Returns median/mean days, pass rate within 30/60 day windows.
    """
    rng     = np.random.default_rng(seed)
    pnl_arr = np.array(all_pnl, dtype=float)

    all_dates = []
    for tdf in [trades_eu, trades_sp]:
        if not tdf.empty and 'ExitTime' in tdf.columns:
            all_dates += pd.to_datetime(tdf['ExitTime']).tolist()
    lam = len(all_pnl) / max(len(pd.bdate_range(min(all_dates), max(all_dates))), 1) if all_dates else 0.25

    days_to_hit = []
    breached    = 0
    pass30      = 0   # hit target within 30 trading days without breach
    pass60      = 0
    never_hit   = 0

    for sim in range(n_sim):
        equity   = ACCOUNT
        cum      = 0.0
        active_d = 0
        hit_day  = None
        blown    = False

        for day in range(1, max_sim_days + 1):
            n_today = int(rng.poisson(lam))
            if n_today > 0:
                active_d += 1
            day_pnl = 0.0
            for _ in range(n_today):
                p = float(pnl_arr[rng.integers(len(pnl_arr))])
                day_pnl += p
                cum     += p
                equity  += p

            if day_pnl < -max_daily_usd or (ACCOUNT - equity) > max_dd_usd:
                blown = True
                breached += 1
                break

            if cum >= target_usd and hit_day is None:
                hit_day = day

        if blown:
            continue
        if hit_day is not None:
            days_to_hit.append(hit_day)
            td_at_hit = round((1 - math.exp(-lam)) * hit_day)
            if hit_day <= 22 and td_at_hit >= min_tdays: pass30 += 1
            if hit_day <= 43 and td_at_hit >= min_tdays: pass60 += 1
        else:
            never_hit += 1

    valid = [d for d in days_to_hit]
    if not valid:
        return dict(
            lam=round(lam, 4),
            breach_rate=round(breached / n_sim * 100, 1),
            hit_rate=0.0,
            p25_days=None, median_days=None, mean_days=None, p75_days=None,
            pass30=0.0, pass60=0.0,
            monthly_exp_months=None,
        )

    return dict(
        lam=round(lam, 4),
        breach_rate=round(breached / n_sim * 100, 1),
        hit_rate=round(len(valid) / n_sim * 100, 1),
        p25_days=int(np.percentile(valid, 25)),
        median_days=int(np.median(valid)),
        mean_days=int(np.mean(valid)),
        p75_days=int(np.percentile(valid, 75)),
        pass30=round(pass30 / n_sim * 100, 1),
        pass60=round(pass60 / n_sim * 100, 1),
        monthly_exp_months=round(target_usd / max(abs(sum(all_pnl) / max(len(all_pnl),1) * lam * 21), 0.01), 1),
    )

# ── HTML Report ───────────────────────────────────────────────────────────────
def _pf_c(v):  return '#4caf50' if v >= 2.0 else ('#f0b429' if v >= 1.5 else '#f44336')
def _dd_c(v):  return '#4caf50' if v < 4.0  else ('#f0b429' if v < 8.0  else '#f44336')
def _tpy_c(v): return '#4caf50' if v >= 40  else '#f44336'
def _pct_c(v): return '#4caf50' if v >= 50  else ('#f0b429' if v >= 20  else '#f44336')

def generate_report(scenarios_out, monthly_data, eu_avg_p, sp_avg_p, eu_comm, sp_comm):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── Scenario cards
    cards_html = ''
    for sc in scenarios_out:
        p  = sc['port']
        t1 = sc['t1']
        t2 = sc['t2']
        label = sc['label']
        risk_usd = sc['risk_dollars']

        def days_cell(t, phase):
            if t['median_days'] is None:
                return f'<td class="bad">Never</td>'
            d = t['median_days']
            weeks = round(d * 7 / 5, 1)
            return f'<td><b>{d}</b> trading days<br><small style="color:#888">~{weeks} weeks<br>P25–P75: {t["p25_days"]}–{t["p75_days"]} days</small></td>'

        cards_html += f'''
<div class="sc-card">
  <div class="sc-title">{label} &nbsp;<span class="sc-risk">${risk_usd:,.0f} risked / trade / leg</span></div>
  <div class="sc-grid">
    <div class="sc-box">
      <div class="sc-head">Combined Portfolio ($100K)</div>
      <table>
        <tr><td>Trades / year</td><td style="color:{_tpy_c(p['tpy'])}">{p['tpy']}</td></tr>
        <tr><td>Profit Factor</td><td style="color:{_pf_c(p['pf'])}">{p['pf']}</td></tr>
        <tr><td>Win Rate</td><td>{p['wr']}%</td></tr>
        <tr><td>Max Drawdown</td><td style="color:{_dd_c(p['dd'])}">{p['dd']}%</td></tr>
        <tr><td>2-Year Return</td><td style="color:{'#4caf50' if p['ret']>0 else '#f44336'}">{p['ret']}%</td></tr>
      </table>
    </div>
    <div class="sc-box">
      <div class="sc-head">Per-Trade P&L (combined, actual $)</div>
      <table>
        <tr><td>Avg profit / trade</td><td style="color:{'#4caf50' if p['avg']>0 else '#f44336'}"><b>${p['avg']:,.2f}</b></td></tr>
        <tr><td>Median</td><td>${p['med']:,.2f}</td></tr>
        <tr><td>Avg winner</td><td class="good">${p['avg_w']:,.2f}</td></tr>
        <tr><td>Avg loser</td><td class="bad">${p['avg_l']:,.2f}</td></tr>
        <tr><td>Best trade</td><td class="good">${p['best']:,.2f}</td></tr>
        <tr><td>Worst trade</td><td class="bad">${p['worst']:,.2f}</td></tr>
        <tr><td>Std deviation</td><td>${p['std']:,.2f}</td></tr>
      </table>
    </div>
    <div class="sc-box">
      <div class="sc-head">Expectancy</div>
      <table>
        <tr><td>Monthly expectancy</td><td><b>${p['monthly_exp']:,.2f}</b></td></tr>
        <tr><td>Annual expectancy</td><td><b>${p['annual_exp']:,.0f}</b></td></tr>
        <tr><td>Months to Phase 1 ($10K)</td><td><b>{round(10000/max(p['monthly_exp'],0.01),1) if p['monthly_exp']>0 else 'N/A'} mo avg</b></td></tr>
        <tr><td>Months to Phase 2 ($5K)</td><td><b>{round(5000/max(p['monthly_exp'],0.01),1) if p['monthly_exp']>0 else 'N/A'} mo avg</b></td></tr>
        <tr><td>Breach rate (DD/daily)</td><td style="color:{'#f44336' if t1['breach_rate']>10 else '#f0b429'}">{t1['breach_rate']}%</td></tr>
      </table>
    </div>
    <div class="sc-box">
      <div class="sc-head">Time to FTMO Targets (Monte Carlo, {N_SIM:,} sims)</div>
      <table>
        <tr><th>Target</th><th>P25</th><th>Median</th><th>P75</th><th>In 30 days</th><th>In 60 days</th></tr>
        <tr>
          <td>Phase 1 ($10K)</td>
          <td>{t1['p25_days'] or '—'}</td>
          <td><b>{t1['median_days'] or '—'}</b></td>
          <td>{t1['p75_days'] or '—'}</td>
          <td style="color:{_pct_c(t1['pass30'])}">{t1['pass30']}%</td>
          <td style="color:{_pct_c(t1['pass60'])}">{t1['pass60']}%</td>
        </tr>
        <tr>
          <td>Phase 2 ($5K)</td>
          <td>{t2['p25_days'] or '—'}</td>
          <td><b>{t2['median_days'] or '—'}</b></td>
          <td>{t2['p75_days'] or '—'}</td>
          <td style="color:{_pct_c(t2['pass30'])}">{t2['pass30']}%</td>
          <td style="color:{_pct_c(t2['pass60'])}">{t2['pass60']}%</td>
        </tr>
      </table>
      <small style="color:#666">Trading days shown. 22 trading days ≈ 30 cal days. 43 ≈ 60 cal days.</small>
    </div>
  </div>
</div>'''

    # ── Monthly charts data (for each scenario)
    chart_blocks = ''
    for i, sc in enumerate(scenarios_out):
        m = sc['monthly']
        if m.empty:
            continue
        months_js  = json.dumps(m['month_str'].tolist())
        pnls_js    = json.dumps([round(p, 2) for p in m['pnl'].tolist()])
        colors_js  = ('[' + ','.join("'#4caf50'" if p >= 0 else "'#f44336'"
                                     for p in m['pnl'].tolist()) + ']')
        chart_blocks += f'''
<div class="chart-wrap">
  <h3>Monthly P&L — {sc['label']}</h3>
  <canvas id="chart{i}" style="max-height:220px"></canvas>
</div>
<script>
new Chart(document.getElementById('chart{i}').getContext('2d'), {{
  type: 'bar',
  data: {{ labels: {months_js}, datasets: [{{
    data: {pnls_js}, backgroundColor: {colors_js}, borderRadius: 4
  }}]}},
  options: {{ responsive: true,
    plugins: {{ legend: {{display:false}}, tooltip: {{callbacks: {{
      label: c => '$' + c.raw.toLocaleString('en', {{minimumFractionDigits:2}})
    }}}}  }},
    scales: {{
      x: {{ticks:{{color:'#888',maxRotation:45}}, grid:{{color:'#1a1a1a'}}}},
      y: {{ticks:{{color:'#888',callback:v=>'$'+v.toLocaleString()}}, grid:{{color:'#222'}}}}
    }}
  }}
}});
</script>'''

    # ── Position sizing explainer
    eu_atr_est = 0.0020
    sp_atr_est = 7.0
    rows = ''
    for sc in SCENARIOS:
        r = sc['risk_dollars']
        eu_stop = ATR_MULT * eu_atr_est
        sp_stop = ATR_MULT * sp_atr_est
        eu_size = int(r / eu_stop)
        sp_size = int(r / sp_stop)
        eu_notl = eu_size * eu_avg_p
        sp_notl = sp_size * sp_avg_p
        eu_lev  = round(eu_notl / LEG_CAPITAL, 2)
        sp_lev  = round(sp_notl / LEG_CAPITAL, 2)
        rows += f'''<tr>
          <td>{sc["label"]}</td>
          <td>${r:,.0f} / leg</td>
          <td>~{eu_atr_est*10000:.0f} pip ATR → stop {eu_stop*10000:.0f} pips</td>
          <td>~{eu_size:,} units (~{eu_size/100000:.1f} lots)</td>
          <td>${eu_notl:,.0f} ({eu_lev}× leverage)</td>
          <td>~${sp_atr_est:.1f} ATR → stop ${sp_stop:.1f}</td>
          <td>~{sp_size} shares</td>
          <td>${sp_notl:,.0f} ({sp_lev}× leverage)</td>
        </tr>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Risk-Sized Portfolio — FTMO Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1500px;margin:0 auto}}
h1{{font-size:22px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:22px}}
h2{{font-size:15px;color:#fff;margin:24px 0 12px;border-left:3px solid #f0b429;padding-left:10px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#666;padding:7px 10px;text-align:left;border-bottom:1px solid #2a2a2a}}
td{{padding:7px 10px;border-bottom:1px solid #161616;vertical-align:top}}
tr:hover td{{background:#1e1e1e}}
.good{{color:#4caf50}} .warn{{color:#f0b429}} .bad{{color:#f44336}}
.box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;margin-bottom:14px}}
.chart-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:18px;margin-bottom:14px}}
.chart-wrap h3{{font-size:12px;color:#888;margin-bottom:12px}}
/* Scenario cards */
.sc-card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:14px;padding:20px;margin-bottom:20px}}
.sc-title{{font-size:16px;font-weight:bold;color:#f0b429;margin-bottom:14px}}
.sc-risk{{font-size:12px;color:#888;font-weight:normal}}
.sc-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
@media(max-width:1200px){{.sc-grid{{grid-template-columns:repeat(2,1fr)}}}}
.sc-box{{background:#111;border-radius:8px;padding:12px}}
.sc-head{{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;border-bottom:1px solid #2a2a2a;padding-bottom:6px}}
.info-box{{background:#0e1a2a;border:1px solid #2196f3;border-radius:8px;padding:14px 16px;margin-top:14px;font-size:12px;line-height:1.8;color:#bbdefb}}
.warn-box{{background:#2a1a0a;border:1px solid #f0b429;border-radius:8px;padding:14px 16px;margin-top:14px;font-size:12px;line-height:1.8;color:#fff3cc}}
</style>
</head>
<body>
<h1>Risk-Based Position Sizing — EURUSD BB(10) EMA(100) + SPY BB(20) EMA(100)</h1>
<p class="sub">ATR(14) stop-loss at {ATR_MULT}× ATR &nbsp;|&nbsp; 50:1 EURUSD &nbsp;·&nbsp; 10:1 SPY &nbsp;|&nbsp; $50K per leg on $100K account &nbsp;|&nbsp; Generated {now}</p>

<h2>Position Sizing Mechanics (estimated at typical ATR)</h2>
<div class="box">
  <table>
    <thead><tr>
      <th>Scenario</th><th>$ Risked/Leg</th>
      <th>EURUSD ATR stop</th><th>EURUSD size</th><th>EURUSD notional</th>
      <th>SPY ATR stop</th><th>SPY size</th><th>SPY notional</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="info-box">
    <b>How it works:</b> At entry, ATR(14) is read from the last bar. Stop is placed at <b>entry &minus; {ATR_MULT}×ATR</b>.
    Position size = <b>Risk $ &div; (entry &minus; stop)</b>, capped at {EU_LEVERAGE}:1 EURUSD / {SP_LEVERAGE}:1 SPY.
    EURUSD avg price {eu_avg_p:.5f} &nbsp;|&nbsp; SPY avg price ${sp_avg_p:.2f} &nbsp;|&nbsp; ATR estimates are illustrative; actual sizes vary bar-by-bar.
  </div>
</div>

<h2>Backtest Results by Risk Scenario</h2>
{cards_html}

<h2>Monthly P&L Distribution</h2>
{chart_blocks}

<div class="warn-box">
  <b>FTMO Min Trading Days:</b> FTMO Phase 1 requires &ge;10 active trading days in 30 calendar days.
  With ~{round(scenarios_out[0]["t1"]["lam"]*21,1)} expected trades/month, the strategy trades on ~{round((1-math.exp(-scenarios_out[0]["t1"]["lam"]))*22,1)} distinct days per 30-day window.
  The "In 30 days" pass rates above include the minimum-days constraint; rows marked "0%" fail this rule even when the profit target is hit.
  Solution: add 3&ndash;5 more instruments to reach 10+ active days/month.
</div>
</body>
</html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] Report: {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading cached data...')
    df_eu = load_df('EURUSD')
    df_sp = load_df('SP500')
    eu_avg_p = float(df_eu['Close'].mean())
    sp_avg_p = float(df_sp['Close'].mean())

    eu_comm = BASE_COMM + EU_TICK * 1 / eu_avg_p
    sp_comm = BASE_COMM + SPY_TICK * 2 / sp_avg_p

    eu_margin = 1 / EU_LEVERAGE    # 0.02
    sp_margin = 1 / SP_LEVERAGE    # 0.10

    print(f'  EURUSD: avg {eu_avg_p:.5f}  |  commission+slip: {eu_comm*10000:.2f} pips/order  |  margin {eu_margin:.3f} ({EU_LEVERAGE}:1)')
    print(f'  SPY:    avg ${sp_avg_p:.2f}  |  commission+slip: {sp_comm*10000:.3f} bps/order   |  margin {sp_margin:.2f} ({SP_LEVERAGE}:1)')
    print(f'  ATR({ATR_PERIOD}) stop at {ATR_MULT}×ATR | LEG_CAPITAL=${LEG_CAPITAL:,.0f}')

    sep = '=' * 66
    scenarios_out = []

    for sc in SCENARIOS:
        rp = sc['risk_pct']
        print(f'\n{sep}')
        print(f'  SCENARIO: {sc["label"]}  (risk_pct={rp:.0%}  risk_$ = ${sc["risk_dollars"]:,.0f}/leg)')
        print(sep)

        print('  Running EURUSD leg...')
        eu = run_bt(df_eu, BBRSIRiskEU, eu_comm, eu_margin, rp)
        print('  Running SP500 leg...')
        sp = run_bt(df_sp, BBRSIRiskSP, sp_comm, sp_margin, rp)

        port = combine(eu, sp)
        monthly = monthly_dist(eu['trades'], sp['trades'])

        print(f'\n  Legs:')
        print(f'    EURUSD  {eu["n"]:3d} trades ({eu["tpy"]}/yr)  PF {eu["pf"]:.3f}  WR {eu["wr"]}%  DD {eu["dd"]}%  avg_w ${eu["avg_w"]:,.0f}  avg_l ${eu["avg_l"]:,.0f}')
        print(f'    SP500   {sp["n"]:3d} trades ({sp["tpy"]}/yr)  PF {sp["pf"]:.3f}  WR {sp["wr"]}%  DD {sp["dd"]}%  avg_w ${sp["avg_w"]:,.0f}  avg_l ${sp["avg_l"]:,.0f}')
        print(f'\n  Combined ($100K portfolio):')
        print(f'    Trades:   {port["n"]} ({port["tpy"]}/yr)  PF: {port["pf"]}  WR: {port["wr"]}%  DD: {port["dd"]}%  Ret: {port["ret"]}%')
        print(f'\n  Per-trade (actual $, combined):')
        print(f'    Avg:     ${port["avg"]:>8,.2f}')
        print(f'    Median:  ${port["med"]:>8,.2f}')
        print(f'    Std:     ${port["std"]:>8,.2f}')
        print(f'    Avg win: ${port["avg_w"]:>8,.2f}  |  Avg loss: ${port["avg_l"]:>8,.2f}')
        print(f'    Best:    ${port["best"]:>8,.2f}  |  Worst:    ${port["worst"]:>8,.2f}')
        print(f'    Monthly expectancy: ${port["monthly_exp"]:>8,.2f}')
        print(f'    Annual expectancy:  ${port["annual_exp"]:>8,.0f}')
        p1_months = round(10000 / max(port['monthly_exp'], 0.01), 1) if port['monthly_exp'] > 0 else 'N/A'
        p2_months = round(5000  / max(port['monthly_exp'], 0.01), 1) if port['monthly_exp'] > 0 else 'N/A'
        print(f'    Months to Phase 1 ($10K): {p1_months}')
        print(f'    Months to Phase 2 ($5K):  {p2_months}')

        print(f'\n  Monthly distribution:')
        if not monthly.empty:
            for _, row in monthly.iterrows():
                bar  = '#' * max(1, int(abs(row['pnl']) / 60))
                sign = '+' if row['pnl'] >= 0 else '-'
                flag = '  <-- negative' if row['pnl'] < 0 else ''
                print(f'    {row["month_str"]}  {sign}${abs(row["pnl"]):>8,.0f} ({int(row["n_trades"])} trades)  {bar}{flag}')

        print(f'\n  Time-to-target Monte Carlo ({N_SIM:,} sims)...')
        t1 = time_to_target(
            port['all_pnl'], eu['trades'], sp['trades'],
            target_usd=10_000,
            max_dd_usd=10_000, max_daily_usd=5_000, min_tdays=10, seed=42)
        t2 = time_to_target(
            port['all_pnl'], eu['trades'], sp['trades'],
            target_usd=5_000,
            max_dd_usd=10_000, max_daily_usd=5_000, min_tdays=10, seed=43)

        def fmt_t(t, name, target):
            if t['median_days'] is None:
                print(f'    {name} (${target:,}): never reached in sims  breach={t["breach_rate"]}%')
            else:
                weeks_med  = round(t['median_days'] * 7 / 5, 1)
                weeks_mean = round(t['mean_days'] * 7 / 5, 1)
                print(f'    {name} (${target:,}):')
                print(f'      Reach rate: {t["hit_rate"]}%  |  Breach rate: {t["breach_rate"]}%')
                print(f'      Median: {t["median_days"]} trading days (~{weeks_med} weeks)')
                print(f'      Mean:   {t["mean_days"]} trading days (~{weeks_mean} weeks)')
                print(f'      P25-P75: {t["p25_days"]}–{t["p75_days"]} trading days')
                print(f'      Pass within 30 tdays (Phase 1 window): {t["pass30"]}%')
                print(f'      Pass within 60 tdays (Phase 2 window): {t["pass60"]}%')

        fmt_t(t1, 'Phase 1', 10_000)
        fmt_t(t2, 'Phase 2',  5_000)

        scenarios_out.append(dict(
            label=sc['label'], risk_dollars=sc['risk_dollars'],
            port=port, monthly=monthly, t1=t1, t2=t2,
        ))

    print(f'\n{sep}')
    print('  SUMMARY')
    print(sep)
    print(f'  {"Scenario":<20}  {"Avg $/trade":>12}  {"Monthly exp":>12}  {"Mo to P1":>10}  {"Mo to P2":>10}  {"PF":>6}  {"DD":>6}')
    for sc in scenarios_out:
        p = sc['port']
        mp = p['monthly_exp']
        m1 = round(10000/max(mp,0.01),1) if mp > 0 else 'N/A'
        m2 = round(5000/max(mp,0.01),1) if mp > 0 else 'N/A'
        print(f'  {sc["label"]:<20}  ${p["avg"]:>10,.2f}  ${mp:>10,.2f}  {str(m1):>10}  {str(m2):>10}  {p["pf"]:>6.3f}  {p["dd"]:>5.2f}%')

    generate_report(scenarios_out, None, eu_avg_p, sp_avg_p, eu_comm, sp_comm)

if __name__ == '__main__':
    main()

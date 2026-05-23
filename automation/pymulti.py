#!/usr/bin/env python3
"""
Multi-instrument portfolio: EURUSD, GBPUSD, AUDUSD, SP500(SPY),
NAS100(QQQ), XAUUSD(spot), DAX(^GDAXI)
Strategy: BB(10) + RSI(7)<35 + EMA(100) | No stop | size=0.99
Filter:   PF > 2.0  AND  Max DD < 8%
$100k account | equal allocation across qualifiers
Outputs:  trades/month, monthly expectancy $, FTMO P1/P2 timeline
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from backtesting import Backtest, Strategy

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT    = 100_000
TRADE_SIZE = 0.99
BASE_COMM  = 0.0001          # 0.01% per order
N_SIM      = 10_000
BB_PERIOD  = 10
RSI_PERIOD = 7
RSI_LO     = 35
EMA_PERIOD = 100

PF_MIN   = 2.0
DD_MAX   = 8.0

_here      = os.path.dirname(__file__)
DATA_DIR   = os.path.join(_here, 'pydata')
REPORT_OUT = os.path.join(_here, 'pymulti-report.html')
os.makedirs(DATA_DIR, exist_ok=True)

FTMO = dict(
    p1=dict(name='Phase 1', target=10_000, max_dd=10_000, max_daily=5_000, tdays=22, min_td=10),
    p2=dict(name='Phase 2', target=5_000,  max_dd=10_000, max_daily=5_000, tdays=43, min_td=10),
)

# tick_size: minimum price move for slippage; n_ticks: ticks of slippage per fill
INSTRUMENTS = {
    'EURUSD': dict(ticker='EURUSD=X', tick=0.0001, n_ticks=1, label='EUR/USD'),
    'GBPUSD': dict(ticker='GBPUSD=X', tick=0.0001, n_ticks=1, label='GBP/USD'),
    'AUDUSD': dict(ticker='AUDUSD=X', tick=0.0001, n_ticks=1, label='AUD/USD'),
    'SP500':  dict(ticker='SPY',      tick=0.01,   n_ticks=2, label='S&P 500 (SPY)'),
    'NAS100': dict(ticker='QQQ',      tick=0.01,   n_ticks=2, label='NAS100 (QQQ)'),
    'XAUUSD': dict(ticker=None,       tick=0.10,   n_ticks=1, label='XAU/USD (Gold)'),  # use cached
    'DAX':    dict(ticker='^GDAXI',   tick=1.0,    n_ticks=2, label='DAX (^GDAXI)'),
}

# ── Indicators ────────────────────────────────────────────────────────────────
def _bb_upper(arr, n):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() + 2*s.rolling(n, min_periods=n).std(ddof=0)).values

def _bb_mid(arr, n):
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().values

def _bb_lower(arr, n):
    s = pd.Series(arr, dtype=float)
    return (s.rolling(n, min_periods=n).mean() - 2*s.rolling(n, min_periods=n).std(ddof=0)).values

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
class BBRSIStrategy(Strategy):
    def init(self):
        c = self.data.Close
        self.bb_up  = self.I(_bb_upper, c, BB_PERIOD)
        self.bb_mid = self.I(_bb_mid,   c, BB_PERIOD)
        self.bb_lo  = self.I(_bb_lower, c, BB_PERIOD)
        self.rsi    = self.I(_rsi,      c, RSI_PERIOD)
        self.ema    = self.I(_ema,      c, EMA_PERIOD)

    def next(self):
        if any(np.isnan(x[-1]) for x in (self.bb_lo, self.rsi, self.ema)):
            return
        c = self.data.Close[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < RSI_LO and c > self.ema[-1]:
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(name, ticker):
    cache = os.path.join(DATA_DIR, f'{name}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) >= 200:
            return df
    if ticker is None:
        print(f'    {name}: no ticker and no cache — skipping')
        return None
    print(f'    {name}: downloading {ticker}...')
    df = yf.download(ticker, period='730d', interval='1h', progress=False)
    if df is None or df.empty:
        print(f'    {name}: download failed')
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    if len(df) < 200:
        print(f'    {name}: only {len(df)} bars — skipping')
        return None
    df.to_csv(cache)
    return df

# ── Backtest runner ───────────────────────────────────────────────────────────
def run_bt(name, df, commission):
    try:
        bt    = Backtest(df, BBRSIStrategy, cash=ACCOUNT,
                         commission=commission, exclusive_orders=True)
        stats = bt.run()
    except Exception as e:
        return dict(name=name, error=str(e), ok=False)

    trades = stats._trades.copy() if hasattr(stats, '_trades') and len(stats._trades) else pd.DataFrame()
    eq     = stats._equity_curve['Equity'].copy() if hasattr(stats, '_equity_curve') else None
    n      = int(stats.get('# Trades', 0))
    yrs    = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)

    pf_raw = stats.get('Profit Factor', float('nan'))
    if isinstance(pf_raw, float) and math.isinf(pf_raw):  pf = 99.0
    elif isinstance(pf_raw, float) and math.isnan(pf_raw): pf = 0.0
    else: pf = float(pf_raw)

    wins   = trades.loc[trades['PnL'] > 0, 'PnL'] if not trades.empty else pd.Series()
    losses = trades.loc[trades['PnL'] < 0, 'PnL'] if not trades.empty else pd.Series()
    dd     = round(abs(float(stats.get('Max. Drawdown [%]', 0))), 2)
    ok     = pf >= PF_MIN and dd < DD_MAX

    return dict(
        name=name, ok=ok, error=None,
        n=n, tpy=round(n/yrs, 1), pf=round(pf, 3),
        wr=round(float(stats.get('Win Rate [%]', 0)), 1),
        dd=dd, ret=round(float(stats.get('Return [%]', 0)), 2),
        avg_w=round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_l=round(float(losses.mean()), 2) if len(losses) else 0.0,
        sharpe=round(float(stats.get('Sharpe Ratio', 0)), 3),
        trades=trades, eq=eq, years=yrs,
        bars=len(df),
        date_from=df.index[0].date(), date_to=df.index[-1].date(),
    )

# ── Portfolio ─────────────────────────────────────────────────────────────────
def build_portfolio(qual):
    """Combine qualified results into one $100k portfolio (equal weight)."""
    n     = len(qual)
    scale = 1.0 / n

    all_pnl      = []
    all_trades   = []   # list of DataFrames with ExitTime + scaled PnL
    eq_combined  = None

    for r in qual:
        pnl_scaled = [p * scale for p in r['trades']['PnL'].tolist()] if not r['trades'].empty else []
        all_pnl   += pnl_scaled

        if not r['trades'].empty and 'ExitTime' in r['trades'].columns:
            tdf = r['trades'][['ExitTime','PnL']].copy()
            tdf['PnL'] = tdf['PnL'] * scale
            tdf['asset'] = r['name']
            all_trades.append(tdf)

        if r['eq'] is not None:
            eq_scaled = r['eq'] * scale
            if eq_combined is None:
                eq_combined = eq_scaled
            else:
                idx         = eq_combined.index.union(eq_scaled.index)
                eq_combined = eq_combined.reindex(idx, method='ffill') + eq_scaled.reindex(idx, method='ffill')

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    gp  = sum(p for p in all_pnl if p > 0)
    gl  = abs(sum(p for p in all_pnl if p < 0))
    pf  = round(gp / gl, 3) if gl > 0 else 99.0
    nt  = len(all_pnl)
    wr  = round(sum(1 for p in all_pnl if p > 0) / nt * 100, 1) if nt else 0.0
    yrs = max(r['years'] for r in qual)
    tpy = round(sum(r['n'] for r in qual) / yrs, 1)

    max_dd = combined_ret = float('nan')
    if eq_combined is not None:
        peak         = eq_combined.cummax()
        max_dd       = round(float(((peak - eq_combined) / peak * 100).max()), 2)
        combined_ret = round(float((eq_combined.iloc[-1] / eq_combined.iloc[0] - 1) * 100), 2)

    avg = round(float(np.mean(all_pnl)), 2) if all_pnl else 0.0
    wins_arr   = [p for p in all_pnl if p > 0]
    losses_arr = [p for p in all_pnl if p < 0]

    return dict(
        n=nt, tpy=tpy, pf=pf, wr=wr, dd=max_dd, ret=combined_ret, years=yrs,
        all_pnl=all_pnl, all_trades=all_trades_df, eq=eq_combined,
        avg=avg,
        med=round(float(np.median(all_pnl)), 2) if all_pnl else 0.0,
        std=round(float(np.std(all_pnl)), 2) if all_pnl else 0.0,
        best=round(float(max(all_pnl)), 2) if all_pnl else 0.0,
        worst=round(float(min(all_pnl)), 2) if all_pnl else 0.0,
        avg_w=round(float(np.mean(wins_arr)), 2) if wins_arr else 0.0,
        avg_l=round(float(np.mean(losses_arr)), 2) if losses_arr else 0.0,
        monthly_exp=round(avg * tpy / 12, 2),
        annual_exp=round(avg * nt / yrs, 0),
        n_instruments=n,
    )

# ── Monthly distribution ──────────────────────────────────────────────────────
def monthly_dist(port):
    df = port['all_trades']
    if df.empty or 'ExitTime' not in df.columns:
        return pd.DataFrame(columns=['month_str','pnl','n_trades','asset_counts'])
    df = df.copy()
    df['month'] = pd.to_datetime(df['ExitTime']).dt.to_period('M')
    grp = df.groupby('month').agg(pnl=('PnL','sum'), n_trades=('PnL','count')).reset_index()
    grp['month_str'] = grp['month'].astype(str)
    return grp

# ── Active-days analysis ──────────────────────────────────────────────────────
def active_days_analysis(port):
    df = port['all_trades']
    if df.empty or 'ExitTime' not in df.columns:
        return dict(avg_active_per_month=0, total_days=0)
    df = df.copy()
    df['date'] = pd.to_datetime(df['ExitTime']).dt.date
    df['month'] = pd.to_datetime(df['ExitTime']).dt.to_period('M')
    daily  = df.groupby(['month','date']).size().reset_index()
    per_mo = daily.groupby('month').size()
    return dict(
        avg_active_per_month=round(float(per_mo.mean()), 1),
        median_active_per_month=round(float(per_mo.median()), 1),
        min_active=int(per_mo.min()),
        max_active=int(per_mo.max()),
        months_meeting_10=int((per_mo >= 10).sum()),
        total_months=len(per_mo),
    )

# ── Monte Carlo: time to target ───────────────────────────────────────────────
def time_to_target(port, target_usd, max_dd_usd, max_daily_usd, min_tdays,
                   max_sim_days=730, seed=42):
    pnl_arr   = np.array(port['all_pnl'], dtype=float)
    rng       = np.random.default_rng(seed)

    df = port['all_trades']
    if not df.empty and 'ExitTime' in df.columns:
        all_dates = pd.to_datetime(df['ExitTime']).tolist()
        bdays     = pd.bdate_range(min(all_dates), max(all_dates))
        lam       = len(pnl_arr) / max(len(bdays), 1)
    else:
        lam = 0.3

    days_to_hit = []
    breached    = 0
    pass22 = pass43 = 0

    for _ in range(N_SIM):
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
            for __ in range(n_today):
                p = float(pnl_arr[rng.integers(len(pnl_arr))])
                day_pnl += p
                cum     += p
                equity  += p

            if day_pnl < -max_daily_usd or (ACCOUNT - equity) > max_dd_usd:
                blown = True; breached += 1; break

            if cum >= target_usd and hit_day is None:
                hit_day = day

        if blown:
            continue
        if hit_day is not None:
            days_to_hit.append(hit_day)
            act = round((1 - math.exp(-lam)) * hit_day)
            if hit_day <= 22 and act >= min_tdays: pass22 += 1
            if hit_day <= 43 and act >= min_tdays: pass43 += 1

    valid = days_to_hit
    exp_td_22 = round((1 - math.exp(-lam)) * 22, 1)
    exp_td_43 = round((1 - math.exp(-lam)) * 43, 1)

    if not valid:
        return dict(
            lam=round(lam, 4), exp_trades_month=round(lam*21, 1),
            exp_tdays_22=exp_td_22, exp_tdays_43=exp_td_43,
            breach_rate=round(breached/N_SIM*100, 1),
            hit_rate=0.0, p25=None, median=None, mean=None, p75=None,
            pass22=0.0, pass43=0.0,
        )

    return dict(
        lam=round(lam, 4),
        exp_trades_month=round(lam * 21, 1),
        exp_tdays_22=exp_td_22,
        exp_tdays_43=exp_td_43,
        breach_rate=round(breached / N_SIM * 100, 1),
        hit_rate=round(len(valid) / N_SIM * 100, 1),
        p25=int(np.percentile(valid, 25)),
        median=int(np.median(valid)),
        mean=int(np.mean(valid)),
        p75=int(np.percentile(valid, 75)),
        pass22=round(pass22 / N_SIM * 100, 1),
        pass43=round(pass43 / N_SIM * 100, 1),
        weeks_median=round(int(np.median(valid)) * 7 / 5, 1),
        weeks_p25=round(int(np.percentile(valid, 25)) * 7 / 5, 1),
        weeks_p75=round(int(np.percentile(valid, 75)) * 7 / 5, 1),
    )

# ── HTML Report ───────────────────────────────────────────────────────────────
def _c(v, ok_thresh, warn_thresh, higher=True):
    if higher:
        return '#4caf50' if v >= ok_thresh else ('#f0b429' if v >= warn_thresh else '#f44336')
    else:
        return '#4caf50' if v <= ok_thresh else ('#f0b429' if v <= warn_thresh else '#f44336')

def generate_report(all_results, qual_results, port, monthly, ada, mc1, mc2):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── Per-instrument table rows
    inst_rows = ''
    for r in all_results:
        if r.get('error'):
            inst_rows += f'<tr><td>{r["name"]}</td>' + '<td colspan="10" class="na">ERROR: ' + r["error"][:60] + '</td></tr>'
            continue
        ok_pf = r['pf'] >= PF_MIN
        ok_dd = r['dd'] < DD_MAX
        ok    = r['ok']
        badge = ('<span class="badge green">PASS</span>' if ok
                 else '<span class="badge red">FAIL</span>')
        inst_rows += (
            f'<tr class="{"ok-row" if ok else ""}">'
            f'<td><b>{r["name"]}</b><br><small class="muted">{INSTRUMENTS[r["name"]]["label"]}</small></td>'
            f'<td>{badge}</td>'
            f'<td style="color:{_c(r["pf"],2.0,1.5)}">{r["pf"]:.3f}</td>'
            f'<td style="color:{_c(r["dd"],4.0,8.0,False)}">{r["dd"]:.2f}%</td>'
            f'<td>{r["tpy"]}</td>'
            f'<td>{r["wr"]}%</td>'
            f'<td style="color:{"#4caf50" if r["avg_w"]>0 else "#f44336"}">${r["avg_w"]:,.0f}</td>'
            f'<td class="bad">${r["avg_l"]:,.0f}</td>'
            f'<td>{r["n"]}</td>'
            f'<td>{r["date_from"]}</td>'
            f'<td>{r["bars"]:,}</td>'
            f'</tr>'
        )

    # ── Monthly chart
    months_js = pnls_js = mcolors_js = '[]'
    if not monthly.empty:
        months_js  = json.dumps(monthly['month_str'].tolist())
        pnls_js    = json.dumps([round(p, 2) for p in monthly['pnl'].tolist()])
        mcolors_js = '[' + ','.join("'#4caf50'" if p >= 0 else "'#f44336'" for p in monthly['pnl']) + ']'

    # ── Gauge helper
    def gauge(label, pct, color, suffix='%'):
        w = min(float(pct or 0), 100)
        return (f'<div class="gauge-row">'
                f'<span class="g-label">{label}</span>'
                f'<div class="g-bar-wrap"><div class="g-bar" style="width:{w}%;background:{color}"></div></div>'
                f'<span class="g-val" style="color:{color}">{pct}{suffix}</span>'
                f'</div>')

    def mc_card(mc, phase):
        ph  = FTMO[phase]
        p22 = mc.get('pass22', 0)
        p43 = mc.get('pass43', 0)
        hit = mc.get('hit_rate', 0)
        brt = mc.get('breach_rate', 0)
        med = mc.get('median')
        p25 = mc.get('p25')
        p75 = mc.get('p75')
        wm  = mc.get('weeks_median')
        wp25 = mc.get('weeks_p25')
        wp75 = mc.get('weeks_p75')
        is_p1 = phase == 'p1'
        pass_val = p22 if is_p1 else p43
        pass_lbl = 'Within 30-cal-day window' if is_p1 else 'Within 60-cal-day window'
        target_str = '$10,000 (+10%)' if is_p1 else '$5,000 (+5%)'
        pc = _c(pass_val, 50, 20)
        td_ok = mc['exp_tdays_22'] if is_p1 else mc['exp_tdays_43']
        td_c  = '#4caf50' if td_ok >= 10 else ('#f0b429' if td_ok >= 7 else '#f44336')

        return f'''<div class="mc-card">
  <div class="mc-title">{ph["name"]} &mdash; {target_str}</div>
  <div class="mc-stats">
    <div class="mc-stat"><span class="mc-v">${ph["target"]:,}</span><span class="mc-l">Profit target</span></div>
    <div class="mc-stat"><span class="mc-v">{mc["exp_trades_month"]:.1f}</span><span class="mc-l">Trades/month</span></div>
    <div class="mc-stat"><span class="mc-v" style="color:{td_c}">{td_ok:.1f}</span><span class="mc-l">Active days/{"30" if is_p1 else "60"}-day</span></div>
    <div class="mc-stat"><span class="mc-v">{hit}%</span><span class="mc-l">Hit rate (unconstrained)</span></div>
  </div>
  <hr class="div">
  {gauge(pass_lbl, pass_val, pc)}
  {gauge("Hit target (any time)", hit, _c(hit,50,20))}
  {gauge("Breach DD/daily limit", brt, '#f44336' if brt > 5 else '#f0b429')}
  <hr class="div">
  <div class="timeline">
    <div class="tl-row"><span class="tl-l">Median time to target</span>
      <span class="tl-v"><b>{med or "N/A"}</b> trading days {"(~" + str(wm) + " weeks)" if wm else ""}</span></div>
    <div class="tl-row"><span class="tl-l">P25 &ndash; P75 range</span>
      <span class="tl-v">{p25 or "—"} &ndash; {p75 or "—"} days {"(" + str(wp25) + "–" + str(wp75) + " wks)" if wp25 else ""}</span></div>
    <div class="tl-row"><span class="tl-l">Min trading days required</span>
      <span class="tl-v" style="color:{td_c}">{ph["min_td"]} &nbsp; (expected: {td_ok:.1f} active days)</span></div>
  </div>
</div>'''

    qual_names = ', '.join(r['name'] for r in qual_results)
    n_qual     = len(qual_results)
    port_dd_c  = _c(port['dd'], 4.0, 8.0, False)
    port_pf_c  = _c(port['pf'], 2.0, 1.5)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi-Instrument Portfolio — FTMO Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1500px;margin:0 auto}}
h1{{font-size:22px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:22px}}
h2{{font-size:14px;color:#fff;margin:24px 0 10px;border-left:3px solid #f0b429;padding-left:10px;text-transform:uppercase;letter-spacing:.5px}}
.sbar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:12px 16px;text-align:center;flex:1;min-width:100px}}
.sv{{display:block;font-size:19px;font-weight:bold}}
.sl{{display:block;font-size:11px;color:#666;margin-top:2px}}
.box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#666;padding:7px 10px;text-align:left;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
td{{padding:7px 10px;border-bottom:1px solid #161616;white-space:nowrap}}
tr:hover td{{background:#1e1e1e}}
tr.ok-row td{{background:#0d1a0d}}
tr.ok-row:hover td{{background:#122212}}
.na{{color:#444;font-style:italic}}
.muted{{color:#555;font-size:10px}}
.good{{color:#4caf50}} .warn{{color:#f0b429}} .bad{{color:#f44336}}
.badge{{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:bold}}
.badge.green{{background:#1a3a1a;color:#4caf50}}
.badge.red{{background:#3a1a1a;color:#f44336}}
.chart-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px;margin-bottom:14px}}
.chart-wrap h3{{font-size:12px;color:#888;margin-bottom:10px}}
/* MC cards */
.mc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
@media(max-width:900px){{.mc-grid{{grid-template-columns:1fr}}}}
.mc-card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:18px}}
.mc-title{{font-size:14px;font-weight:bold;color:#f0b429;margin-bottom:12px}}
.mc-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}}
.mc-stat{{background:#111;border-radius:6px;padding:8px;text-align:center}}
.mc-v{{display:block;font-size:15px;font-weight:bold}}
.mc-l{{display:block;font-size:10px;color:#555;margin-top:2px}}
.gauge-row{{display:flex;align-items:center;gap:10px;margin-bottom:7px}}
.g-label{{font-size:11px;color:#888;width:200px;flex-shrink:0}}
.g-bar-wrap{{flex:1;background:#111;border-radius:4px;height:12px;overflow:hidden}}
.g-bar{{height:100%;border-radius:4px}}
.g-val{{font-size:12px;font-weight:bold;width:52px;text-align:right}}
.div{{border:none;border-top:1px solid #2a2a2a;margin:12px 0}}
.timeline .tl-row{{display:flex;justify-content:space-between;font-size:12px;padding:5px 0;border-bottom:1px solid #161616}}
.timeline .tl-row:last-child{{border-bottom:none}}
.tl-l{{color:#888}} .tl-v{{color:#e0e0e0}}
/* Active days */
.ada-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}}
@media(max-width:900px){{.ada-grid{{grid-template-columns:repeat(3,1fr)}}}}
.ada-box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px;text-align:center}}
.ada-v{{font-size:18px;font-weight:bold}}
.ada-l{{font-size:10px;color:#555;margin-top:3px}}
.info-box{{background:#0e1a2a;border:1px solid #2196f3;border-radius:8px;padding:12px 16px;margin-top:12px;font-size:12px;line-height:1.8;color:#bbdefb}}
.ok-box{{background:#0d1a0d;border:1px solid #4caf50;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:12px;line-height:1.8;color:#c8e6c9}}
</style>
</head>
<body>
<h1>Multi-Instrument Portfolio &mdash; 7 Assets, BB({BB_PERIOD}) RSI({RSI_PERIOD})&lt;{RSI_LO} EMA({EMA_PERIOD})</h1>
<p class="sub">$100K account &nbsp;|&nbsp; Commission 0.01%/order + slippage &nbsp;|&nbsp; Equal weight among qualifiers &nbsp;|&nbsp; Generated {now}</p>

<h2>Combined Portfolio ({n_qual} qualifiers: {qual_names})</h2>
<div class="sbar">
  <div class="ss"><span class="sv" style="color:{port_pf_c}">{port['pf']}</span><span class="sl">Combined PF</span></div>
  <div class="ss"><span class="sv" style="color:{_c(port['tpy'],40,20)}">{port['tpy']}/yr</span><span class="sl">Trades/Year</span></div>
  <div class="ss"><span class="sv">{round(port['tpy']/12,1)}/mo</span><span class="sl">Trades/Month</span></div>
  <div class="ss"><span class="sv" style="color:{port_dd_c}">{port['dd']}%</span><span class="sl">Max DD</span></div>
  <div class="ss"><span class="sv">{port['wr']}%</span><span class="sl">Win Rate</span></div>
  <div class="ss"><span class="sv" style="color:{"#4caf50" if port["monthly_exp"]>0 else "#f44336"}">${port["monthly_exp"]:,.0f}</span><span class="sl">Monthly Expect.</span></div>
  <div class="ss"><span class="sv" style="color:{"#4caf50" if port["annual_exp"]>0 else "#f44336"}">${port["annual_exp"]:,.0f}</span><span class="sl">Annual Expect.</span></div>
  <div class="ss"><span class="sv" style="color:{"#4caf50" if port["ret"]>0 else "#f44336"}">{port["ret"]}%</span><span class="sl">2-Year Return</span></div>
</div>

<h2>Per-Trade Statistics (combined, actual $ on $100K portfolio)</h2>
<div class="box" style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
  <div><div style="font-size:11px;color:#666">Avg profit/trade</div><div style="font-size:17px;font-weight:bold;color:{"#4caf50" if port["avg"]>0 else "#f44336"}">${port["avg"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Median</div><div style="font-size:17px;font-weight:bold">${port["med"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Avg winner</div><div style="font-size:17px;font-weight:bold;color:#4caf50">${port["avg_w"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Avg loser</div><div style="font-size:17px;font-weight:bold;color:#f44336">${port["avg_l"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Best trade</div><div style="font-size:17px;font-weight:bold;color:#4caf50">${port["best"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Worst trade</div><div style="font-size:17px;font-weight:bold;color:#f44336">${port["worst"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Std deviation</div><div style="font-size:17px;font-weight:bold">${port["std"]:,.2f}</div></div>
  <div><div style="font-size:11px;color:#666">Total trades</div><div style="font-size:17px;font-weight:bold">{port["n"]}</div></div>
</div>

<h2>Active Trading Days Per Month</h2>
<div class="ada-grid">
  <div class="ada-box"><div class="ada-v" style="color:{_c(ada["avg_active_per_month"],10,7)}">{ada["avg_active_per_month"]}</div><div class="ada-l">Avg active days/month</div></div>
  <div class="ada-box"><div class="ada-v">{ada["median_active_per_month"]}</div><div class="ada-l">Median active days/month</div></div>
  <div class="ada-box"><div class="ada-v">{ada["min_active"]}</div><div class="ada-l">Min active days (worst month)</div></div>
  <div class="ada-box"><div class="ada-v">{ada["max_active"]}</div><div class="ada-l">Max active days (best month)</div></div>
  <div class="ada-box"><div class="ada-v" style="color:{_c(ada["months_meeting_10"]/max(ada["total_months"],1)*100,80,50)}">{ada["months_meeting_10"]}/{ada["total_months"]}</div><div class="ada-l">Months with 10+ active days</div></div>
</div>
{('<div class="ok-box">Active trading days: avg <b>' + str(ada["avg_active_per_month"]) + '</b>/month. ' +
  str(ada["months_meeting_10"]) + '/' + str(ada["total_months"]) + ' months meet FTMO minimum 10-day rule. ' +
  ('Frequency constraint SOLVED.' if ada['avg_active_per_month'] >= 10 else 'Still slightly below target — consider adding 1-2 more instruments.') + '</div>')
  if ada['total_months'] > 0 else ''}

<h2>All Instruments — BB({BB_PERIOD}) RSI({RSI_PERIOD})&lt;{RSI_LO} EMA({EMA_PERIOD})</h2>
<div class="box">
  <table>
    <thead><tr>
      <th>Instrument</th><th>Status</th><th>PF</th><th>Max DD</th>
      <th>Trades/yr</th><th>Win Rate</th><th>Avg Win</th><th>Avg Loss</th>
      <th>Total trades</th><th>Data from</th><th>Bars</th>
    </tr></thead>
    <tbody>{inst_rows}</tbody>
  </table>
  <div class="info-box">Filter: PF &ge; {PF_MIN} AND Max DD &lt; {DD_MAX}%.
  Commission 0.01%/order + asset-specific slippage. No stop-loss. Size = 99% of cash per trade.</div>
</div>

<h2>Monthly P&L Distribution (Combined Portfolio, $100K)</h2>
<div class="chart-wrap">
  <h3>Monthly profit/loss — all qualified instruments, equal weight</h3>
  <canvas id="monthlyChart" style="max-height:260px"></canvas>
</div>

<h2>FTMO Phase 1 &amp; Phase 2 — Time to Target (Monte Carlo, {N_SIM:,} simulations)</h2>
<div class="mc-grid">
  {mc_card(mc1, 'p1')}
  {mc_card(mc2, 'p2')}
</div>

<script>
new Chart(document.getElementById('monthlyChart').getContext('2d'), {{
  type: 'bar',
  data: {{ labels: {months_js}, datasets: [{{
    data: {pnls_js}, backgroundColor: {mcolors_js}, borderRadius: 4
  }}]}},
  options: {{ responsive: true,
    plugins: {{ legend:{{display:false}}, tooltip:{{callbacks:{{
      label: c => '$' + c.raw.toLocaleString('en',{{minimumFractionDigits:2}})
    }}}}  }},
    scales: {{
      x:{{ticks:{{color:'#888',maxRotation:45}},grid:{{color:'#1a1a1a'}}}},
      y:{{ticks:{{color:'#888',callback:v=>'$'+v.toLocaleString()}},grid:{{color:'#222'}}}}
    }}
  }}
}});
</script>
</body>
</html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] Report: {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep = '=' * 66
    print(sep)
    print(f'  Multi-instrument portfolio: BB({BB_PERIOD}) RSI<{RSI_LO} EMA({EMA_PERIOD})')
    print(f'  Filter: PF >= {PF_MIN}  AND  DD < {DD_MAX}%')
    print(sep)

    # ── Load / download all instruments ──────────────────────────────────────
    print('\nLoading data...')
    dfs = {}
    for name, cfg in INSTRUMENTS.items():
        df = load_data(name, cfg['ticker'])
        if df is not None:
            dfs[name] = df
            print(f'  {name}: {len(df)} bars  avg={df["Close"].mean():.4f}  ({df.index[0].date()} to {df.index[-1].date()})')
        else:
            print(f'  {name}: SKIPPED')

    # ── Run backtests ──────────────────────────────────────────────────────────
    print('\nRunning backtests...')
    all_results = []
    for name, df in dfs.items():
        cfg        = INSTRUMENTS[name]
        avg_price  = float(df['Close'].mean())
        slip_frac  = cfg['tick'] * cfg['n_ticks'] / avg_price
        commission = BASE_COMM + slip_frac
        r = run_bt(name, df, commission)
        all_results.append(r)
        if r.get('error'):
            print(f'  {name}: ERROR {r["error"][:50]}')
        else:
            status = 'PASS' if r['ok'] else 'FAIL'
            print(f'  {name}: {status}  PF {r["pf"]:.3f}  DD {r["dd"]:.2f}%  '
                  f'{r["tpy"]}/yr  WR {r["wr"]}%  avg_w ${r["avg_w"]:,.0f}  avg_l ${r["avg_l"]:,.0f}')

    # ── Filter qualifiers ──────────────────────────────────────────────────────
    qual = [r for r in all_results if r.get('ok') and not r.get('error')]
    print(f'\n{sep}')
    print(f'  QUALIFIERS ({len(qual)}/{len(all_results)}): {", ".join(r["name"] for r in qual)}')
    print(sep)

    if not qual:
        print('  No instruments qualify. Exiting.')
        return

    # ── Build portfolio ───────────────────────────────────────────────────────
    print('\nBuilding combined portfolio...')
    port    = build_portfolio(qual)
    monthly = monthly_dist(port)
    ada     = active_days_analysis(port)

    print(f'  Combined: {port["n"]} trades ({port["tpy"]}/yr = {port["tpy"]/12:.1f}/mo)  '
          f'PF {port["pf"]}  WR {port["wr"]}%  DD {port["dd"]}%  Ret {port["ret"]}%')
    print(f'  Per-trade: avg ${port["avg"]:,.2f}  med ${port["med"]:,.2f}  std ${port["std"]:,.2f}')
    print(f'  Monthly expectancy: ${port["monthly_exp"]:,.2f}')
    print(f'  Annual expectancy:  ${port["annual_exp"]:,.0f}')
    print(f'  Active days/month:  {ada["avg_active_per_month"]} avg  ({ada["months_meeting_10"]}/{ada["total_months"]} months >= 10 days)')

    print(f'\n  Monthly distribution:')
    if not monthly.empty:
        for _, row in monthly.iterrows():
            sign = '+' if row['pnl'] >= 0 else '-'
            bar  = '#' * max(1, int(abs(row['pnl']) / 30))
            flag = '  <-- neg' if row['pnl'] < 0 else ''
            print(f'    {row["month_str"]}  {sign}${abs(row["pnl"]):>8,.0f}  ({int(row["n_trades"])} trades)  {bar}{flag}')

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print(f'\nMonte Carlo ({N_SIM:,} simulations)...')
    mc1 = time_to_target(port, target_usd=10_000,
                         max_dd_usd=10_000, max_daily_usd=5_000, min_tdays=10, seed=42)
    mc2 = time_to_target(port, target_usd=5_000,
                         max_dd_usd=10_000, max_daily_usd=5_000, min_tdays=10, seed=43)

    print(f'\n  Trades/month (expected): {mc1["exp_trades_month"]:.1f}')
    print(f'  Active days / 22 tdays:  {mc1["exp_tdays_22"]:.1f}  (need 10 for Phase 1)')
    print(f'  Active days / 43 tdays:  {mc1["exp_tdays_43"]:.1f}  (need 10 for Phase 2)')

    def print_mc(mc, name, target):
        print(f'\n  FTMO {name} (target ${target:,}):')
        if mc['median'] is None:
            print(f'    Target never reached  breach={mc["breach_rate"]}%')
        else:
            print(f'    Hit rate (unconstrained): {mc["hit_rate"]}%')
            print(f'    Breach rate (DD/daily):   {mc["breach_rate"]}%')
            print(f'    Median time to target:    {mc["median"]} trading days (~{mc["weeks_median"]} weeks)')
            print(f'    P25-P75:                  {mc["p25"]}-{mc["p75"]} trading days')
            print(f'    Pass within 30-tday win:  {mc["pass22"]}%  (Phase 1 window)')
            print(f'    Pass within 60-tday win:  {mc["pass43"]}%  (Phase 2 window)')

    print_mc(mc1, 'Phase 1', 10_000)
    print_mc(mc2, 'Phase 2',  5_000)

    print(f'\n{sep}')
    print('  SUMMARY')
    print(sep)
    print(f'  Qualifiers:        {len(qual)}  ({", ".join(r["name"] for r in qual)})')
    print(f'  Trades/month:      {round(port["tpy"]/12,1)}')
    print(f'  Monthly expect.:   ${port["monthly_exp"]:,.2f}')
    print(f'  Time to P1 $10K:   {mc1.get("median","N/A")} trading days  ({mc1.get("weeks_median","N/A")} weeks)  pass rate {mc1["pass22"]}%')
    print(f'  Time to P2 $5K:    {mc2.get("median","N/A")} trading days  ({mc2.get("weeks_median","N/A")} weeks)  pass rate {mc2["pass43"]}%')
    print(f'  Active days/mo:    {ada["avg_active_per_month"]} avg  '
          f'(FTMO 10-day rule: {"MET" if ada["avg_active_per_month"] >= 10 else "NOT MET"})')
    print(sep)

    generate_report(all_results, qual, port, monthly, ada, mc1, mc2)

if __name__ == '__main__':
    main()

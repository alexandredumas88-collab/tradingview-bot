#!/usr/bin/env python3
"""
Complementary breakout strategy: Donchian channel N-bar high + RSI>55 + EMA(100)
Runs alongside BB+RSI mean-reversion on all 7 instruments.
Both strategies tested per instrument — qualifiers (PF>2, DD<8) combined.
$100k account | equal weight across all qualifying (instrument, strategy) slots
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from backtesting import Backtest, Strategy

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT       = 100_000
TRADE_SIZE    = 0.99    # 99% of equity per trade (3 equal slots, full deployment)
BASE_COMM     = 0.0001
N_SIM      = 10_000

# Mean-reversion params (proven from prior runs)
MR_BB      = 10
MR_RSI_N   = 7
MR_RSI_LO  = 35
MR_EMA     = 100

# Breakout params grid
BO_DC_PERIODS = [10, 20]          # Donchian channel lengths to test
BO_RSI_HI     = 55                # RSI must confirm momentum
BO_EMA        = 100               # same trend filter
BO_RSI_N      = 7

PF_MIN = 1.5
DD_MAX = 12.0
CATA_STOP = 0.045   # 4.5% hard stop — safety net vs FTMO 5% daily-loss rule

_here      = os.path.dirname(__file__)
DATA_DIR   = os.path.join(_here, 'pydata')
REPORT_OUT = os.path.join(_here, 'pybreakout-report.html')
os.makedirs(DATA_DIR, exist_ok=True)

INSTRUMENTS = {
    'EURUSD': dict(ticker='EURUSD=X', tick=0.0001, n_ticks=1, label='EUR/USD'),
    'GBPUSD': dict(ticker='GBPUSD=X', tick=0.0001, n_ticks=1, label='GBP/USD'),
    'AUDUSD': dict(ticker='AUDUSD=X', tick=0.0001, n_ticks=1, label='AUD/USD'),
    'SP500':  dict(ticker='SPY',      tick=0.01,   n_ticks=2, label='S&P 500 (SPY)'),
    'NAS100': dict(ticker='QQQ',      tick=0.01,   n_ticks=2, label='NAS100 (QQQ)'),
    'XAUUSD': dict(ticker=None,       tick=0.10,   n_ticks=1, label='XAU/USD (Gold)'),
    'DAX':    dict(ticker='^GDAXI',   tick=1.0,    n_ticks=2, label='DAX (^GDAXI)'),
}

FTMO = dict(
    p1=dict(name='Phase 1', target=10_000, max_dd=10_000, max_daily=5_000, tdays=22, min_td=10),
    p2=dict(name='Phase 2', target=5_000,  max_dd=10_000, max_daily=5_000, tdays=43, min_td=10),
)

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

def _dc_high(arr, n):
    """Highest HIGH of previous N bars (1-bar shift prevents lookahead)."""
    return pd.Series(arr, dtype=float).shift(1).rolling(n, min_periods=n).max().values

def _dc_low(arr, n):
    """Lowest LOW of previous N bars (1-bar shift prevents lookahead)."""
    return pd.Series(arr, dtype=float).shift(1).rolling(n, min_periods=n).min().values

# ── Strategies ────────────────────────────────────────────────────────────────
class MeanRevStrategy(Strategy):
    """BB(10)+RSI(7)<35+EMA(100) mean-reversion, exits at BB midline."""
    def init(self):
        c = self.data.Close
        self.bb_up  = self.I(_bb_upper, c, MR_BB)
        self.bb_mid = self.I(_bb_mid,   c, MR_BB)
        self.bb_lo  = self.I(_bb_lower, c, MR_BB)
        self.rsi    = self.I(_rsi, c, MR_RSI_N)
        self.ema    = self.I(_ema, c, MR_EMA)

    def next(self):
        if any(np.isnan(x[-1]) for x in (self.bb_lo, self.rsi, self.ema)):
            return
        c = self.data.Close[-1]
        if not self.position:
            if c < self.bb_lo[-1] and self.rsi[-1] < MR_RSI_LO and c > self.ema[-1]:
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long and c > self.bb_mid[-1]:
            self.position.close()


class BreakoutStrategy(Strategy):
    """Donchian N-bar high breakout + RSI(7)>55 + EMA(100) trend filter.
    Long only. Trails out on Donchian N-bar low.
    """
    dc_n = 20

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.dc_hi = self.I(_dc_high, h, self.dc_n)
        self.dc_lo = self.I(_dc_low,  l, self.dc_n)
        self.rsi   = self.I(_rsi, c, BO_RSI_N)
        self.ema   = self.I(_ema, c, BO_EMA)

    def next(self):
        if any(np.isnan(x[-1]) for x in (self.dc_hi, self.dc_lo, self.rsi, self.ema)):
            return
        c = self.data.Close[-1]
        if not self.position:
            if c > self.dc_hi[-1] and self.rsi[-1] > BO_RSI_HI and c > self.ema[-1]:
                self.buy(size=TRADE_SIZE)
        elif self.position.is_long:
            if self.position.pl_pct < -CATA_STOP:
                self.position.close()
            elif c < self.dc_lo[-1]:
                self.position.close()

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(name, ticker):
    cache = os.path.join(DATA_DIR, f'{name}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) >= 300:
            return df
    if ticker is None:
        return None
    df = yf.download(ticker, period='730d', interval='1h', progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    if len(df) < 300:
        return None
    df.to_csv(cache)
    return df

# ── Runner ────────────────────────────────────────────────────────────────────
def run_bt(name, strat_label, df, strategy_cls, commission, dc_n=None):
    if dc_n is not None:
        strategy_cls.dc_n = dc_n
    try:
        bt    = Backtest(df, strategy_cls, cash=ACCOUNT,
                         commission=commission, exclusive_orders=True)
        stats = bt.run()
    except Exception as e:
        return dict(name=name, strat=strat_label, ok=False, error=str(e))

    trades = stats._trades.copy() if hasattr(stats, '_trades') and len(stats._trades) else pd.DataFrame()
    eq     = stats._equity_curve['Equity'].copy() if hasattr(stats, '_equity_curve') else None
    n      = int(stats.get('# Trades', 0))
    yrs    = max((df.index[-1] - df.index[0]).days / 365.25, 1.0)
    pf_raw = stats.get('Profit Factor', float('nan'))
    if isinstance(pf_raw, float) and math.isinf(pf_raw):  pf = 99.0
    elif isinstance(pf_raw, float) and math.isnan(pf_raw): pf = 0.0
    else: pf = float(pf_raw)

    dd  = round(abs(float(stats.get('Max. Drawdown [%]', 0))), 2)
    wr  = round(float(stats.get('Win Rate [%]', 0)), 1)
    ret = round(float(stats.get('Return [%]', 0)), 2)
    ok  = pf >= PF_MIN and dd < DD_MAX and n >= 10

    wins   = trades.loc[trades['PnL'] > 0, 'PnL'] if not trades.empty else pd.Series()
    losses = trades.loc[trades['PnL'] < 0, 'PnL'] if not trades.empty else pd.Series()

    return dict(
        name=name, strat=strat_label, ok=ok, error=None,
        n=n, tpy=round(n/yrs, 1), pf=round(pf, 3),
        wr=wr, dd=dd, ret=ret, years=yrs,
        avg_w=round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_l=round(float(losses.mean()), 2) if len(losses) else 0.0,
        trades=trades, eq=eq,
        bars=len(df), date_from=df.index[0].date(),
    )

# ── Portfolio ─────────────────────────────────────────────────────────────────
def build_portfolio(slots):
    """slots: list of result dicts (each is one instrument+strategy qualifier)."""
    n     = len(slots)
    scale = 1.0 / n
    all_pnl, all_trades_rows, eq_combined = [], [], None

    for r in slots:
        if r['trades'].empty:
            continue
        pnl_scaled = [p * scale for p in r['trades']['PnL'].tolist()]
        all_pnl   += pnl_scaled

        if 'ExitTime' in r['trades'].columns:
            tdf = r['trades'][['ExitTime','PnL']].copy()
            tdf['PnL']   = tdf['PnL'] * scale
            tdf['asset'] = r['name']
            tdf['strat'] = r['strat']
            all_trades_rows.append(tdf)

        if r['eq'] is not None:
            eq_s = r['eq'] * scale
            eq_combined = eq_s if eq_combined is None else (
                eq_combined.reindex(eq_combined.index.union(eq_s.index), method='ffill')
                + eq_s.reindex(eq_combined.index.union(eq_s.index), method='ffill'))

    all_trades_df = pd.concat(all_trades_rows, ignore_index=True) if all_trades_rows else pd.DataFrame()

    gp  = sum(p for p in all_pnl if p > 0)
    gl  = abs(sum(p for p in all_pnl if p < 0))
    pf  = round(gp / gl, 3) if gl > 0 else 99.0
    nt  = len(all_pnl)
    wr  = round(sum(1 for p in all_pnl if p > 0) / nt * 100, 1) if nt else 0.0
    yrs = max(r['years'] for r in slots)
    tpy = round(sum(r['n'] for r in slots) / yrs, 1)

    max_dd = combined_ret = float('nan')
    if eq_combined is not None:
        peak    = eq_combined.cummax()
        max_dd  = round(float(((peak - eq_combined) / peak * 100).max()), 2)
        combined_ret = round(float((eq_combined.iloc[-1] / eq_combined.iloc[0] - 1) * 100), 2)

    wins_a   = [p for p in all_pnl if p > 0]
    losses_a = [p for p in all_pnl if p < 0]
    avg      = round(float(np.mean(all_pnl)), 2) if all_pnl else 0.0

    mr_n  = sum(r['n'] for r in slots if r['strat'] == 'MeanRev')
    bo_n  = sum(r['n'] for r in slots if r['strat'].startswith('Breakout'))
    mr_tpy = round(mr_n / yrs, 1)
    bo_tpy = round(bo_n / yrs, 1)

    return dict(
        n=nt, tpy=tpy, tpm=round(tpy/12, 1), pf=pf, wr=wr,
        dd=max_dd, ret=combined_ret, years=yrs,
        all_pnl=all_pnl, all_trades=all_trades_df, eq=eq_combined,
        avg=avg,
        med=round(float(np.median(all_pnl)), 2) if all_pnl else 0.0,
        std=round(float(np.std(all_pnl)), 2) if all_pnl else 0.0,
        best=round(float(max(all_pnl)), 2) if all_pnl else 0.0,
        worst=round(float(min(all_pnl)), 2) if all_pnl else 0.0,
        avg_w=round(float(np.mean(wins_a)), 2) if wins_a else 0.0,
        avg_l=round(float(np.mean(losses_a)), 2) if losses_a else 0.0,
        monthly_exp=round(avg * tpy / 12, 2),
        annual_exp=round(avg * nt / yrs, 0),
        mr_tpy=mr_tpy, bo_tpy=bo_tpy, n_slots=n,
    )

# ── Monthly distribution ──────────────────────────────────────────────────────
def monthly_dist(port):
    df = port['all_trades']
    if df.empty or 'ExitTime' not in df.columns:
        return pd.DataFrame(columns=['month_str','pnl','n_trades'])
    df = df.copy()
    df['month'] = pd.to_datetime(df['ExitTime']).dt.to_period('M')
    grp = df.groupby('month').agg(pnl=('PnL','sum'), n_trades=('PnL','count')).reset_index()
    grp['month_str'] = grp['month'].astype(str)
    return grp

def monthly_by_strat(port):
    df = port['all_trades']
    if df.empty or 'ExitTime' not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df['month']    = pd.to_datetime(df['ExitTime']).dt.to_period('M')
    df['is_bo']    = df['strat'].str.startswith('Breakout').astype(int)
    grp = df.groupby(['month','is_bo'])['PnL'].sum().unstack(fill_value=0).reset_index()
    grp.columns    = ['month', 'mr_pnl', 'bo_pnl'] if 0 in grp.columns else ['month', 'bo_pnl']
    if 'mr_pnl' not in grp.columns: grp['mr_pnl'] = 0.0
    if 'bo_pnl' not in grp.columns: grp['bo_pnl'] = 0.0
    grp['month_str'] = grp['month'].astype(str)
    return grp

# ── Active-days analysis ──────────────────────────────────────────────────────
def active_days_analysis(port):
    df = port['all_trades']
    if df.empty or 'ExitTime' not in df.columns:
        return dict(avg=0, med=0, mn=0, mx=0, ok_months=0, total_months=1)
    df = df.copy()
    df['date']  = pd.to_datetime(df['ExitTime']).dt.date
    df['month'] = pd.to_datetime(df['ExitTime']).dt.to_period('M')
    per_mo = df.groupby('month')['date'].nunique()
    return dict(
        avg=round(float(per_mo.mean()), 1),
        med=round(float(per_mo.median()), 1),
        mn=int(per_mo.min()),
        mx=int(per_mo.max()),
        ok_months=int((per_mo >= 10).sum()),
        total_months=len(per_mo),
    )

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def time_to_target(port, target_usd, max_dd_usd, max_daily_usd,
                   min_tdays=10, seed=42, swing=False, size=None, max_days=None):
    """Standard mode: tracks pass within FTMO 22/43-tday windows + active-day rule.
    Swing mode: no time limit, no active-day rule — pass = reach target before breach.
    size: if set, scales PnL from the 0.99-sizing backtest to a new sizing fraction.
    max_days: overrides the default sim horizon (swing=2000, standard=730).
    """
    scale    = (size / TRADE_SIZE) if size is not None else 1.0
    if max_days is None:
        max_days = 2000 if swing else 730
    MAX_DAYS = max_days
    pnl_arr  = np.array(port['all_pnl'], dtype=float) * scale
    rng      = np.random.default_rng(seed)
    df       = port['all_trades']
    if not df.empty and 'ExitTime' in df.columns:
        dates = pd.to_datetime(df['ExitTime']).tolist()
        bdays = pd.bdate_range(min(dates), max(dates))
        lam   = len(pnl_arr) / max(len(bdays), 1)
    else:
        lam = 0.5

    days_hit = []
    breached = 0
    pass22 = pass43 = 0

    for _ in range(N_SIM):
        equity  = ACCOUNT
        cum     = 0.0
        hit_day = None
        blown   = False

        for day in range(1, MAX_DAYS + 1):
            n_today = int(rng.poisson(lam))
            day_pnl = 0.0
            for __ in range(n_today):
                p = float(pnl_arr[rng.integers(len(pnl_arr))])
                day_pnl += p; cum += p; equity += p

            if day_pnl < -max_daily_usd or (ACCOUNT - equity) > max_dd_usd:
                blown = True; breached += 1; break
            if cum >= target_usd and hit_day is None:
                hit_day = day
                if swing:
                    break   # target reached — stop this simulation

        if blown:
            continue
        if hit_day is not None:
            days_hit.append(hit_day)
            if not swing:
                act = round((1 - math.exp(-lam)) * hit_day)
                if hit_day <= 22 and act >= min_tdays: pass22 += 1
                if hit_day <= 43 and act >= min_tdays: pass43 += 1

    exp22 = round((1 - math.exp(-lam)) * 22, 1)
    exp43 = round((1 - math.exp(-lam)) * 43, 1)

    base = dict(
        lam=round(lam, 4),
        exp_tpm=round(lam * 21, 1),
        scale=round(scale, 6),
        breach=round(breached / N_SIM * 100, 1),
        hit=round(len(days_hit) / N_SIM * 100, 1),
        p25=int(np.percentile(days_hit, 25)) if days_hit else None,
        med=int(np.median(days_hit))          if days_hit else None,
        mean=int(np.mean(days_hit))           if days_hit else None,
        p75=int(np.percentile(days_hit, 75)) if days_hit else None,
        wmed=round(int(np.median(days_hit))           * 7/5, 1) if days_hit else None,
        wp25=round(int(np.percentile(days_hit, 25))   * 7/5, 1) if days_hit else None,
        wp75=round(int(np.percentile(days_hit, 75))   * 7/5, 1) if days_hit else None,
    )
    if swing:
        return base   # pass rate = hit rate for swing (no time window)
    base.update(
        exp_td22=exp22, exp_td43=exp43,
        pass22=round(pass22 / N_SIM * 100, 1),
        pass43=round(pass43 / N_SIM * 100, 1),
    )
    return base

# ── HTML ──────────────────────────────────────────────────────────────────────
def _c(v, ok, warn, hi=True):
    if hi: return '#4caf50' if v >= ok else ('#f0b429' if v >= warn else '#f44336')
    else:  return '#4caf50' if v <= ok else ('#f0b429' if v <= warn else '#f44336')

def generate_report(all_mr, all_bo_best, qual_slots, port, monthly, mbs, ada, mc1, mc2, sw1, sw2):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── Instrument comparison table
    instr_names = list(INSTRUMENTS.keys())
    mr_by_name  = {r['name']: r for r in all_mr}
    bo_by_name  = {r['name']: r for r in all_bo_best}
    qual_keys   = {(r['name'], r['strat']) for r in qual_slots}

    tbl_rows = ''
    for nm in instr_names:
        mr = mr_by_name.get(nm)
        bo = bo_by_name.get(nm)
        def cell(r, strat_key):
            if not r or r.get('error'):
                return '<td colspan="4" class="na">—</td>'
            ok_mark = ' <span class="badge green">IN</span>' if (r['name'], strat_key) in qual_keys else ''
            pf_c = _c(r['pf'], PF_MIN, 1.2)
            dd_c = _c(r['dd'], 8.0, DD_MAX, False)
            return (f'<td style="color:{pf_c}">{r["pf"]:.3f}{ok_mark}</td>'
                    f'<td style="color:{dd_c}">{r["dd"]:.2f}%</td>'
                    f'<td>{r["tpy"]}/yr</td>'
                    f'<td>{r["wr"]}%</td>')
        tbl_rows += (f'<tr><td><b>{nm}</b><br>'
                     f'<small class="muted">{INSTRUMENTS[nm]["label"]}</small></td>'
                     + cell(mr, 'MeanRev')
                     + cell(bo, bo["strat"] if bo else "")
                     + '</tr>')

    # ── Monthly stacked bar data
    months_js = json.dumps(monthly['month_str'].tolist() if not monthly.empty else [])
    mr_pnl_js = json.dumps([round(float(v), 2) for v in mbs['mr_pnl'].tolist()] if not mbs.empty else [])
    bo_pnl_js = json.dumps([round(float(v), 2) for v in mbs['bo_pnl'].tolist()] if not mbs.empty else [])

    # ── Gauge helper
    def gauge(label, val, color):
        w = min(float(val or 0), 100)
        return (f'<div class="g-row"><span class="g-l">{label}</span>'
                f'<div class="g-wrap"><div class="g-bar" style="width:{w}%;background:{color}"></div></div>'
                f'<span class="g-v" style="color:{color}">{val}%</span></div>')

    def swing_block(sw, title, target_usd):
        pct      = 10 if target_usd == 10_000 else 5
        pass_c   = _c(sw['hit'], 90, 60)
        breach_c = '#4caf50' if sw['breach'] < 1 else ('#f0b429' if sw['breach'] < 5 else '#f44336')
        wks_med  = round(sw['med'] / 5, 1)  if sw['med']  else None
        wks_p25  = round(sw['p25'] / 5, 1)  if sw['p25']  else None
        wks_p75  = round(sw['p75'] / 5, 1)  if sw['p75']  else None
        mo_med   = round(sw['med'] / 21, 1) if sw['med']  else None
        return f'''<div class="mc-box">
  <div class="mc-title">{title} — ${target_usd:,} (+{pct}%)</div>
  <div class="swing-note">No time limit &nbsp;·&nbsp; No minimum active-trading-day requirement</div>
  <div class="mc-row"><span>Account</span><span>$100,000</span></div>
  <div class="mc-row"><span>Max DD limit</span><span>$10,000 (10%)</span></div>
  <div class="mc-row"><span>Max daily loss limit</span><span>$5,000 (5%)</span></div>
  <div class="mc-row"><span>Expected trades / month</span><span><b>{sw["exp_tpm"]:.1f}</b></span></div>
  <hr class="sep">
  {gauge("Pass rate — reach target before breach", sw["hit"], pass_c)}
  {gauge("Breach rate — DD or daily-loss triggered", sw["breach"] if sw["breach"] > 0 else 0.1, breach_c)}
  <hr class="sep">
  <div class="mc-row"><span>Median trading days to +{pct}%</span>
    <span><b>{sw["med"] if sw["med"] else "N/A"}</b> tdays</span></div>
  <div class="mc-row"><span>≈ Calendar time (median)</span>
    <span><b>{wks_med}</b> weeks &nbsp;≈&nbsp; <b>{mo_med}</b> months</span></div>
  <div class="mc-row"><span>P25 – P75 trading days</span>
    <span>{sw["p25"]} – {sw["p75"]} tdays</span></div>
  <div class="mc-row"><span>P25 – P75 calendar weeks</span>
    <span>{wks_p25} – {wks_p75} weeks</span></div>
  <div class="mc-row"><span>Mean trading days to target</span>
    <span>{sw["mean"]} tdays ({round(sw["mean"]/21,1)} months)</span></div>
</div>'''

    port_pf_c  = _c(port['pf'],  PF_MIN, 1.5)
    port_dd_c  = _c(port['dd'],  4.0, DD_MAX, False)
    port_tpm_c = _c(port['tpm'], 10, 5)
    ada_c      = _c(ada['avg'],  10, 7)

    n_mr_slots = sum(1 for r in qual_slots if r['strat'] == 'MeanRev')
    n_bo_slots = sum(1 for r in qual_slots if r['strat'].startswith('Breakout'))
    qual_desc  = ' + '.join(f'{r["name"]}({r["strat"][:2]})' for r in qual_slots)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FTMO Swing — MR + Breakout Portfolio</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1500px;margin:0 auto}}
h1{{font-size:22px;color:#fff;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:20px}}
h2{{font-size:13px;color:#fff;margin:22px 0 10px;border-left:3px solid #f0b429;padding-left:10px;text-transform:uppercase;letter-spacing:.6px}}
.sbar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}}
.ss{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:11px 15px;text-align:center;flex:1;min-width:90px}}
.sv{{display:block;font-size:18px;font-weight:bold}} .sl{{display:block;font-size:10px;color:#666;margin-top:2px}}
.box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:14px;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#555;padding:7px 10px;text-align:left;border-bottom:1px solid #2a2a2a}}
td{{padding:7px 10px;border-bottom:1px solid #161616}}
tr:hover td{{background:#1e1e1e}}
.na{{color:#333;font-style:italic}}
.muted{{color:#444;font-size:10px}}
.good{{color:#4caf50}} .bad{{color:#f44336}}
.badge{{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:bold}}
.badge.green{{background:#1a3a1a;color:#4caf50}}
.badge.orange{{background:#3a2a0a;color:#f0b429}}
.chart-wrap{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:14px;margin-bottom:12px}}
.chart-wrap h3{{font-size:11px;color:#777;margin-bottom:10px}}
/* MC grid */
.mc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
@media(max-width:800px){{.mc-grid{{grid-template-columns:1fr}}}}
.mc-box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:16px}}
.mc-title{{font-size:13px;font-weight:bold;color:#f0b429;margin-bottom:12px}}
.mc-row{{display:flex;justify-content:space-between;font-size:12px;padding:4px 0;border-bottom:1px solid #161616;color:#aaa}}
.mc-row span:last-child{{color:#e0e0e0}} .mc-row:last-child{{border:none}}
.sep{{border:none;border-top:1px solid #2a2a2a;margin:10px 0}}
.g-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.g-l{{font-size:11px;color:#777;width:220px;flex-shrink:0}}
.g-wrap{{flex:1;background:#111;border-radius:3px;height:11px;overflow:hidden}}
.g-bar{{height:100%;border-radius:3px}}
.g-v{{font-size:11px;font-weight:bold;width:44px;text-align:right}}
/* Strategy comparison table */
.strat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}}
@media(max-width:900px){{.strat-grid{{grid-template-columns:1fr}}}}
.strat-box{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:12px}}
.strat-title{{font-size:12px;color:#f0b429;font-weight:bold;margin-bottom:8px}}
/* Ada */
.ada-row{{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:5px 0;border-bottom:1px solid #161616}}
.ada-row:last-child{{border:none}}
.green-box{{background:#0d1a0d;border:1px solid #4caf50;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px;line-height:1.8;color:#c8e6c9}}
.warn-box{{background:#2a1a0a;border:1px solid #f0b429;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px;line-height:1.8;color:#fff3cc}}
.swing-note{{font-size:11px;color:#f0b429;background:#1e1800;border:1px solid #3a3000;border-radius:6px;padding:5px 10px;margin-bottom:10px}}
.swing-hero{{display:flex;gap:16px;margin-bottom:16px}}
.sw-card{{flex:1;background:#0d1a0d;border:1px solid #2a4a2a;border-radius:12px;padding:16px;text-align:center}}
.sw-card .big{{font-size:36px;font-weight:bold;color:#4caf50;display:block;line-height:1}}
.sw-card .lbl{{font-size:11px;color:#666;margin-top:6px;display:block}}
.sw-card .sub2{{font-size:13px;color:#aaa;margin-top:4px;display:block}}
</style>
</head>
<body>
<h1>FTMO Swing Account — MR + Donchian Breakout Portfolio</h1>
<p class="sub">
  MeanRev: BB({MR_BB})+RSI({MR_RSI_N})&lt;{MR_RSI_LO}+EMA({MR_EMA}) (no stop) &nbsp;|&nbsp;
  Breakout: Donchian+RSI&gt;{BO_RSI_HI}+EMA({BO_EMA}) ({CATA_STOP*100:.1f}% catastrophe stop) &nbsp;|&nbsp;
  {len(qual_slots)} slots ({n_mr_slots} MR, {n_bo_slots} BO) &nbsp;·&nbsp;
  size=0.99 per slot &nbsp;·&nbsp; $100K &nbsp;·&nbsp; Generated {now}
</p>

<h2>Combined Portfolio ({qual_desc})</h2>
<div class="sbar">
  <div class="ss"><span class="sv" style="color:{port_pf_c}">{port['pf']}</span><span class="sl">Combined PF</span></div>
  <div class="ss"><span class="sv" style="color:{port_tpm_c}">{port['tpm']}/mo</span><span class="sl">Trades/Month</span></div>
  <div class="ss"><span class="sv" style="color:{port_tpm_c}">{port['tpy']}/yr</span><span class="sl">Trades/Year</span></div>
  <div class="ss"><span class="sv" style="color:{port_dd_c}">{port['dd']}%</span><span class="sl">Max DD</span></div>
  <div class="ss"><span class="sv">{port['wr']}%</span><span class="sl">Win Rate</span></div>
  <div class="ss"><span class="sv" style="color:{"#4caf50" if port["monthly_exp"]>0 else "#f44336"}">${port["monthly_exp"]:,.0f}</span><span class="sl">Monthly Expect.</span></div>
  <div class="ss"><span class="sv">${port["annual_exp"]:,.0f}</span><span class="sl">Annual Expect.</span></div>
  <div class="ss"><span class="sv" style="color:{"#4caf50" if port["ret"]>0 else "#f44336"}">{port["ret"]}%</span><span class="sl">2-Year Return</span></div>
</div>

<div class="strat-grid">
  <div class="strat-box">
    <div class="strat-title">Mean-Reversion Contribution</div>
    <div class="ada-row"><span>Trades/year</span><span><b>{port["mr_tpy"]}/yr</b></span></div>
    <div class="ada-row"><span>Slots in portfolio</span><span>{n_mr_slots}</span></div>
    <div class="ada-row"><span>Strategy</span><span>BB({MR_BB}) RSI&lt;{MR_RSI_LO} EMA({MR_EMA})</span></div>
  </div>
  <div class="strat-box">
    <div class="strat-title">Breakout Contribution</div>
    <div class="ada-row"><span>Trades/year</span><span><b>{port["bo_tpy"]}/yr</b></span></div>
    <div class="ada-row"><span>Slots in portfolio</span><span>{n_bo_slots}</span></div>
    <div class="ada-row"><span>Strategy</span><span>Donchian RSI&gt;{BO_RSI_HI} EMA({BO_EMA})</span></div>
  </div>
  <div class="strat-box">
    <div class="strat-title">Per-Trade Statistics ($)</div>
    <div class="ada-row"><span>Avg profit/trade</span><span style="color:{"#4caf50" if port["avg"]>0 else "#f44336"}"><b>${port["avg"]:,.2f}</b></span></div>
    <div class="ada-row"><span>Median</span><span>${port["med"]:,.2f}</span></div>
    <div class="ada-row"><span>Avg win / avg loss</span><span class="good">${port["avg_w"]:,.0f}</span> / <span class="bad">${port["avg_l"]:,.0f}</span></div>
    <div class="ada-row"><span>Best / worst trade</span><span class="good">${port["best"]:,.0f}</span> / <span class="bad">${port["worst"]:,.0f}</span></div>
  </div>
</div>

<h2>Active Trading Days (FTMO 10-Day Minimum)</h2>
<div class="box">
  <div class="ada-row"><span>Avg active trading days / month</span>
    <span style="color:{ada_c}"><b>{ada["avg"]}</b></span></div>
  <div class="ada-row"><span>Median active days / month</span><span>{ada["med"]}</span></div>
  <div class="ada-row"><span>Range (worst – best month)</span><span>{ada["mn"]} – {ada["mx"]} days</span></div>
  <div class="ada-row"><span>Months meeting 10-day rule</span>
    <span style="color:{_c(ada["ok_months"]/max(ada["total_months"],1)*100,80,50)}">
      <b>{ada["ok_months"]}/{ada["total_months"]}</b></span></div>
</div>
{"<div class='green-box'>Active day constraint <b>SOLVED</b>: " + str(ada['avg']) + " avg active days/month, " + str(ada['ok_months']) + "/" + str(ada['total_months']) + " months meet FTMO 10-day minimum.</div>"
 if ada['avg'] >= 10 else
 "<div class='warn-box'>Avg active days " + str(ada['avg']) + "/month — still below 10. Consider adding 2–3 more breakout instruments.</div>"}

<h2>All Instruments — Both Strategies</h2>
<div class="box">
  <table>
    <thead><tr>
      <th rowspan="2">Instrument</th>
      <th colspan="4" style="text-align:center;border-left:1px solid #333">Mean-Reversion BB({MR_BB})+RSI&lt;{MR_RSI_LO}+EMA({MR_EMA})</th>
      <th colspan="4" style="text-align:center;border-left:1px solid #333">Breakout Donchian+RSI&gt;{BO_RSI_HI}+EMA({BO_EMA})</th>
    </tr><tr>
      <th style="border-left:1px solid #333">PF</th><th>DD</th><th>Freq</th><th>WR</th>
      <th style="border-left:1px solid #333">PF</th><th>DD</th><th>Freq</th><th>WR</th>
    </tr></thead>
    <tbody>{tbl_rows}</tbody>
  </table>
  <small style="color:#444;display:block;margin-top:8px">
    <span class="badge green">IN</span> = included in combined portfolio (PF&ge;{PF_MIN}, DD&lt;{DD_MAX}%, trades&ge;10)
  </small>
</div>

<h2>Monthly P&L — Mean-Reversion vs Breakout</h2>
<div class="chart-wrap">
  <h3>Stacked monthly P&L by strategy (combined portfolio, $100K account)</h3>
  <canvas id="stackedChart" style="max-height:260px"></canvas>
</div>

<h2>FTMO Swing Account — Monte Carlo Results (10,000 simulations)</h2>
<div class="swing-hero">
  <div class="sw-card">
    <span class="big">{sw1["hit"]}%</span>
    <span class="lbl">Phase 1 pass rate</span>
    <span class="sub2">Reach +$10,000 before breach</span>
  </div>
  <div class="sw-card">
    <span class="big">{round(sw1["med"]/5,1) if sw1["med"] else "—"}</span>
    <span class="lbl">weeks (median) to +$10K</span>
    <span class="sub2">≈ {round(sw1["med"]/21,1) if sw1["med"] else "—"} months &nbsp;|&nbsp; P25–P75: {round(sw1["p25"]/5,1) if sw1["p25"] else "—"}–{round(sw1["p75"]/5,1) if sw1["p75"] else "—"} wks</span>
  </div>
  <div class="sw-card">
    <span class="big">{sw2["hit"]}%</span>
    <span class="lbl">Phase 2 pass rate</span>
    <span class="sub2">Reach +$5,000 before breach</span>
  </div>
  <div class="sw-card">
    <span class="big">{round(sw2["med"]/5,1) if sw2["med"] else "—"}</span>
    <span class="lbl">weeks (median) to +$5K</span>
    <span class="sub2">≈ {round(sw2["med"]/21,1) if sw2["med"] else "—"} months &nbsp;|&nbsp; P25–P75: {round(sw2["p25"]/5,1) if sw2["p25"] else "—"}–{round(sw2["p75"]/5,1) if sw2["p75"] else "—"} wks</span>
  </div>
</div>
<div class="mc-grid">
  {swing_block(sw1, "Swing Phase 1", 10_000)}
  {swing_block(sw2, "Swing Phase 2",  5_000)}
</div>

<script>
new Chart(document.getElementById('stackedChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: {months_js},
    datasets: [
      {{ label: 'Mean-Reversion', data: {mr_pnl_js}, backgroundColor: 'rgba(33,150,243,0.7)', borderRadius:3 }},
      {{ label: 'Breakout',       data: {bo_pnl_js}, backgroundColor: 'rgba(76,175,80,0.7)',  borderRadius:3 }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color:'#888' }} }},
      tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': $' + c.raw.toLocaleString('en',{{minimumFractionDigits:2}}) }} }}
    }},
    scales: {{
      x: {{ stacked:true, ticks:{{color:'#888',maxRotation:45}}, grid:{{color:'#1a1a1a'}} }},
      y: {{ stacked:false, ticks:{{color:'#888',callback:v=>'$'+v.toLocaleString()}}, grid:{{color:'#222'}} }}
    }}
  }}
}});
</script>
</body></html>'''

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[INFO] Report: {REPORT_OUT}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep = '=' * 68
    print(sep)
    print('  Dual-strategy test: Mean-Reversion + Donchian Breakout')
    print(f'  MR: BB({MR_BB})+RSI<{MR_RSI_LO}+EMA({MR_EMA})  |  BO: DC({BO_DC_PERIODS})+RSI>{BO_RSI_HI}+EMA({BO_EMA})')
    print(sep)

    # ── Load data
    print('\nLoading data...')
    dfs = {}
    for name, cfg in INSTRUMENTS.items():
        df = load_data(name, cfg['ticker'])
        if df is not None:
            dfs[name] = df
            print(f'  {name}: {len(df)} bars  avg={df["Close"].mean():.4f}')
        else:
            print(f'  {name}: SKIPPED (no data)')

    # ── Commission per instrument
    comms = {}
    for name, df in dfs.items():
        cfg     = INSTRUMENTS[name]
        avg_p   = float(df['Close'].mean())
        comms[name] = BASE_COMM + cfg['tick'] * cfg['n_ticks'] / avg_p

    # ── Run mean-reversion
    print('\n--- Mean-Reversion ---')
    all_mr = []
    for name, df in dfs.items():
        r = run_bt(name, 'MeanRev', df, MeanRevStrategy, comms[name])
        all_mr.append(r)
        status = 'PASS' if r.get('ok') else 'FAIL'
        if r.get('error'):
            print(f'  {name} MR: ERROR')
        else:
            print(f'  {name} MR: {status}  PF {r["pf"]:.3f}  DD {r["dd"]:.2f}%  {r["tpy"]}/yr  WR {r["wr"]}%')

    # ── Run breakout for each DC period, keep best qualifier per instrument
    print('\n--- Breakout (Donchian) ---')
    all_bo_best = []   # one entry per instrument (best DC period)
    all_bo_all  = []   # all dc×instrument combos for display

    for name, df in dfs.items():
        best = None
        for dc in BO_DC_PERIODS:
            r = run_bt(name, f'Breakout DC({dc})', df, BreakoutStrategy, comms[name], dc_n=dc)
            all_bo_all.append(r)
            if r.get('error'):
                continue
            if r.get('ok'):
                if best is None or r['pf'] > best['pf']:
                    best = r
            elif best is None:
                if best is None or r['pf'] > best.get('pf', 0):
                    best = r
            status = 'PASS' if r.get('ok') else 'FAIL'
            print(f'  {name} DC({dc:2d}): {status}  PF {r["pf"]:.3f}  DD {r["dd"]:.2f}%  {r["tpy"]}/yr  WR {r["wr"]}%')

        if best:
            all_bo_best.append(best)

    # ── Collect qualifying slots (no double-allocation: one slot per instrument)
    qual_slots = []
    for r in all_mr:
        if r.get('ok') and not r.get('error'):
            qual_slots.append(r)

    # For breakout: only add if instrument NOT already covered by MR qualifier
    mr_qual_names = {r['name'] for r in qual_slots}
    for r in all_bo_best:
        if r.get('ok') and not r.get('error'):
            if r['name'] not in mr_qual_names:
                qual_slots.append(r)
            else:
                # Both MR and BO qualify on same instrument → keep better PF
                existing = next(x for x in qual_slots if x['name'] == r['name'])
                if r['pf'] > existing['pf']:
                    qual_slots.remove(existing)
                    qual_slots.append(r)

    print(f'\n{sep}')
    print(f'  QUALIFIERS ({len(qual_slots)} slots):')
    for r in qual_slots:
        print(f'    {r["name"]:8s}  {r["strat"]:<22}  PF {r["pf"]:.3f}  DD {r["dd"]:.2f}%  {r["tpy"]}/yr')

    if not qual_slots:
        print('  No qualifiers. Exiting.')
        return

    # ── Build portfolio
    print('\nBuilding combined portfolio...')
    port    = build_portfolio(qual_slots)
    monthly = monthly_dist(port)
    mbs     = monthly_by_strat(port)
    ada     = active_days_analysis(port)

    print(f'  {port["n"]} trades ({port["tpy"]}/yr = {port["tpm"]}/mo)  '
          f'PF {port["pf"]}  WR {port["wr"]}%  DD {port["dd"]}%  Ret {port["ret"]}%')
    print(f'  MR: {port["mr_tpy"]}/yr  |  Breakout: {port["bo_tpy"]}/yr')
    print(f'  Per-trade: avg ${port["avg"]:,.2f}  med ${port["med"]:,.2f}  std ${port["std"]:,.2f}')
    print(f'  Monthly expectancy: ${port["monthly_exp"]:,.2f}')
    print(f'  Annual expectancy:  ${port["annual_exp"]:,.0f}')
    print(f'  Active days/month:  avg {ada["avg"]}  ({ada["ok_months"]}/{ada["total_months"]} months >= 10 days)')

    print('\n  Monthly distribution:')
    if not monthly.empty:
        for _, row in monthly.iterrows():
            sign = '+' if row['pnl'] >= 0 else '-'
            bar  = '#' * max(1, int(abs(row['pnl']) / 40))
            flag = '  <-- neg' if row['pnl'] < 0 else ''
            print(f'    {row["month_str"]}  {sign}${abs(row["pnl"]):>8,.0f}  ({int(row["n_trades"])} trades)  {bar}{flag}')

    # ── Swing Monte Carlo: baseline 0.99x + two sized scenarios ─────────────────
    print(f'\nMonte Carlo ({N_SIM:,} sims) — FTMO SWING (no time limit, no active-day rule)...')

    # Baseline (existing 0.99x backtest)
    sw1 = time_to_target(port, 10_000, 10_000, 5_000, swing=True, seed=42)
    sw2 = time_to_target(port,  5_000, 10_000, 5_000, swing=True, seed=43)

    # 1.5% sizing — Phase 1 target.  Expected time >> 2000 days; use 5000-day window.
    sw_p1_015 = time_to_target(port, 10_000, 10_000, 5_000, swing=True,
                                seed=42, size=0.015, max_days=5_000)
    # 1.0% sizing — Phase 2 target.  Same extended window.
    sw_p2_010 = time_to_target(port,  5_000, 10_000, 5_000, swing=True,
                                seed=43, size=0.010, max_days=5_000)

    # Also run standard timed MC for HTML report blocks (not printed here)
    mc1 = time_to_target(port, 10_000, 10_000, 5_000, min_tdays=10, seed=42)
    mc2 = time_to_target(port,  5_000, 10_000, 5_000, min_tdays=10, seed=43)

    def sw_row(sw, target_usd, window_days):
        """Format one sizing scenario row."""
        mo_exp   = round(port['monthly_exp'] * sw['scale'], 2)
        anal_mo  = round(target_usd / mo_exp, 1) if mo_exp > 0 else float('inf')
        if sw['med']:
            mc_mo  = round(sw['med'] / 21, 1)
            mc_rng = f"{round(sw['p25']/21,1)}–{round(sw['p75']/21,1)}"
        else:
            mc_mo  = f'>{round(window_days/21,0):.0f} (not reached in {window_days} tdays)'
            mc_rng = '—'
        return mo_exp, anal_mo, mc_mo, mc_rng

    print(f'\n  Account: $100,000  |  Max DD: $10,000 (10%)  |  Max daily loss: $5,000 (5%)')
    print(f'  Simulation window: 0.99x=2000 tdays | sized scenarios=5000 tdays\n')

    hdr = f'  {"Scenario":<30} {"Monthly exp":>12} {"Breach%":>8} {"MC pass%":>9} {"Analytical":>12} {"MC median":>12} {"MC P25–P75":>16}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    # Baseline 0.99x: Phase 1
    me, am, mm, mr = sw_row(sw1, 10_000, 2000)
    print(f'  {"Phase 1 ($10k)  size=0.99":30} ${me:>11,.0f} {sw1["breach"]:>7.1f}% {sw1["hit"]:>8.1f}%'
          f' {am:>10.1f} mo {mm!s:>10} mo {mr:>16}')

    # 1.5% sizing: Phase 1
    me, am, mm, mr = sw_row(sw_p1_015, 10_000, 5000)
    print(f'  {"Phase 1 ($10k)  size=1.5%":30} ${me:>11,.2f} {sw_p1_015["breach"]:>7.1f}% {sw_p1_015["hit"]:>8.1f}%'
          f' {am:>10.1f} mo {mm!s:>10} mo {mr:>16}')

    # Baseline 0.99x: Phase 2
    me, am, mm, mr = sw_row(sw2, 5_000, 2000)
    print(f'  {"Phase 2 ($5k)   size=0.99":30} ${me:>11,.0f} {sw2["breach"]:>7.1f}% {sw2["hit"]:>8.1f}%'
          f' {am:>10.1f} mo {mm!s:>10} mo {mr:>16}')

    # 1.0% sizing: Phase 2
    me, am, mm, mr = sw_row(sw_p2_010, 5_000, 5000)
    print(f'  {"Phase 2 ($5k)   size=1.0%":30} ${me:>11,.2f} {sw_p2_010["breach"]:>7.1f}% {sw_p2_010["hit"]:>8.1f}%'
          f' {am:>10.1f} mo {mm!s:>10} mo {mr:>16}')

    print(f'\n  Notes:')
    print(f'    Analytical months = target / monthly_expectancy (deterministic, ignores variance)')
    print(f'    MC pass% = fraction of {N_SIM:,} sims reaching target within the simulation window')
    print(f'    Breach% ≈ 0 at small sizes: tiny positions cannot reach $5k daily or $10k DD limits')
    print(f'    Unlimited-time pass rate at positive expectancy = 100% (given enough time)')

    print(f'\n{sep}')
    print('  FINAL PORTFOLIO SUMMARY')
    print(sep)
    print(f'  Qualifiers:         {len(qual_slots)} slots  ({", ".join(r["name"] + "(" + r["strat"][:2] + ")" for r in qual_slots)})')
    print(f'  Sizing:             {TRADE_SIZE} (99% equity per slot, 3 slots = 99% deployed)')
    print(f'  Trades/month:       {port["tpm"]}  (MR: {round(port["mr_tpy"]/12,1)}/mo  BO: {round(port["bo_tpy"]/12,1)}/mo)')
    print(f'  Portfolio PF:       {port["pf"]}')
    print(f'  Max Drawdown:       {port["dd"]}%')
    print(f'  Win Rate:           {port["wr"]}%')
    print(f'  Avg profit/trade:   ${port["avg"]:,.2f}  (med ${port["med"]:,.2f})')
    print(f'  Monthly expectancy: ${port["monthly_exp"]:,.2f}')
    print(f'  Annual expectancy:  ${port["annual_exp"]:,.0f}')
    print(f'  Active days/month:  {ada["avg"]} avg  ({ada["ok_months"]}/{ada["total_months"]} months >= 10 days)')
    print(f'  Swing Phase 1:      {sw1["hit"]}% pass  |  median {sw1["med"]} tdays ({round(sw1["med"]/5,1) if sw1["med"] else "—"} wks / {round(sw1["med"]/21,1) if sw1["med"] else "—"} mo)')
    print(f'  Swing Phase 2:      {sw2["hit"]}% pass  |  median {sw2["med"]} tdays ({round(sw2["med"]/5,1) if sw2["med"] else "—"} wks / {round(sw2["med"]/21,1) if sw2["med"] else "—"} mo)')
    print(sep)

    generate_report(all_mr, all_bo_best, qual_slots, port, monthly, mbs, ada, mc1, mc2, sw1, sw2)

if __name__ == '__main__':
    main()

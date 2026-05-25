#!/usr/bin/env python3
"""
tv_opr_sweep.py
Parameter sweep for OPR Pine Script strategy on TradingView via Playwright.

Fixed: session=0900-0915, tz=Europe/Paris, revengeTrade=False
Varied: sl_ratio, mult1, ma_check, ma_length, percentage_tp1
Instruments: FRA40 (CAC40), GER40 (DAX)
Timeframes: 1min, 3min
Total: 288 runs with checkpointing for resume.
"""
import os, sys, re, json, time, base64, math, datetime, traceback, itertools, subprocess, shutil, tempfile

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
_here          = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE   = os.path.join(_here, '.tv-cookies.json')
CHART_URL_FILE = os.path.join(_here, '.tv-chart-url.txt')
REPORT_OUT     = os.path.join(_here, 'tv-opr-sweep-report.html')
CHECKPOINT     = os.path.join(_here, 'tv-opr-sweep-results.json')
CHROME_USER_DATA   = r'C:\Users\alexa\AppData\Local\Google\Chrome\User Data'
CHROME_PROFILE_NAME = 'Default'
CHROME_EXE         = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

# Use URL saved by get_cookies.py; fall back to generic chart
if os.path.exists(CHART_URL_FILE):
    with open(CHART_URL_FILE, encoding='utf-8-sig') as _f:  # utf-8-sig strips BOM
        TV_URL = _f.read().strip()
else:
    TV_URL = 'https://www.tradingview.com/chart/'
print(f"  Chart URL: {TV_URL}")

INSTRUMENTS = [
    dict(name='CAC40', symbol='FRA40', label='CAC 40 (FRA40)'),
    dict(name='GER40',  symbol='GER40',  label='DAX 40 (GER40)'),
]
TIMEFRAMES = ['1', '3']  # minutes

# Parameter grid
SL_RATIOS    = [0.5, 1.0, 1.5]
MULT1_LIST   = [1.0, 2.0, 3.0]
MA_CHECKS    = [True, False]
MA_LENGTHS   = [20, 32, 50]
PCT_TP1_LIST = [50.0, 100.0]

# Timing
SETTINGS_WAIT   = 4    # seconds after dialog opens
MAX_BT_WAIT     = 90   # max wait for backtest to stabilise
STABLE_TICKS    = 2    # how many consecutive 3s ticks with same hash = stable
SETTLE          = 3    # extra settle after stability detected
SYMBOL_WAIT     = 8    # after symbol change
TF_WAIT         = 8    # after timeframe change

# TradingView dialog input label candidates
# Exact labels confirmed from dialog screenshot + discovery output
LABELS = {
    'sl_ratio':       ['SL Ratio', 'sl_ratio', 'SL ratio', 'Stop Loss Ratio'],
    'mult1':          ['mult1', 'Mult1', 'TP Multiplier', 'TP1 Multiplier'],
    'ma_check':       ['Check trend with MA', 'ma_check', 'MA Check', 'Use MA',
                       'EMA Filter', 'MA Filter', 'Use EMA'],
    'ma_length':      ['MA Length', 'ma_length', 'EMA Length', 'MA Period', 'Length MA'],
    'percentage_tp1': ['Percentage TP1', 'percentage_tp1', 'TP1 %', 'TP1 Percentage',
                       'Pct TP1', 'TP1 Size'],
    'revengeTrade':   ['Allowed to take a second position inversed if stopped',
                       'revengeTrade', 'Revenge Trade', 'Revenge'],
}

# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------
def load_cookies(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as f:
            content = f.read().strip()
        return json.loads(content) if content else []
    except Exception:
        return []

def save_cookies(context, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(context.cookies(), f, indent=2)

# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------
def take_shot(page, name):
    path = os.path.join(_here, f'debug-sweep-{name}.png')
    try:
        page.screenshot(path=path, full_page=False)
        return path
    except Exception:
        return None

def img_b64(path):
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ''

# ---------------------------------------------------------------------------
# Login check
# ---------------------------------------------------------------------------
def check_logged_in(page):
    """Return True only when authenticated as owner — no View Only Mode banner."""
    try:
        found = page.evaluate(
            "() => document.body.innerText.includes('View Only Mode')"
            " || document.body.innerText.includes('Mode Voir uniquement')"
        )
        if found:
            return False
    except Exception:
        pass
    return True

# ---------------------------------------------------------------------------
# Symbol change
# ---------------------------------------------------------------------------
def dismiss_errors(page):
    """Dismiss TradingView error/warning/signup popups that can block interaction."""
    # Close button selectors — ordered from most to least specific
    close_selectors = [
        '[data-name="close-dialog"]',
        'button[aria-label="Close"]',
        'button[aria-label="Fermer"]',          # French locale
        '[class*="closeButton"]',
        '[class*="close-button"]',
        '[class*="CloseButton"]',
        '[class*="modal"] button[class*="close" i]',
        '[class*="dialog"] button[class*="close" i]',
        '[role="dialog"] button[class*="close" i]',
        '[class*="notification"] button[class*="close"]',
        '[class*="error"] button[class*="close"]',
        'button:has-text("OK"):visible',
        'button:has-text("Got it"):visible',
        'button:has-text("Fermer"):visible',
    ]
    for sel in close_selectors:
        try:
            btns = page.locator(sel).all()
            for btn in btns[:3]:
                if btn.is_visible(timeout=400):
                    btn.click()
                    time.sleep(0.3)
        except Exception:
            pass

    # Press Escape 3× to close any modal/overlay
    for _ in range(3):
        try:
            page.keyboard.press('Escape')
            time.sleep(0.4)
        except Exception:
            pass

    # If a signup/promo modal is still visible, click outside it (top-left chart area)
    try:
        modal_selectors = [
            '[class*="modal"]:visible',
            '[class*="dialog"]:visible',
            '[role="dialog"]:visible',
        ]
        for ms in modal_selectors:
            els = page.locator(ms).all()
            for el in els[:2]:
                if el.is_visible(timeout=300):
                    # Click outside the modal — top-left corner of the page
                    page.mouse.click(20, 20)
                    time.sleep(0.5)
                    break
    except Exception:
        pass


def change_symbol(page, symbol):
    """Change chart symbol. Uses fill() on the search input — same as working backtest script."""
    print(f"    symbol -> {symbol}", flush=True)

    dismiss_errors(page)

    # Try clicking the symbol search button in the toolbar
    search_opened = False
    for sel in [
        '[data-name="header-toolbar-symbol-search"]',
        '#header-toolbar-symbol-search',
        '[class*="symbolSearch"]',
        'button[class*="symbolButton"]',
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=2500):
                btn.click()
                time.sleep(1.0)
                search_opened = True
                break
        except Exception:
            continue

    if not search_opened:
        # Click chart canvas area first so '/' goes to the chart, not the watchlist
        for chart_sel in [
            '[class*="chart-markup-table"]',
            '[class*="chartContainer"]',
            'canvas',
            '[id*="layout"]',
        ]:
            try:
                c = page.locator(chart_sel).first
                if c.is_visible(timeout=1000):
                    c.click(position={'x': 400, 'y': 400})
                    time.sleep(0.4)
                    break
            except Exception:
                continue
        page.keyboard.press('/')
        time.sleep(1.5)

    # Find the search input and fill it
    for sel in [
        '[data-name="symbol-search-items-dialog"] input',
        'input[data-role="search"]',
        'input[class*="search"]',
        '[placeholder*="Search"]',
        '[placeholder*="symbol"]',
        'input[aria-label*="symbol"]',
        'input[aria-label*="Search"]',
        'input:focus',  # catch whatever has focus after '/' opens search
    ]:
        try:
            inp = page.locator(sel).first
            inp.wait_for(state='visible', timeout=4000)
            inp.fill(symbol)
            time.sleep(1.5)
            inp.press('Enter')
            time.sleep(SYMBOL_WAIT)
            print(f"    [ok] symbol changed to {symbol}", flush=True)
            return
        except Exception:
            continue

    # Last fallback: type blindly after search opened
    page.keyboard.type(symbol, delay=100)
    time.sleep(1.5)
    page.keyboard.press('Enter')
    time.sleep(SYMBOL_WAIT)
    print(f"    [ok] symbol change attempted (fallback)", flush=True)

# ---------------------------------------------------------------------------
# Timeframe
# ---------------------------------------------------------------------------
def set_timeframe(page, tf_minutes: str):
    print(f"    timeframe -> {tf_minutes}m", flush=True)
    for sel in [
        '[data-name="header-toolbar-intervals"]',
        '#header-toolbar-intervals',
        'button[class*="intervalButton"]',
        '[class*="interval-button"]',
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=3000):
                btn.click()
                time.sleep(1.5)
                set_ok = False
                for opt_sel in [
                    f'[data-value="{tf_minutes}"]',
                    f'button[data-value="{tf_minutes}"]',
                    f'[class*="item"]:has-text("{tf_minutes} minute")',
                    f'[class*="item"]:has-text("{tf_minutes}m")',
                ]:
                    try:
                        opt = page.locator(opt_sel).first
                        if opt.is_visible(timeout=2000):
                            opt.click()
                            time.sleep(TF_WAIT)
                            print(f"    [ok] timeframe set to {tf_minutes}m", flush=True)
                            set_ok = True
                            return
                    except Exception:
                        continue
                if not set_ok:
                    page.keyboard.press('Escape')
                    time.sleep(1)
                break
        except Exception:
            continue
    print(f"    [warn] could not set timeframe {tf_minutes}m — proceeding anyway", flush=True)

# ---------------------------------------------------------------------------
# Dialog helpers
# ---------------------------------------------------------------------------
def _find_dialog(page):
    for sel in [
        '[data-name="indicator-properties-dialog"]',
        '[class*="indicatorPropertiesDialog"]',
        '[role="dialog"]:visible',
    ]:
        try:
            dlg = page.locator(sel).first
            if dlg.is_visible(timeout=2000):
                return dlg
        except Exception:
            continue
    return None


def open_strategy_settings(page):
    """Open OPR settings dialog. Returns dialog locator or None."""
    for sel in [
        '[class*="study"]:has-text("OPR")',
        '[data-name="legend-series-item"]:has-text("OPR")',
        '[class*="legendItem"]:has-text("OPR")',
        '[class*="paneTitle"]:has-text("OPR")',
    ]:
        try:
            legend = page.locator(sel).first
            if not legend.is_visible(timeout=3000):
                continue

            legend.hover()
            time.sleep(0.8)

            # Try gear icon first
            for gear_sel in [
                '[data-name="legend-settings-action"]',
                '[data-action="open-settings"]',
                'button[aria-label*="ettings"]',
                '[class*="legendAction"]',
            ]:
                for container in (legend, page):
                    try:
                        g = container.locator(gear_sel).first
                        if g.is_visible(timeout=1200):
                            g.click()
                            time.sleep(SETTINGS_WAIT)
                            dlg = _find_dialog(page)
                            if dlg:
                                return dlg
                    except Exception:
                        continue

            # Fallback: dblclick (confirmed working in previous runs)
            legend.dblclick()
            time.sleep(SETTINGS_WAIT)
            dlg = _find_dialog(page)
            if dlg:
                return dlg

            # Fallback: right-click -> Properties
            legend.click(button='right')
            time.sleep(0.5)
            for prop_sel in [
                '[class*="menuItem"]:has-text("Properties")',
                '[class*="item"]:has-text("Properties")',
                '[data-name="properties"]',
            ]:
                try:
                    item = page.locator(prop_sel).first
                    if item.is_visible(timeout=1500):
                        item.click()
                        time.sleep(SETTINGS_WAIT)
                        dlg = _find_dialog(page)
                        if dlg:
                            return dlg
                except Exception:
                    continue
        except Exception:
            continue
    return None


def _click_inputs_tab(dlg):
    try:
        tab = dlg.locator(
            'button:has-text("Inputs"), [role="tab"]:has-text("Inputs")'
        ).first
        if tab.is_visible(timeout=2000):
            tab.click()
            time.sleep(0.5)
    except Exception:
        pass


def discover_dialog_inputs(dlg, page):
    """Print all inputs visible in the open dialog (for debugging label names)."""
    print("    [discover] inspecting dialog inputs:")
    try:
        inputs = dlg.locator('input').all()
        print(f"    [discover] found {len(inputs)} input elements")
        for i, inp in enumerate(inputs):
            try:
                itype = inp.get_attribute('type') or 'text'
                if itype == 'checkbox':
                    ival = str(inp.is_checked())
                else:
                    try:
                        ival = inp.get_attribute('value') or ''
                    except Exception:
                        ival = '?'
                # Walk up DOM to find label context
                ctx = ''
                try:
                    # Get grandparent text as label context
                    row = inp.locator('xpath=../..').first
                    ctx = row.inner_text()[:80].replace('\n', ' ').strip()
                except Exception:
                    try:
                        parent = inp.locator('xpath=..').first
                        ctx = parent.inner_text()[:60].replace('\n', ' ').strip()
                    except Exception:
                        pass
                print(f"      [{i:02d}] type={itype:10s} val={ival!r:12} ctx={ctx!r}")
            except Exception as e:
                print(f"      [{i:02d}] error: {e}")
    except Exception as e:
        print(f"    [discover error] {e}")


def _find_input_by_label(dlg, param_key):
    """Return the input element for a given parameter key, or None."""
    for label in LABELS[param_key]:
        for row_sel in [
            f'[class*="row"]:has-text("{label}")',
            f'tr:has-text("{label}")',
            f'[class*="cell"]:has-text("{label}")',
            f'[class*="input-container"]:has-text("{label}")',
            f'div:has-text("{label}")',
        ]:
            try:
                row = dlg.locator(row_sel).first
                if not row.is_visible(timeout=1000):
                    continue
                inp = row.locator('input').first
                if inp.count() > 0 and inp.is_visible(timeout=800):
                    return inp, label
            except Exception:
                continue
    return None, None


def _set_numeric(dlg, param_key, value):
    inp, found_label = _find_input_by_label(dlg, param_key)
    if inp is None:
        return False
    try:
        inp.click(click_count=3)   # select all (triple_click not in this Playwright version)
        inp.fill(str(value))
        inp.press('Tab')
        return True
    except Exception as e:
        print(f"    [warn] set_numeric {param_key}={value} failed: {e}")
        return False


def _set_bool(dlg, param_key, value):
    for label in LABELS[param_key]:
        for row_sel in [
            f'[class*="row"]:has-text("{label}")',
            f'tr:has-text("{label}")',
            f'[class*="cell"]:has-text("{label}")',
            f'div:has-text("{label}")',
        ]:
            try:
                row = dlg.locator(row_sel).first
                if not row.is_visible(timeout=1000):
                    continue

                # Native checkbox
                cb = row.locator('input[type="checkbox"]').first
                if cb.count() > 0 and cb.is_visible(timeout=800):
                    if cb.is_checked() != bool(value):
                        cb.click()
                    return True

                # Custom toggle (aria-checked)
                toggle = row.locator(
                    '[role="checkbox"], [class*="toggle"], [class*="switch"]'
                ).first
                if toggle.count() > 0 and toggle.is_visible(timeout=800):
                    aria = (toggle.get_attribute('aria-checked') or '').lower()
                    currently_on = aria == 'true'
                    if currently_on != bool(value):
                        toggle.click()
                    return True
            except Exception:
                continue
    return False


def set_params(page, dlg, params, first_call=False):
    """Apply all parameters in the open dialog."""
    _click_inputs_tab(dlg)
    if first_call:
        discover_dialog_inputs(dlg, page)
        take_shot(page, 'dialog_first')

    # Set ma_check first — it may show/hide dependent fields
    ok_ma = _set_bool(dlg, 'ma_check', params['ma_check'])
    if ok_ma:
        time.sleep(0.4)  # let UI update
    else:
        print(f"    [warn] ma_check not set (label not found)")

    ok_sl  = _set_numeric(dlg, 'sl_ratio',       params['sl_ratio'])
    ok_m1  = _set_numeric(dlg, 'mult1',           params['mult1'])
    ok_p1  = _set_numeric(dlg, 'percentage_tp1',  params['percentage_tp1'])

    ok_ml = True
    if params['ma_check']:
        ok_ml = _set_numeric(dlg, 'ma_length', params['ma_length'])
        if not ok_ml:
            print(f"    [warn] ma_length not set (label not found)")

    # revengeTrade fixed False
    _set_bool(dlg, 'revengeTrade', False)

    success = ok_sl or ok_m1 or ok_p1
    if not success:
        print(f"    [warn] no parameters were set — dialog structure unknown")
    return success


def click_ok(page):
    for sel in [
        'button:has-text("Ok")',
        'button:has-text("OK")',
        '[data-name="submit-button"]',
        'button[class*="ok"]',
        'button[class*="apply"]',
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=2000):
                btn.click()
                time.sleep(1)
                return True
        except Exception:
            continue
    print("    [warn] OK button not found")
    return False

# ---------------------------------------------------------------------------
# Backtest wait (smart polling)
# ---------------------------------------------------------------------------
def _get_bt_hash(page):
    for sel in [
        '[data-name="backtesting-content"]',
        '[class*="backtestingContent"]',
        '[class*="strategyTester"]',
    ]:
        try:
            panel = page.locator(sel).first
            if panel.is_visible(timeout=1000):
                return hash(panel.inner_text())
        except Exception:
            continue
    return None


def wait_stable(page):
    """Wait until strategy tester output stabilises."""
    time.sleep(5)  # give Pine a moment to start
    last_hash = None
    stable = 0
    deadline = time.time() + MAX_BT_WAIT
    while time.time() < deadline:
        h = _get_bt_hash(page)
        if h is not None and h == last_hash:
            stable += 1
            if stable >= STABLE_TICKS:
                break
        else:
            stable = 0
        last_hash = h
        time.sleep(3)
        print('.', end='', flush=True)
    time.sleep(SETTLE)

# ---------------------------------------------------------------------------
# Strategy Tester panel
# ---------------------------------------------------------------------------
def ensure_strategy_tester(page):
    for sel in [
        '[data-name="backtesting"]',
        'button:has-text("Strategy Tester")',
        '[class*="strategyTester"]',
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=2000):
                btn.click()
                time.sleep(2)
                return True
        except Exception:
            continue
    return False

# ---------------------------------------------------------------------------
# Read results
# ---------------------------------------------------------------------------
def read_results(page):
    metrics = dict(
        profit_factor=None, max_dd=None,
        win_rate=None, total_trades=None, net_profit=None,
    )
    # Click Performance Summary tab
    for tab_sel in [
        '[data-id="performance-summary-tab"]',
        'button:has-text("Performance Summary")',
        '[class*="performanceSummary"]',
    ]:
        try:
            tab = page.locator(tab_sel).first
            if tab.is_visible(timeout=2000):
                tab.click()
                time.sleep(1)
                break
        except Exception:
            continue

    # Get panel text
    panel_text = ''
    for sel in [
        '[data-name="backtesting-content"]',
        '[class*="backtestingContent"]',
        '[class*="strategyTester"]',
        '[class*="bottomArea"]',
    ]:
        try:
            panel = page.locator(sel).first
            if panel.is_visible(timeout=2000):
                panel_text = panel.inner_text()
                break
        except Exception:
            continue
    if not panel_text:
        try:
            panel_text = page.inner_text('body')
        except Exception:
            pass

    def grab(patterns):
        for pat in patterns:
            m = re.search(pat, panel_text, re.IGNORECASE | re.DOTALL)
            if m:
                raw = re.sub(r'[^0-9.\-]',
                             '',
                             m.group(1).replace(',', '').replace('\xa0', '').strip())
                try:
                    return float(raw)
                except Exception:
                    continue
        return None

    metrics['profit_factor'] = grab([
        r'Profit\s*Factor\s*\n?([\d.,]+)',
        r'Profit\s*Factor[^\d]+([\d.]+)',
    ])
    metrics['max_dd'] = grab([
        r'Max(?:imum)?\s*(?:Equity\s*)?(?:Drawdown|DD)[^\d\-]+([\d.,]+)\s*%',
        r'Max(?:imum)?\s*(?:Drawdown|DD)\s*\n?([\d.,]+)',
    ])
    metrics['win_rate'] = grab([
        r'Percent\s*Profitable\s*\n?([\d.,]+)',
        r'Percent\s*Profitable[^\d]+([\d.,]+)',
        r'Win\s*Rate[^\d]+([\d.,]+)',
    ])
    metrics['total_trades'] = grab([
        r'Total\s*Closed\s*Trades\s*\n?([\d,]+)',
        r'Total\s*(?:Closed\s*)?Trades[^\d]+([\d,]+)',
    ])
    metrics['net_profit'] = grab([
        r'Net\s*Profit\s*\n?([−\-]?[\d.,]+)',
        r'Net\s*Profit[^\d\-]+([\d.,]+)',
    ])
    return metrics

# ---------------------------------------------------------------------------
# Parameter combinations
# ---------------------------------------------------------------------------
def build_combos():
    combos = []
    for sl_r, m1, ma_on, pct1 in itertools.product(
            SL_RATIOS, MULT1_LIST, MA_CHECKS, PCT_TP1_LIST):
        if ma_on:
            for ma_len in MA_LENGTHS:
                combos.append(dict(
                    sl_ratio=sl_r, mult1=m1,
                    ma_check=ma_on, ma_length=ma_len,
                    percentage_tp1=pct1,
                ))
        else:
            # ma_length irrelevant when filter off; use 32 as placeholder
            combos.append(dict(
                sl_ratio=sl_r, mult1=m1,
                ma_check=ma_on, ma_length=32,
                percentage_tp1=pct1,
            ))
    return combos


def combo_key(inst_name, tf, p):
    return (f"{inst_name}_{tf}m_sl{p['sl_ratio']}_m1{p['mult1']}"
            f"_ma{int(p['ma_check'])}_ml{p['ma_length']}_pct{int(p['percentage_tp1'])}")

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return {}
    try:
        with open(CHECKPOINT, encoding='utf-8') as f:
            content = f.read().strip()
        return json.loads(content) if content else {}
    except Exception:
        return {}

def save_checkpoint(done):
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(done, f, indent=2)

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def generate_report(all_results_dict):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    all_results = list(all_results_dict.values())
    valid = [r for r in all_results
             if r.get('profit_factor') is not None and not r.get('error')]
    valid.sort(key=lambda x: x.get('profit_factor', 0) or 0, reverse=True)

    top = [r for r in valid
           if (r.get('profit_factor') or 0) > 2.0
           and (r.get('max_dd') or 999) < 8.0
           and (r.get('total_trades') or 0) >= 30]

    def fmt(v, sfx='', dec=2):
        if v is None:
            return 'N/A'
        return f'{v:.{dec}f}{sfx}'

    def row_cls(r):
        pf = r.get('profit_factor') or 0
        dd = r.get('max_dd') or 999
        tr = r.get('total_trades') or 0
        if pf > 2.0 and dd < 8.0 and tr >= 30:
            return 'good'
        if pf >= 1.5:
            return 'warn'
        return 'bad'

    def row_html(r):
        c    = row_cls(r)
        ma_l = fmt(r.get('ma_length'), '', 0) if r.get('ma_check') else 'off'
        return (
            f'<tr class="{c}">'
            f'<td>{r.get("inst","?")} {r.get("tf","?")}m</td>'
            f'<td>{fmt(r.get("sl_ratio"))}</td>'
            f'<td>{fmt(r.get("mult1"))}</td>'
            f'<td>{"on" if r.get("ma_check") else "off"}</td>'
            f'<td>{ma_l}</td>'
            f'<td>{fmt(r.get("percentage_tp1"), "%", 0)}</td>'
            f'<td class="pf">{fmt(r.get("profit_factor"))}</td>'
            f'<td>{fmt(r.get("max_dd"), "%", 1)}</td>'
            f'<td>{fmt(r.get("win_rate"), "%", 1)}</td>'
            f'<td>{fmt(r.get("total_trades"), "", 0)}</td>'
            f'<td>{fmt(r.get("net_profit"), "", 0)}</td>'
            '</tr>'
        )

    thead = '''<thead><tr>
      <th>Instrument TF</th><th>SL</th><th>Mult</th>
      <th>MA</th><th>MA Len</th><th>TP1%</th>
      <th>PF</th><th>Max DD</th><th>WR</th><th>Trades</th><th>Net P&amp;L</th>
    </tr></thead>'''

    top_body = '\n'.join(row_html(r) for r in top) if top else (
        '<tr><td colspan="11" class="na">No combos pass all 3 filters</td></tr>')
    all_body = '\n'.join(row_html(r) for r in valid)

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<title>OPR Sweep -- {ts}</title>
<style>
  :root{{--bg:#0d0d0d;--box:#1a1a1a;--border:#2a2a2a;--accent:#f0b429;
        --good:#10b981;--bad:#ef4444;--warn:#f59e0b;--text:#e5e5e5;--muted:#888}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;padding:24px}}
  h1{{color:var(--accent);font-size:1.5rem;margin-bottom:4px}}
  h2{{color:var(--accent);font-size:1rem;margin:24px 0 10px}}
  .sub{{color:var(--muted);font-size:0.83rem;margin-bottom:18px}}
  table{{width:100%;border-collapse:collapse;font-size:0.80rem;margin-bottom:20px}}
  th{{background:#1e1e1e;color:var(--accent);text-align:left;
      padding:7px 9px;border-bottom:2px solid var(--border)}}
  td{{padding:6px 9px;border-bottom:1px solid var(--border)}}
  tr.good{{background:#071410}} tr.good .pf{{color:var(--good);font-weight:700}}
  tr.warn .pf{{color:var(--warn);font-weight:700}}
  tr.bad  .pf{{color:var(--bad)}}
  .na{{color:var(--muted);font-style:italic}}
  .stats{{background:var(--box);border:1px solid var(--border);border-radius:6px;
           padding:12px 16px;font-size:0.83rem;line-height:1.9;margin-bottom:18px}}
  .stats b{{color:var(--accent)}}
</style></head><body>
<h1>OPR Parameter Sweep -- TradingView Backtest</h1>
<p class="sub">Generated {ts} | Fixed: session=0900-0915 / Europe/Paris / revengeTrade=off</p>

<div class="stats">
  <b>Sweep scope</b><br>
  Instruments: FRA40 (CAC40), GER40 (DAX) | Timeframes: 1min, 3min<br>
  sl_ratio: 0.5 / 1.0 / 1.5 | mult1 (TP): 1 / 2 / 3<br>
  ma_check: on/off | ma_length: 20 / 32 / 50 | percentage_tp1: 50% / 100%<br>
  <b>Tested:</b> {len(valid)} valid / {len(all_results)} total &nbsp;|&nbsp;
  <b>Passing (PF&gt;2.0, DD&lt;8%, Trades&ge;30):</b> {len(top)}
</div>

<h2>Top Combos -- PF &gt; 2.0 / Max DD &lt; 8% / Trades &ge; 30</h2>
<table>{thead}<tbody>{top_body}</tbody></table>

<h2>All Results (sorted by PF descending)</h2>
<table>{thead}<tbody>{all_body}</tbody></table>
</body></html>"""

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Report: {REPORT_OUT}")

# ---------------------------------------------------------------------------
# Chrome profile helpers
# ---------------------------------------------------------------------------
def _chrome_running():
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq chrome.exe', '/NH'],
                       capture_output=True, text=True)
    return 'chrome.exe' in r.stdout


def _copy_chrome_profile() -> str:
    """
    Close Chrome briefly, copy the minimal profile files to a temp dir,
    relaunch Chrome, return the temp dir path.
    Playwright uses this temp dir so Chrome can run in parallel with the sweep.
    """
    was_running = _chrome_running()
    if was_running:
        print("  Closing Chrome to copy profile (will relaunch automatically) ...", flush=True)
        subprocess.run(['taskkill', '/IM', 'chrome.exe'], capture_output=True)
        time.sleep(4)   # let Chrome flush all data

    src_root    = CHROME_USER_DATA
    src_default = os.path.join(src_root, CHROME_PROFILE_NAME)
    tmp_root    = tempfile.mkdtemp(prefix='tv-pw-profile-')
    tmp_default = os.path.join(tmp_root, CHROME_PROFILE_NAME)
    os.makedirs(tmp_default, exist_ok=True)

    # Local State — contains the DPAPI-wrapped AES key for cookie decryption
    shutil.copy2(os.path.join(src_root, 'Local State'),
                 os.path.join(tmp_root, 'Local State'))

    # Cookies
    for rel in [os.path.join('Network', 'Cookies'), 'Cookies']:
        src = os.path.join(src_default, rel)
        dst = os.path.join(tmp_default, rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            break

    # Local Storage — TradingView auth token lives here
    src_ls = os.path.join(src_default, 'Local Storage')
    dst_ls = os.path.join(tmp_default, 'Local Storage')
    if os.path.exists(src_ls):
        shutil.copytree(src_ls, dst_ls,
                        ignore=shutil.ignore_patterns('LOCK'))

    # Preferences — needed for a valid Chromium profile
    src_pref = os.path.join(src_default, 'Preferences')
    if os.path.exists(src_pref):
        shutil.copy2(src_pref, os.path.join(tmp_default, 'Preferences'))

    # Relaunch Chrome so the user gets their browser back immediately
    if was_running:
        subprocess.Popen([CHROME_EXE])
        print("  Chrome relaunched.", flush=True)

    print(f"  Temp profile created: {tmp_root}", flush=True)
    return tmp_root


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep():
    combos = build_combos()
    total  = len(INSTRUMENTS) * len(TIMEFRAMES) * len(combos)
    print(f"  Combos/pair: {len(combos)} | Total runs: {total}")

    done = load_checkpoint()
    if done:
        print(f"  Resuming from checkpoint: {len(done)} already done")

    run_n = 0

    # --- Copy Chrome profile to temp dir (so Chrome can reopen while sweep runs) ---
    tmp_profile = _copy_chrome_profile()

    with sync_playwright() as pw:
        # Use persistent context with the temp profile copy
        # Playwright's Chromium reads Chrome's cookies (decrypts via DPAPI) + localStorage
        context = pw.chromium.launch_persistent_context(
            user_data_dir=tmp_profile,
            headless=False,
            slow_mo=150,
            args=[
                '--start-maximized',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-extensions',
            ],
        )
        print("  Browser launched with Chrome profile copy", flush=True)

        page = context.new_page()
        page.goto(TV_URL, wait_until='domcontentloaded', timeout=60000)
        print("  Chart loading ... DO NOT CLOSE THIS BROWSER WINDOW", flush=True)
        time.sleep(14)

        try:
            page.evaluate("document.title = 'OPR Sweep - DO NOT CLOSE'")
        except Exception:
            pass

        if check_logged_in(page):
            print("  [ok] Chart in edit mode — authenticated.", flush=True)
        else:
            print("  [!] View Only Mode visible — waiting up to 15 min for login ...", flush=True)
            for _i in range(180):
                time.sleep(5)
                if check_logged_in(page):
                    print(f"  [ok] Edit mode after {(_i+1)*5}s!", flush=True)
                    break
                if (_i + 1) % 6 == 0:
                    print(f"  ... {(_i+1)*5}s still waiting ...", flush=True)
            page.goto(TV_URL, wait_until='domcontentloaded', timeout=45000)
            time.sleep(12)

        # Dismiss any popups (signup modals, error dialogs) before starting
        time.sleep(4)
        dismiss_errors(page)
        time.sleep(2)
        dismiss_errors(page)   # second pass — some modals appear after first Escape

        # Debug: take screenshot of current chart state
        take_shot(page, 'startup')
        print("  Startup screenshot saved: debug-sweep-startup.png", flush=True)

        # current_symbol = None forces symbol change on run 1 (confirmed to work)
        current_symbol = None
        current_tf     = None
        first_dialog   = True  # trigger discovery on first open

        for inst in INSTRUMENTS:
            for tf in TIMEFRAMES:
                for combo in combos:
                    run_n += 1
                    key = combo_key(inst['name'], tf, combo)

                    if key in done:
                        print(f"  [{run_n:3d}/{total}] cached: {key}")
                        continue

                    ma_tag = (f"ma_len={combo['ma_length']}"
                              if combo['ma_check'] else "ma=off")
                    print(f"\n  [{run_n:3d}/{total}] {inst['name']} {tf}m | "
                          f"sl={combo['sl_ratio']} m1={combo['mult1']} "
                          f"{ma_tag} pct={combo['percentage_tp1']}")

                    result = dict(
                        inst=inst['name'],
                        label=inst['label'],
                        tf=tf,
                        **combo,
                        profit_factor=None,
                        max_dd=None,
                        win_rate=None,
                        total_trades=None,
                        net_profit=None,
                        error=None,
                    )

                    try:
                        dismiss_errors(page)
                        time.sleep(1)

                        # Symbol
                        if current_symbol != inst['symbol']:
                            change_symbol(page, inst['symbol'])
                            current_symbol = inst['symbol']
                            current_tf = None  # may reset
                            # Let chart fully reload after symbol change
                            time.sleep(4)

                        # Timeframe
                        if current_tf != tf:
                            set_timeframe(page, tf)
                            current_tf = tf
                            time.sleep(3)  # extra settle after TF change

                        # Debug screenshot on first real run
                        if first_dialog:
                            take_shot(page, f'before_first_settings')

                        # Open settings dialog
                        dlg = open_strategy_settings(page)
                        if dlg is None:
                            take_shot(page, f'settings_fail_{run_n}')
                            raise RuntimeError("Could not open strategy settings")

                        # Apply parameters
                        set_params(page, dlg, combo, first_call=first_dialog)
                        first_dialog = False

                        click_ok(page)

                        # Open Strategy Tester panel (before waiting for results)
                        ensure_strategy_tester(page)

                        # Wait for backtest
                        print(f"    waiting", end='', flush=True)
                        wait_stable(page)
                        print(' done', flush=True)

                        # Read metrics
                        metrics = read_results(page)
                        result.update(metrics)

                        pf = metrics.get('profit_factor')
                        dd = metrics.get('max_dd')
                        wr = metrics.get('win_rate')
                        tr = metrics.get('total_trades')
                        flag = ' ***' if (pf and pf > 2.0 and (dd or 999) < 8.0
                                         and (tr or 0) >= 30) else ''
                        print(f"    PF={pf}  DD={dd}%  WR={wr}%  T={tr}{flag}", flush=True)

                    except Exception as e:
                        result['error'] = str(e)
                        print(f"    ERROR: {e}", flush=True)
                        try:
                            page.keyboard.press('Escape')
                            time.sleep(1.5)
                        except Exception:
                            pass

                    done[key] = result
                    save_checkpoint(done)

        try:
            context.close()
        except Exception:
            pass

    try:
        shutil.rmtree(tmp_profile, ignore_errors=True)
    except Exception:
        pass

    generate_report(done)
    print("\n  Sweep complete.")
    return done


if __name__ == '__main__':
    print("=" * 60)
    print("  OPR Parameter Sweep -- TradingView Playwright")
    print("=" * 60)
    run_sweep()

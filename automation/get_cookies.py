#!/usr/bin/env python3
"""
get_cookies.py
Opens fr.tradingview.com → log in → navigate to your OPR chart.
Saves cookies when the chart loads WITHOUT "Mode Voir uniquement".
Run this ONCE before the sweep.
"""
import os, sys, json, time

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from playwright.sync_api import sync_playwright

_here          = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE   = os.path.join(_here, '.tv-cookies.json')
CHART_URL_FILE = os.path.join(_here, '.tv-chart-url.txt')
TV_HOME        = 'https://fr.tradingview.com'


def page_has_view_only(page):
    """Return True if the page shows View Only / Mode Voir uniquement."""
    try:
        return page.evaluate(
            "() => document.body.innerText.includes('View Only Mode')"
            " || document.body.innerText.includes('Mode Voir uniquement')"
        )
    except Exception:
        return True   # assume not logged in if we can't check


def main():
    print("=" * 60)
    print("  TradingView Cookie Capture  (fr.tradingview.com)")
    print("=" * 60)
    print()
    print("  Steps:")
    print("  1. Log in using the 'Se connecter' button in the browser.")
    print("  2. Navigate to your OPR chart (URL must contain /chart/).")
    print("  3. Chart must open WITHOUT 'Mode Voir uniquement' banner.")
    print("     (that banner = not authenticated as chart owner)")
    print("  Script saves cookies + URL then exits automatically.")
    print()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            slow_mo=100,
            args=['--start-maximized'],
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
        )

        # Load existing cookies (may speed up login)
        if os.path.exists(COOKIES_FILE):
            try:
                with open(COOKIES_FILE, encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    existing = json.loads(content)
                    context.add_cookies(existing)
                    print(f"  Loaded {len(existing)} existing cookies")
            except Exception:
                pass

        CHART_URL = 'https://fr.tradingview.com/chart/gM9KOvcb/?symbol=FUSIONMARKETS%3AFRA40'

        page = context.new_page()
        # Go straight to the chart — user will see the chart, not the home page
        page.goto(CHART_URL, wait_until='domcontentloaded', timeout=45000)
        time.sleep(8)

        print("  Browser shows the OPR chart.")
        if page_has_view_only(page):
            print("  'Mode Voir uniquement' visible — log in NOW in THIS browser window.")
            print("  Click 'Se connecter' (top right of the page in the Playwright window).")
            print("  Waiting ...")
            print()
        else:
            print("  Already authenticated — no View Only Mode!")

        tick = 0
        while True:
            time.sleep(3)
            tick += 1
            try:
                if not page_has_view_only(page):
                    print(f"  [ok] Authenticated — no View Only Mode after ~{tick*3}s!")
                    break
            except Exception:
                pass
            if tick % 10 == 0:
                print(f"  ... {tick*3}s — still waiting for login ...")

        url = page.url

        time.sleep(3)  # let cookies settle

        # Save cookies
        with open(COOKIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(context.cookies(), f, indent=2)
        print(f"  Cookies saved  -> {COOKIES_FILE}")

        # Save chart URL (strip query params that might pin a symbol)
        base_url = url.split('?')[0]
        with open(CHART_URL_FILE, 'w', encoding='utf-8') as f:
            f.write(base_url)
        print(f"  Chart URL saved -> {CHART_URL_FILE}  ({base_url})")

        print()
        print("  Done. Run tv_opr_sweep.py now.")
        print("  Closing browser in 5s ...")
        time.sleep(5)
        browser.close()


if __name__ == '__main__':
    main()

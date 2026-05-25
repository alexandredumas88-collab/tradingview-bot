#!/usr/bin/env python3
"""
extract_chrome_cookies.py
Decrypts Chrome cookies for tradingview.com using DPAPI + AES-256-GCM.
Closes Chrome if running (to release file lock), extracts cookies, then relaunches Chrome.
Requires: pywin32, pycryptodome
"""
import os, json, sqlite3, shutil, base64, tempfile, time, subprocess


CHROME_EXE     = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
CHROME_PROFILE = r'C:\Users\alexa\AppData\Local\Google\Chrome\User Data\Default'


def _chrome_is_running() -> bool:
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq chrome.exe', '/NH'],
                       capture_output=True, text=True)
    return 'chrome.exe' in r.stdout


def _wait_for_file_unlocked(path: str, timeout: int = 15) -> bool:
    """Return True once the file can be opened for read."""
    for _ in range(timeout):
        try:
            with open(path, 'rb'):
                return True
        except PermissionError:
            time.sleep(1)
    return False


def get_chrome_cookies(profile_dir: str = CHROME_PROFILE,
                       domain_filter: str = "tradingview.com") -> list:
    """
    Extract and decrypt TradingView cookies from Chrome profile.
    Closes Chrome if needed (and relaunches it when done).
    """
    import win32crypt
    from Crypto.Cipher import AES

    # -- Master key --------------------------------------------------------
    local_state_path = os.path.join(os.path.dirname(profile_dir), "Local State")
    with open(local_state_path, encoding='utf-8') as f:
        local_state = json.load(f)
    enc_key_b64  = local_state["os_crypt"]["encrypted_key"]
    enc_key      = base64.b64decode(enc_key_b64)[5:]   # strip "DPAPI" prefix
    master_key   = win32crypt.CryptUnprotectData(enc_key, None, None, None, 0)[1]

    # -- Cookies path -------------------------------------------------------
    cookies_path = os.path.join(profile_dir, "Network", "Cookies")
    if not os.path.exists(cookies_path):
        cookies_path = os.path.join(profile_dir, "Cookies")

    # -- Close Chrome if running (file lock) --------------------------------
    chrome_was_running = _chrome_is_running()
    if chrome_was_running:
        print("  Chrome is running — closing it briefly to read cookies ...")
        subprocess.run(['taskkill', '/IM', 'chrome.exe'], capture_output=True)
        time.sleep(2)
        if not _wait_for_file_unlocked(cookies_path, timeout=15):
            print("  [warn] Cookies file still locked after 15s — proceeding anyway")

    # -- Copy database to temp file -----------------------------------------
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    try:
        shutil.copy2(cookies_path, tmp.name)
    except Exception as e:
        print(f"  [warn] Could not copy Cookies db: {e}")
        if chrome_was_running:
            subprocess.Popen([CHROME_EXE])
        return []

    # -- Relaunch Chrome immediately so user notices minimal interruption ----
    if chrome_was_running:
        subprocess.Popen([CHROME_EXE])
        print("  Chrome relaunched.")

    # -- Decrypt & return ---------------------------------------------------
    def decrypt_value(enc: bytes) -> str:
        if enc[:3] in (b'v10', b'v11'):
            nonce      = enc[3:15]
            ciphertext = enc[15:-16]
            tag        = enc[-16:]
            try:
                return AES.new(master_key, AES.MODE_GCM, nonce=nonce) \
                           .decrypt_and_verify(ciphertext, tag) \
                           .decode('utf-8', errors='replace')
            except Exception:
                return ""
        try:
            return win32crypt.CryptUnprotectData(enc, None, None, None, 0)[1] \
                              .decode('utf-8', errors='replace')
        except Exception:
            return ""

    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(
        "SELECT host_key, name, encrypted_value, path, expires_utc, "
        "       is_secure, is_httponly, samesite "
        "FROM cookies WHERE host_key LIKE ?",
        (f"%{domain_filter}%",),
    )
    rows = cur.fetchall()
    conn.close()
    try: os.unlink(tmp.name)
    except Exception: pass

    SAMESITE = ["Strict", "Lax", "None", "Lax"]
    result   = []
    for row in rows:
        value = decrypt_value(bytes(row["encrypted_value"]))
        if not value:
            continue
        expires = 0
        if row["expires_utc"]:
            expires = (row["expires_utc"] / 1_000_000) - 11_644_473_600
        result.append({
            "name":     row["name"],
            "value":    value,
            "domain":   row["host_key"],
            "path":     row["path"],
            "expires":  expires,
            "httpOnly": bool(row["is_httponly"]),
            "secure":   bool(row["is_secure"]),
            "sameSite": SAMESITE[max(0, min(row["samesite"] or 0, 3))],
        })
    return result


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tv-cookies.json")
    cookies = get_chrome_cookies()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cookies, f, indent=2)
    print(f"Extracted {len(cookies)} TradingView cookies -> {out}")

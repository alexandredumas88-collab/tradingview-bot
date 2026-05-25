#!/usr/bin/env python3
import sqlite3, shutil, tempfile, os, json, base64, win32crypt
from Crypto.Cipher import AES

profile = r"C:\Users\alexa\AppData\Local\Google\Chrome\User Data\Default"
cookies_path = os.path.join(profile, "Network", "Cookies")

ls_path = r"C:\Users\alexa\AppData\Local\Google\Chrome\User Data\Local State"
with open(ls_path, encoding="utf-8") as f:
    ls = json.load(f)
osc = ls.get("os_crypt", {})
print("os_crypt keys:", list(osc.keys()))
enc_key_b64 = osc.get("encrypted_key", "")
raw_key = base64.b64decode(enc_key_b64) if enc_key_b64 else b""
print(f"encrypted_key: total={len(raw_key)} bytes, prefix={raw_key[:5]}")

master_key = win32crypt.CryptUnprotectData(raw_key[5:], None, None, None, 0)[1]
print(f"master_key length: {len(master_key)} bytes")

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
tmp.close()
shutil.copy2(cookies_path, tmp.name)
conn = sqlite3.connect(tmp.name)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute(
    "SELECT name, host_key, encrypted_value, value "
    "FROM cookies WHERE host_key LIKE '%tradingview%' LIMIT 5"
)
for row in cur.fetchall():
    enc = bytes(row["encrypted_value"])
    print(f"\nCookie: {row['name']} @ {row['host_key']}")
    print(f"  encrypted_value length: {len(enc)}")
    print(f"  first 20 bytes (hex): {enc[:20].hex()}")
    print(f"  prefix text: {enc[:3]}")
    print(f"  plain value col: {row['value']!r}")
    if enc[:3] in (b'v10', b'v11'):
        nonce = enc[3:15]
        ct    = enc[15:-16]
        tag   = enc[-16:]
        try:
            pt = AES.new(master_key, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ct, tag)
            print(f"  decrypted (first 40 bytes): {pt[:40]!r}")
        except Exception as e:
            print(f"  decrypt error: {e}")
    else:
        print("  Not v10/v11 — raw hex:", enc[:8].hex())
conn.close()
os.unlink(tmp.name)

import sqlite3, shutil, tempfile, os, win32crypt, base64, json
from Crypto.Cipher import AES

cookies_path = r'C:\Users\alexa\AppData\Local\Google\Chrome\User Data\Default\Network\Cookies'
ls_path = r'C:\Users\alexa\AppData\Local\Google\Chrome\User Data\Local State'

with open(ls_path, encoding='utf-8') as f:
    ls = json.load(f)
enc_key = base64.b64decode(ls['os_crypt']['encrypted_key'])[5:]
master_key = win32crypt.CryptUnprotectData(enc_key, None, None, None, 0)[1]

def decrypt(enc):
    if enc[:3] in (b'v10', b'v11'):
        nonce = enc[3:15]; ct = enc[15:-16]; tag = enc[-16:]
        try:
            pt = AES.new(master_key, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ct, tag)
            return pt[32:].decode('utf-8', errors='replace')  # strip 32-byte Chrome prefix
        except Exception as e:
            return f'[ERR:{e}]'
    try:
        return win32crypt.CryptUnprotectData(enc, None, None, None, 0)[1].decode('utf-8', errors='replace')
    except Exception:
        return '[dpapi-err]'

tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.db'); tmp.close()
shutil.copy2(cookies_path, tmp.name)
conn = sqlite3.connect(tmp.name)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT host_key, name, encrypted_value FROM cookies WHERE host_key LIKE '%tradingview%' ORDER BY host_key, name")
rows = cur.fetchall()
conn.close()
os.unlink(tmp.name)

print(f"Total TradingView cookies: {len(rows)}")
for r in rows:
    v = decrypt(bytes(r['encrypted_value']))
    print(f"  {r['host_key']:45s}  {r['name']:40s}  = {v[:60]!r}")

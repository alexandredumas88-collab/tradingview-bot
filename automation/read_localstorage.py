#!/usr/bin/env python3
"""
Read Chrome's localStorage LevelDB and show TradingView-related entries.
Install plyvel: pip install plyvel-wheels  (Windows-friendly build)
"""
import os, sys

try:
    import plyvel
    HAS_PLYVEL = True
except ImportError:
    HAS_PLYVEL = False

if not HAS_PLYVEL:
    print("plyvel not installed. Trying to install plyvel-wheels ...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'plyvel-wheels'], check=True)
    import plyvel

ls_path = r'C:\Users\alexa\AppData\Local\Google\Chrome\User Data\Default\Local Storage\leveldb'
db = plyvel.DB(ls_path, create_if_missing=False)

print("TradingView localStorage entries:")
count = 0
for key, value in db:
    k = key.decode('utf-8', errors='replace')
    if 'tradingview' in k.lower():
        v_str = value.decode('utf-8', errors='replace')
        print(f"  KEY: {k!r}")
        print(f"  VAL: {v_str[:200]!r}")
        print()
        count += 1
db.close()
print(f"Total TradingView entries: {count}")

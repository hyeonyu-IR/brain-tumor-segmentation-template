# scripts/99_debug_train.py
import sys
from pathlib import Path

print("DEBUG: starting 99_debug_train.py")
print("DEBUG: sys.executable =", sys.executable)
print("DEBUG: sys.path[0] =", sys.path[0])
print("DEBUG: cwd =", Path.cwd())

# Force project root onto path (prevents src name collision)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
print("DEBUG: inserted project root:", PROJECT_ROOT)

print("DEBUG: importing src.train...")
import src.train as train
print("DEBUG: src.train imported from:", train.__file__)

print("DEBUG: calling main()...")
train.main()

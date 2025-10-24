import sys
import os

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    import app.dashboard.app
    print("Successfully imported app.dashboard.app")
except ImportError as e:
    print("Import error:", e)
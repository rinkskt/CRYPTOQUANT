import sys, os
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)
print("Loaded modules:", list(sys.modules.keys())[:50])

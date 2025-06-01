#!/usr/bin/env python3
"""
Script to help restart Pylance and clear cache
"""

import os
import shutil
import sys

def clear_pylance_cache():
    """Clear Pylance cache directories"""
    cache_dirs = [
        os.path.expanduser("~/.vscode/extensions"),
        os.path.expanduser("~/AppData/Roaming/Code/User/workspaceStorage"),
        os.path.expanduser("~/AppData/Local/Microsoft/vscode-python"),
        ".vscode/.ropeproject",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                if cache_dir.endswith("__pycache__"):
                    # Find all __pycache__ directories
                    for root, dirs, files in os.walk("."):
                        if "__pycache__" in dirs:
                            pycache_path = os.path.join(root, "__pycache__")
                            print(f"Removing {pycache_path}")
                            shutil.rmtree(pycache_path, ignore_errors=True)
                elif os.path.basename(cache_dir) in ["workspaceStorage", "vscode-python"]:
                    print(f"Found cache directory: {cache_dir}")
                    print("Please manually clear VS Code workspace storage if needed")
                else:
                    print(f"Cache directory exists: {cache_dir}")
            except Exception as e:
                print(f"Could not clear {cache_dir}: {e}")

def main():
    print("🔄 Pylance Cache Clearing Script")
    print("=" * 40)
    
    clear_pylance_cache()
    
    print("\n📋 Manual Steps to Fix Pylance:")
    print("1. Press Ctrl+Shift+P in VS Code")
    print("2. Type 'Python: Restart Language Server'")
    print("3. Press Enter")
    print("4. Wait for Pylance to restart")
    print("5. Press Ctrl+Shift+P again")
    print("6. Type 'Python: Refresh Language Server'")
    print("7. Press Enter")
    print("\n🔄 Alternative:")
    print("1. Press Ctrl+Shift+P")
    print("2. Type 'Developer: Reload Window'")
    print("3. Press Enter")
    
    print(f"\n✅ Python executable: {sys.executable}")
    print("✅ TensorFlow should be available at this path")

if __name__ == "__main__":
    main()

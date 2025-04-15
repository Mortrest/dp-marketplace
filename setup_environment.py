#!/usr/bin/env python3
"""
Setup script to initialize the environment for PrivateCovariance.

This script:
1. Clones the PrivateCovariance repository if it doesn't exist
2. Sets up the necessary Python paths
3. Verifies all dependencies are installed
4. Checks that the imports work correctly
"""

import subprocess
import os
import sys
import importlib
import traceback


def ensure_repo_cloned():
    """Ensures that the PrivateCovariance repository is cloned."""
    if not os.path.exists("PrivateCovariance"):
        print("Cloning PrivateCovariance repository...")
        subprocess.run(["git", "clone", "https://github.com/Mortrest/PrivateCovariance.git"], check=True)
        print("Repository cloned successfully.")
    else:
        print("PrivateCovariance repository already exists.")


def setup_python_paths():
    """Sets up the necessary Python paths."""
    # Add main directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        print(f"Added {current_dir} to Python path.")

    # Add the PrivateCovariance repository to path
    repo_dir = os.path.join(current_dir, "PrivateCovariance")
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
        print(f"Added {repo_dir} to Python path.")


def check_dependencies():
    """Checks that all required dependencies are installed."""
    dependencies = ["numpy", "torch", "matplotlib", "scipy", "tqdm"]
    missing = []

    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} is installed.")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep} is missing.")

    if missing:
        print("\nInstalling missing dependencies...")
        for dep in missing:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"Installed {dep}.")


def verify_imports():
    """Verifies that key imports work correctly."""
    try:
        # Our local imports
        from config import Args
        from utils.data import generate_synthetic_data
        from utils.covariance import compute_covariance
        from algorithms import EMCov, CoinpressCov
        
        # Try loading adaptive methods
        import sys
        sys.path.append("PrivateCovariance")
        from adaptive.algos import AdaptiveCov, GaussCov, SeparateCov
        from algorithms.adaptive import AdaptiveCovWrapper
        
        print("\n✓ All imports working correctly.")
        return True
    except Exception as e:
        print("\n✗ Import error:", str(e))
        print("\nTraceback:")
        traceback.print_exc()
        return False


def main():
    """Main setup function."""
    print("Setting up environment for PrivateCovariance...\n")
    
    ensure_repo_cloned()
    setup_python_paths()
    check_dependencies()
    
    success = verify_imports()
    
    if success:
        print("\nSetup completed successfully! You can now run the experiments.")
        print("Try running: python main.py --experiment demo")
    else:
        print("\nSetup encountered errors. Please fix them before running experiments.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import shutil

def pip_install(package: str) -> bool:
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e.stderr.decode()}")
        return False

def ensure_dependencies_installed():
    """Ensure mypy and required type stubs are installed"""
    required_packages = [
        "mypy",
        "types-aiohttp",  # For async HTTP
        "types-pyyaml",   # For YAML config files
        "types-requests"  # For HTTP requests
    ]
    
    for package in required_packages:
        try:
            if package == "mypy":
                import mypy
                print("mypy is already installed")
                continue
            
            if pip_install(package):
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}, but continuing...")
                
        except ImportError:
            if not pip_install(package):
                print(f"Failed to install {package}, but continuing...")

def get_mypy_path() -> str:
    """Get the path to mypy executable"""
    if sys.platform == "win32":
        mypy_path = shutil.which("mypy.exe")
        if not mypy_path:
            mypy_path = str(Path(sys.executable).parent / "Scripts" / "mypy.exe")
    else:
        mypy_path = shutil.which("mypy")
        if not mypy_path:
            mypy_path = str(Path(sys.executable).parent / "bin" / "mypy")
    
    if not Path(mypy_path).exists():
        raise FileNotFoundError(f"Could not find mypy at {mypy_path}")
    
    return mypy_path

def run_mypy(paths: list[str]) -> tuple[int, str]:
    """Run mypy on specified paths"""
    try:
        ensure_dependencies_installed()
        mypy_path = get_mypy_path()
        
        cmd = [
            mypy_path,
            "--config-file=mypy.ini",
            "--show-error-codes",
            "--pretty",
            *paths
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout
        
    except FileNotFoundError as e:
        return 1, f"Error: {str(e)}"
    except subprocess.CalledProcessError as e:
        return 1, f"Error running mypy: {str(e)}"
    except Exception as e:
        return 1, f"Unexpected error: {str(e)}"

def main():
    # Paths to check
    project_root = Path(__file__).parent.parent
    paths_to_check = [
        str(project_root / "MAX"),
        str(project_root / "tests"),
    ]
    
    print("Running type checks...")
    return_code, output = run_mypy(paths_to_check)
    
    if return_code == 0:
        print("✅ No type errors found!")
        return 0
    else:
        print("❌ Type errors found:")
        print(output)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
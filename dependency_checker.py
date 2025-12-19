"""
Dependency Version Checker for Kokoro TTS Local
----------------------------------------------
This module checks if all required dependencies are installed and compatible.
"""

import sys
import importlib
import subprocess
from typing import Dict, List, Tuple, Optional
from packaging import version
import logging

logger = logging.getLogger(__name__)

# Required dependencies with minimum versions
REQUIRED_DEPENDENCIES = {
    'torch': '1.9.0',
    'kokoro': '0.9.2',
    'gradio': '3.0.0',
    'soundfile': '0.10.0',
    'huggingface_hub': '0.10.0',
    'pydub': '0.25.0',
    'numpy': '1.19.0',
    'pathlib': None,  # Built-in module
    'tqdm': '4.60.0'
}

# Optional dependencies
OPTIONAL_DEPENDENCIES = {
    'espeakng_loader': '0.1.0',
    'phonemizer': '3.0.0',
    'misaki': '0.1.0',
    'spacy': '3.0.0',
    'num2words': '0.5.0'
}

class DependencyChecker:
    """Check and validate dependencies"""
    
    def __init__(self):
        self.missing_required = []
        self.missing_optional = []
        self.version_conflicts = []
        self.warnings = []
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        min_python = (3, 8)
        current_python = sys.version_info[:2]
        
        if current_python < min_python:
            logger.error(f"Python {min_python[0]}.{min_python[1]}+ required, but {current_python[0]}.{current_python[1]} found")
            return False
        
        logger.info(f"Python version {current_python[0]}.{current_python[1]} is compatible")
        return True
    
    def get_package_version(self, package_name: str) -> Optional[str]:
        """Get installed version of a package"""
        try:
            module = importlib.import_module(package_name)
            # Try different version attributes
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    return getattr(module, attr)
            
            # For some packages, try getting version via pip
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', package_name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            return line.split(':', 1)[1].strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
            
            return "unknown"
            
        except ImportError:
            return None
    
    def check_dependency(self, package_name: str, min_version: Optional[str]) -> Tuple[bool, str]:
        """Check if a dependency is installed and meets version requirements"""
        installed_version = self.get_package_version(package_name)
        
        if installed_version is None:
            return False, f"{package_name} is not installed"
        
        if min_version is None:
            return True, f"{package_name} is installed (version: {installed_version})"
        
        try:
            if installed_version == "unknown":
                self.warnings.append(f"Could not determine version of {package_name}")
                return True, f"{package_name} is installed (version: unknown)"
            
            if version.parse(installed_version) >= version.parse(min_version):
                return True, f"{package_name} {installed_version} meets requirement (>= {min_version})"
            else:
                return False, f"{package_name} {installed_version} is too old (>= {min_version} required)"
                
        except Exception as e:
            self.warnings.append(f"Error checking version of {package_name}: {e}")
            return True, f"{package_name} is installed but version check failed"
    
    def check_all_dependencies(self) -> bool:
        """Check all required and optional dependencies"""
        logger.info("Checking dependencies...")
        
        # Check Python version first
        if not self.check_python_version():
            return False
        
        all_good = True
        
        # Check required dependencies
        logger.info("Checking required dependencies...")
        for package, min_ver in REQUIRED_DEPENDENCIES.items():
            is_ok, message = self.check_dependency(package, min_ver)
            
            if is_ok:
                logger.info(f"✓ {message}")
            else:
                logger.error(f"✗ {message}")
                self.missing_required.append(package)
                all_good = False
        
        # Check optional dependencies
        logger.info("Checking optional dependencies...")
        for package, min_ver in OPTIONAL_DEPENDENCIES.items():
            is_ok, message = self.check_dependency(package, min_ver)
            
            if is_ok:
                logger.info(f"✓ {message}")
            else:
                logger.warning(f"○ {message} (optional)")
                self.missing_optional.append(package)
        
        # Report warnings
        for warning in self.warnings:
            logger.warning(warning)
        
        return all_good
    
    def get_installation_commands(self) -> List[str]:
        """Get pip install commands for missing dependencies"""
        commands = []
        
        if self.missing_required:
            required_packages = []
            for package in self.missing_required:
                min_ver = REQUIRED_DEPENDENCIES.get(package)
                if min_ver:
                    required_packages.append(f"{package}>={min_ver}")
                else:
                    required_packages.append(package)
            
            if required_packages:
                commands.append(f"pip install {' '.join(required_packages)}")
        
        if self.missing_optional:
            optional_packages = []
            for package in self.missing_optional:
                min_ver = OPTIONAL_DEPENDENCIES.get(package)
                if min_ver:
                    optional_packages.append(f"{package}>={min_ver}")
                else:
                    optional_packages.append(package)
            
            if optional_packages:
                commands.append(f"pip install {' '.join(optional_packages)}  # Optional")
        
        return commands
    
    def check_cuda_availability(self) -> Dict[str, Any]:
        """Check CUDA availability and provide information"""
        cuda_info = {
            'available': False,
            'version': None,
            'device_count': 0,
            'devices': []
        }
        
        try:
            import torch
            cuda_info['available'] = torch.cuda.is_available()
            
            if cuda_info['available']:
                cuda_info['version'] = torch.version.cuda
                cuda_info['device_count'] = torch.cuda.device_count()
                
                for i in range(cuda_info['device_count']):
                    device_props = torch.cuda.get_device_properties(i)
                    cuda_info['devices'].append({
                        'id': i,
                        'name': device_props.name,
                        'memory': device_props.total_memory // (1024**3)  # GB
                    })
                
                logger.info(f"CUDA {cuda_info['version']} available with {cuda_info['device_count']} device(s)")
                for device in cuda_info['devices']:
                    logger.info(f"  Device {device['id']}: {device['name']} ({device['memory']}GB)")
            else:
                logger.info("CUDA not available, will use CPU")
                
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
        
        return cuda_info

def check_dependencies() -> bool:
    """Main function to check all dependencies"""
    checker = DependencyChecker()
    
    # Check dependencies
    all_good = checker.check_all_dependencies()
    
    # Check CUDA
    cuda_info = checker.check_cuda_availability()
    
    # Print summary
    if not all_good:
        logger.error("Some required dependencies are missing or incompatible!")
        logger.info("To install missing dependencies, run:")
        for cmd in checker.get_installation_commands():
            logger.info(f"  {cmd}")
        return False
    
    if checker.missing_optional:
        logger.info("Some optional dependencies are missing. The application will work but some features may be disabled.")
        logger.info("To install optional dependencies, run:")
        for cmd in checker.get_installation_commands():
            if "Optional" in cmd:
                logger.info(f"  {cmd}")
    
    logger.info("All required dependencies are satisfied!")
    return True

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    success = check_dependencies()
    sys.exit(0 if success else 1)
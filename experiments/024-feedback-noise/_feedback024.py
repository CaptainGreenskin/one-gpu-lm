"""Load experiment-local modules without colliding with frozen older modules."""
import importlib.util
from pathlib import Path
import sys

def module(name,path=None):
    key='feedback024_'+name
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,path or Path(__file__).with_name(name+'.py'))
        value=importlib.util.module_from_spec(spec);sys.modules[key]=value
        spec.loader.exec_module(value)
    return sys.modules[key]

"""Check pytorch_grad_cam cleanup API."""
import sys
sys.path.insert(0, '.')
import importlib.metadata
try:
    v = importlib.metadata.version("grad-cam")
    print(f"grad-cam version: {v}")
except Exception as e:
    print(f"Version not found: {e}")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
methods = [m for m in dir(GradCAM) if not m.startswith('__') or m in ('__exit__',)]
print(f"GradCAM attributes: {methods}")
print(f"Has remove_handlers: {hasattr(GradCAM, 'remove_handlers')}")
print(f"Has __exit__: {hasattr(GradCAM, '__exit__')}")

# Check BaseCAM for activations_and_grads
import inspect
src_lines = inspect.getsource(BaseCAM.__init__).split('\n')
print("BaseCAM.__init__ snippet:")
for line in src_lines[:25]:
    print(f"  {line}")

# Check __exit__
try:
    src_exit = inspect.getsource(BaseCAM.__exit__).split('\n')
    print("BaseCAM.__exit__:")
    for line in src_exit:
        print(f"  {line}")
except Exception as e:
    print(f"__exit__ not found: {e}")

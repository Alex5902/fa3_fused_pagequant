import os, sys, pathlib
from importlib.machinery import ExtensionFileLoader
from importlib.util import spec_from_file_location, module_from_spec

# 1) Preload PyTorch so its libs (libc10, libtorch_*) are already in memory
import torch as _torch  # noqa: F401

# (optional but helpful) add torch's lib dir to search path for future dlopen calls
try:
    torch_lib = pathlib.Path(_torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        os.environ["LD_LIBRARY_PATH"] = f"{str(torch_lib)}:{os.environ.get('LD_LIBRARY_PATH','')}"
except Exception:
    pass

# 2) Find the built .so and load it under the real init name `_C`
here = pathlib.Path(__file__).parent
cands = sorted(list(here.glob("fa3_cuda.cpython-*.so")) + list(here.parent.glob("fa3_cuda.cpython-*.so")))
if not cands:
    raise ImportError(f"fa3_cuda: no shared object found under {here} or {here.parent}")
so = str(cands[-1])

loader = ExtensionFileLoader("_C", so)
spec = spec_from_file_location("_C", so, loader=loader)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[attr-defined]

# 3) Publish as the package itself
sys.modules[__name__] = mod

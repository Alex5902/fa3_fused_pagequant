import importlib.util, importlib.machinery, pathlib, sys
here = pathlib.Path(__file__).parent
# grab the first fa3_cuda*.so in this folder
so = next(here.glob("fa3_cuda*.so"))
# load it under its real init name `_C`
spec = importlib.util.spec_from_file_location("_C", so)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)  # type: ignore
# publish as 'fa3_cuda'
sys.modules[__name__] = m
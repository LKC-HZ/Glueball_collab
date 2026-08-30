from color_wf_py_sparse import run_color_wf_py
import time

t0 = time.perf_counter()
out = run_color_wf_py(0, 0, 5)
t1 = time.perf_counter()
print("Done in %.2f s" % (t1 - t0))
print(out[:1000])

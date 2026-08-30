from color_wf_py_sparse import run_color_wf_py
import time

t0 = time.perf_counter()
try:
    out = run_color_wf_py(1, 1, 4)
    t1 = time.perf_counter()
    print("Done in %.2f s" % (t1 - t0))
    lines = out.split('\n')
    for line in lines[:10]:
        print(line)
except Exception as e:
    t1 = time.perf_counter()
    print("Failed after %.2f s" % (t1 - t0))
    print(type(e).__name__, e)

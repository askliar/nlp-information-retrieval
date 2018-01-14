import torch
prof = torch.autograd.profiler.load_nvprof('trace_name.prof')
print(prof)

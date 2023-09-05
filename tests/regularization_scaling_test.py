# #%%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.regularization import get_regs_dict
# import numpy as np
# import matplotlib.pyplot as plt
# regs = get_regs_dict()
# %load_ext autoreload
# %autoreload 2
# # %%
# reg_tracker = {}
# init = 5
# x = torch.randn(5, init, init, init)
# #
# d = 3
# for i in range(5):
#     for k, v in regs.items():
#         if k not in reg_tracker:
#             reg_tracker[k] = []
#         if d == 1:
#             reg_v = v(shape=x.shape[:-2], target=x[..., 0, 0], dims=(1,))(x[..., 0, 0]).item()
#         elif d == 2:
#             reg_v = v(shape=x.shape[:-1], target=x[..., 0], dims=(1,2))(x[..., 0]).item()
#         elif d == 3:
#             reg_v = v(shape=x.shape, target=x, dims=(1,2,3))(x).item()
#         else:
#             raise NotImplementedError
            
        
        
#         reg_tracker[k].append(reg_v)
#     x = torch.repeat_interleave(x, 2, dim=1)
#     x = torch.repeat_interleave(x, 2, dim=2)
# # %%
# plt.figure(figsize=(5, len(reg_tracker)*5))
# plt.suptitle(f"Stability of Regularization over Side Length ({d}D)")
# for k, v in reg_tracker.items():
#     #generate plot for each regularization
#     plt.subplot(len(reg_tracker), 1, list(reg_tracker.keys()).index(k)+1)
#     plt.plot(init*2**np.arange(len(v)), v)
#     plt.title(k)
#     plt.xlabel("Side Length")
#     plt.ylabel("Penalty")
#     maxv = max(max(v)*1.1, max(v)*0.9)
#     minv = min(v) if min(v) < 0 else 0
#     plt.ylim([minv, maxv])
# plt.tight_layout()
    
# # %%

import torch
import numpy as np
from scipy.stats import rankdata
from numpy import linalg as LA

# weights_file = '../results_segmentation/model_espnetv2_pascal/s_2.0_sch_hybrid_loss_ce_res_256_sc_0.5_2.0/20191130-153707/espnetv2_2.0_256_best.pth'
weights_file = '../model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_pascal_384x384.pth'
# weights_file = '../model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_pascal_256x256.pth'

a = torch.load(weights_file)

# # Weight pruning
# # for k in [.25, .50, .60, .70, .80, .90, .95 .97, .99]:
# for k in [0.0]:
# 	ranks = {}
# 	for l in a.keys():
# 		w = np.array(a[l].cpu())

# 		# print(l)
# 		# print(w.shape)

# 		ranks[l] = (rankdata(np.abs(w), method = 'dense') - 1).astype(int).reshape(w.shape)
# 		lower_bound_rank = np.ceil(np.max(ranks[l]) * k).astype(int)

# 		# print(lower_bound_rank)

# 		ranks[l][ranks[l] <= lower_bound_rank] = 0
# 		ranks[l][ranks[l] > lower_bound_rank] = 1
# 		w = w * ranks[l]
# 		# print(w)
# 		w = torch.from_numpy(np.asarray(w)).to('cuda:0')
# 		a[l] = w
# 		# exit()

# 	torch.save(a, 'weight_new_{}.pth'.format(k))

# # Neuron pruning
# for k in [.25, .50, .60, .70, .80, .90, .95, .97, .99]:
# 	ranks = {}
# 	for l in a.keys():
# 		w = np.array(a[l].cpu())
# 		norm = LA.norm(w, axis = 0)
# 		norm = np.tile(norm, (w.shape[0], 1))
# 		ranks[l] = (rankdata(norm, method = 'dense') - 1).astype(int).reshape(norm.shape)
# 		lower_bound_rank = np.ceil(np.max(ranks[l]) * k).astype(int)
# 		ranks[l][ranks[l] <= lower_bound_rank] = 0
# 		ranks[l][ranks[l] > lower_bound_rank] = 1
# 		w = w * ranks[l]
# 		w = torch.from_numpy(np.asarray(w)).to('cuda:0')
# 		a[l] = w

# 	torch.save(a, 'neuron_{}.pth'.format(k))		

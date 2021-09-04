import torch
import math
from tqdm import tqdm

def count_lipz(net, dataset, device, rand_times=64, eps=1e-2):
    sample_size = dataset.shape[0]
    length_fea = dataset.shape[1]
    net.eval()

    all_Lipz = 0
    for it in range(sample_size):
        while True:
            # pick a random example id
            rand_vector = eps * (2*torch.rand(rand_times, length_fea)-1).to(device)
            
            example_0 = dataset[it:it+1, :]
            example_1 = example_0 + rand_vector
    
            # do a forward pass on the example
            pred_0 = net(example_0)
            pred_1 = net(example_1)

            diff = pred_1-pred_0
            diff_0 = diff[:, 0].unsqueeze(1) / rand_vector
            diff_1 = diff[:, 1].unsqueeze(1) / rand_vector
            length_diff = torch.sum(torch.abs(diff_0) + torch.abs(diff_1), dim=1)

            # avoid divide 0
            zero = torch.zeros_like(length_diff)
            length_diff = torch.where(length_diff == math.inf, zero, length_diff)
            length_diff = torch.where(length_diff == math.nan, zero, length_diff)
            
            Local_Lipz = torch.max(length_diff, dim=0).values.item()
            if Local_Lipz>0:
                break
        
        all_Lipz += Local_Lipz / sample_size

    return all_Lipz
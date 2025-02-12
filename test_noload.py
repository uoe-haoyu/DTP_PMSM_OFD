from dataset import get_pathdata
import test
import model_repo
import torch
import random
import numpy as np
import os

def seed_everthing(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('Seed everthing {}'.format(seed))

if __name__ == '__main__':
    # seed_everthing(78)
    testdata_path = r'./wval_data/test_data.npy'
    os.makedirs('Validation/', exist_ok=True)
    save_path = r'Validation/'  # save predicted results


   # MLP_T
   #  pth = r'./Experiment1_used_MLP_Adam_0.001/epoch26.pth'
   #  netname = model_repo.MLP

    # Proposed
    pth = r'./Experiment1_used_MLP_cross_domain_Adam_0.001/epoch27.pth'
    netname = model_repo.MLP_cross_domain

    config = {
        'netname': netname.Net,
        'dataset': {'test': get_pathdata(testdata_path),},
        'pth_repo': pth,
        'test_path': save_path,
    }

    tester = test.Test(config)
    accuracys = []

    for i in range(1):
        print(f"Running test iteration {i + 1}...")
        accuracy = tester.start()
        accuracy = accuracy*100
        accuracys.append(accuracy)
        print(f"Test iteration {i + 1} completed.")

    accuracys = np.array(accuracys)
    print("\t".join(map(str, accuracys)))

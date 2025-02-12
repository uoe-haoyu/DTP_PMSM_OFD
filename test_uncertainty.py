import torch
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import torch.nn.functional as F


def enable_mc_dropout(model):
    """
    Enables MC Dropout by setting Dropout layers to training mode.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def compute_uncertainties(predictions):
    """
    Computes UQ, AU, and EU from MC Dropout predictions.
    Args:
        predictions: Tensor of shape [num_samples, batch_size, num_classes]
    Returns:
        uq: Total Uncertainty (Entropy of mean prediction)
        au: Aleatoric Uncertainty (Mean entropy of individual predictions)
        eu: Epistemic Uncertainty (UQ - AU)
    """
    mean_probs = predictions.mean(dim=0)  # Mean prediction probabilities [batch_size, num_classes]
    uq = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)  # Total Uncertainty

    individual_entropies = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=-1)  # Entropy per sample
    au = individual_entropies.mean(dim=0)  # Aleatoric Uncertainty

    eu = uq - au  # Epistemic Uncertainty
    return uq, au, eu

class Test:
    def __init__(self, configs):
        self.t_time = 0.0
        self.t_sec = 0.0
        self.net = configs['netname']('_')
        # self.net = configs['netname']()

        self.test = configs['dataset']['test']
        self.val_dataloader = torch.utils.data.DataLoader(self.test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=8)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pth = configs['pth_repo']
        self.sava_path = configs['test_path']
        self.print_staistaic_text = self.sava_path + 'print_staistaic_text.txt'

    def start(self):
        print("Loading .......   path:{}".format(self.pth))

        state = torch.load(self.pth, map_location=self.device)
        new_state_dict = {}
        for key, value in state['model'].items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the prefix.
            else:
                new_key = key
            new_state_dict[new_key] = value

        self.net.load_state_dict(new_state_dict)
        self.net.to(self.device)
        test_normstatic=1
        accuracy = self.val_step(test_normstatic,self.pth[-5],self.val_dataloader)

        return accuracy

    # num_mc_samples, Change the iteration times
    def val_step(self,test_normstatic, epoch, dataset,num_mc_samples=50):
        print('-----------------start test--------------------')


        self.csv_onlylable = []
        self.net = self.net.eval()
        enable_mc_dropout(self.net)  # Enable MC Dropout
        star_time = time.time()

        for i, data in enumerate(dataset):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            # Collect predictions for MC Dropout
            mc_predictions = []
            for _ in range(num_mc_samples):
                with torch.no_grad():
                    prediction = F.softmax(self.net(images), dim=-1)  # Use softmax for probabilities
                    mc_predictions.append(prediction)

            # Stack predictions and compute uncertainties
            mc_predictions = torch.stack(mc_predictions)
            uq, au, eu = compute_uncertainties(mc_predictions)

            # Obtain the predicted labels (most probable class) 50 times calculate
            p1 = torch.argmax(mc_predictions.mean(dim=0), dim=1).to(self.device)  # Operate on mean prediction
            l1 = labels.to(self.device)

            # Save labels and predictions to CSV-like format
            temp_onlylable = torch.cat([l1, p1, uq, au, eu], dim=-1)
            self.csv_onlylable.append(temp_onlylable.cpu().detach().numpy().squeeze())


        duration = time.time() - star_time
        speed = 1 / (duration / len(dataset))
        print('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle = open(self.print_staistaic_text, mode='a')

        file_handle.write('-----------------start test--------------------')
        file_handle.write('\n')
        file_handle.write('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle.write('\n')
        file_handle.close()

        self.net = self.net.train()
        accuracy = self.tocsv_onlylable(epoch)

        print('-----------------test over--------------------')

        return accuracy





    def tocsv_onlylable(self, epoch):

        np_data = np.array(self.csv_onlylable)

        label = np_data[:,0]
        pred = np_data[:,1]

        uq = np_data[:,2]
        au = np_data[:,3]
        eu = np_data[:,4]

        # Print the mean
        print("Mean of uq:", np.mean(uq))
        print("Mean of au:", np.mean(au))
        print("Mean of eu:", np.mean(eu))

        # Calculate Metrics
        precision = precision_score(label, pred, average='macro')  # 'micro', 'weighted'
        recall = recall_score(label, pred, average='macro')  # 'micro', 'weighted'
        f1 = f1_score(label, pred, average='macro')
        accuracy = accuracy_score(label, pred)

        #
        print(
            'epoch:{} 测试accuracy:{}'.format(epoch,
                        accuracy)
        )
        print(
            'epoch:{} 测试precision:{}'.format(epoch,
                        precision)
        )
        print(
            'epoch:{} 测试recall:{}'.format(epoch,
                        recall)
        )
        print(
            'epoch:{} 测试f1:{}'.format(epoch,
                        f1)
        )


        file_handle = open(self.print_staistaic_text, mode='a')


        file_handle.write('epoch:{}测试accuracy:{}'.format(epoch,
                                                    accuracy ))
        file_handle.write('\n')

        file_handle.write('epoch:{}测试precision:{}'.format(epoch,
                                                    precision ))
        file_handle.write('\n')
        file_handle.write('epoch:{}测试recall:{}'.format(epoch,
                                                    recall))
        file_handle.write('\n')
        #
        file_handle.write('epoch:{}测试f1:{}'.format(epoch,f1))
        file_handle.write('\n')

        file_handle.write('-----------------测试结束--------------------')
        file_handle.write('\n')

        file_handle.close()

        np.savetxt(self.sava_path + str(epoch)+'_pred_onlylable.csv', np_data, delimiter=',')

        return accuracy

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import matplotlib.pyplot as plt


class client(object):
    def __init__(self, trainDataset, dev):
        self.train_dataset = trainDataset
        self.dev = dev
        self.train_dataloader = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, model, lossFun, optimizer, global_parameters):
        model.load_state_dict(global_parameters, strict=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dataloader:
                data, label = data.to(self.dev), label.to(self.dev)
                predictions = model(data)
                loss = lossFun(predictions, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict()

    def local_val(self):
        pass


class clientsGroup(object):
    def __init__(self, datasetName, alpha, numOfClients, dev):
        self.dataset_name = datasetName
        self.alpha = alpha
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.localDataset(alpha)

    def localDataset(self, alpha):
        mnistDataSet = GetDataSet(self.dataset_name, self.alpha)
        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.tensor(mnistDataSet.test_label)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        shard_size = mnistDataSet.train_data_size // self.num_of_clients
        
        if alpha == 0:
            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label
            shards_ids = np.random.permutation(self.num_of_clients)
            for i in range(self.num_of_clients):
                shards_id = shards_ids[i]
                local_data = train_data[shards_id * shard_size: shards_id * shard_size + shard_size]
                local_label = train_label[shards_id * shard_size: shards_id * shard_size + shard_size]
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
        else:
            if alpha == 0:
                alpha = 0.001
            print('Non-IID setting')
            dirichlet_pdf = np.random.dirichlet([alpha / 10] * 10, self.num_of_clients)
            train_data = mnistDataSet.train_data

            for i in range(self.num_of_clients):
                local_pdf = np.floor(dirichlet_pdf[i]*shard_size).astype('int64')
                local_label = []
                local_data = []
                pdf_list = []

                start_index = 0
                for k in range(10):
                    pdf_list.append((start_index, local_pdf[k]))
                    start_index += local_pdf[k]
                plt.broken_barh(pdf_list,
                                (i * 4, 3),
                                facecolors=(
                                "b", "#FF7F50", "g", "r", "purple", "#8B4513", "#FFC0CB", "#808080", "#FFD700",
                                "#00FFFF"),
                                alpha=1)

                for label_num, label_value in zip(local_pdf, range(10)):
                    rand_arr = np.arange(train_data[label_value].shape[0])
                    np.random.shuffle(rand_arr)
                    if label_num > 1:
                        related_label = np.array([label_value]*label_num, dtype='int64')
                        random_data = train_data[label_value][rand_arr[0:label_num]]
                        if np.size(local_data) == 0:
                            local_label = related_label
                            local_data = random_data
                        else:
                            local_label = np.hstack((local_label, related_label))
                            local_data = np.vstack((local_data, random_data))
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
            plt.xlim((0, 295))
            plt.axis('off')
            plt.savefig("./result/alpha_" + format(alpha) + ".png")

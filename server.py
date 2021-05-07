import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from model import CNN
from clients import clientsGroup
import csv
import time

path = './result'
database = 'mnist'
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100)
parser.add_argument('-pf', '--participants_fraction', type=float, default=0.1,
                    help='the proportion of participants in all of cliants')
parser.add_argument('-e', '--local_epoch', type=int, default=5, help='local epoch round of one participant')
parser.add_argument('-b', '--batch_size', type=int, default=10, help='local train batch size')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01)
parser.add_argument('-dc', "--learning_rate_decay", type=float, default=0.9934)
parser.add_argument('-comm', '--num_of_communication_rounds', type=int, default=200)
parser.add_argument('-alpha', '--alpha', type=float, default=10, help='Dirichlet distribution concentration parameter, '
                                                                     'if set to 0, use iid distribution')
parser.add_argument('-sb', '--use_shared_batch', type=bool, default=False)


def get_args(args):
    gpu_num = args['gpu']
    total_clients = args['num_of_clients']
    participants_fraction = args['participants_fraction']
    local_epoch = args['local_epoch']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    learning_rate_decay = args['learning_rate_decay']
    total_round = args['num_of_communication_rounds']
    alpha = args['alpha']
    use_shared_batch = args['use_shared_batch']
    return gpu_num, total_clients, participants_fraction, local_epoch, batch_size, learning_rate, learning_rate_decay\
        , total_round, alpha, use_shared_batch


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def create_excel_file(path, alpha):
    record_time = time.strftime("_%Y-%m-%d-%H-%M-%S", time.localtime())
    if alpha == -1:
        file_name = path + "/FedAVG_iid" + record_time + '.csv'
    else:
        file_name = path + "/FedAVG_alpha-" + format(alpha) + record_time + '.csv'
    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'accuracy'])
        csvfile.close()
    return file_name


def update_excel_file(path, round_accuracy, last_time, current_time):
    record_list = [current_time - begin_time + last_time, format(round_accuracy)]
    with open(path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(record_list)
        csvfile.close()


def activate_gpu(model, gpu_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    return model.to(dev), dev


def initial_model(model):
    parameters = {}
    for model_key, value in model.state_dict().items():
        parameters[model_key] = value.clone()
    return parameters


def update_aggregated_parameters(parameters, new_parameters):
    if parameters is None:
        parameters = {}
        for key, value in new_parameters.items():
            parameters[key] = value.clone()
    else:
        for value in parameters:
            parameters[value] = parameters[value] + new_parameters[value]
    return parameters


def update_global_parameters(parameters, aggregated_parameters, participants_num):
    for value in global_parameters:
        parameters[value] = (aggregated_parameters[value] / participants_num)
    return parameters


def get_round_result(model, global_parameters, testDataLoader):
    with torch.no_grad():
        model.load_state_dict(global_parameters, strict=True)
        accuracy = 0
        num = 0
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            prediction = model(data)
            prediction = torch.argmax(prediction, dim=1)
            accuracy += (prediction == label).float().mean()
            num += 1
        accuracy = accuracy / num
        print('accuracy: {}'.format(accuracy))
    return accuracy


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__
    test_mkdir(path)
    gpu_num, total_clients, participants_fraction, local_epoch, batch_size, learning_rate, learning_rate_decay \
        , total_round, alpha, use_shared_batch = get_args(args)
    excel_file_name = create_excel_file(path, alpha)
    participants_num = int(max(total_clients * participants_fraction, 1))
    loss_function = F.cross_entropy
    model = CNN()
    model, dev = activate_gpu(model, gpu_num)
    clients_group = clientsGroup(database, alpha, total_clients, dev, use_shared_batch)
    testDataLoader = clients_group.test_data_loader
    global_parameters = initial_model(model)
    last_time = 0

    for i in range(total_round):
        print("Round {}".format(i + 1))
        order = np.random.permutation(total_clients)
        participants = ['client{}'.format(i) for i in order[0:participants_num]]
        aggregated_parameters = None
        begin_time = time.time()
        opti = optim.SGD(model.parameters(), lr=learning_rate)

        for participant in participants:
            local_parameters = clients_group.clients_set[participant].localUpdate(local_epoch, batch_size, model,
                                                                                  loss_function, opti, global_parameters)
            aggregated_parameters = update_aggregated_parameters(aggregated_parameters, local_parameters)

        global_parameters = update_global_parameters(global_parameters, aggregated_parameters, participants_num)
        get_round_result(model, global_parameters, testDataLoader)
        round_accuracy = get_round_result(model, global_parameters, testDataLoader)

        current_time = time.time()
        update_excel_file(excel_file_name, round_accuracy, last_time, current_time)
        last_time = current_time - begin_time + last_time
        learning_rate *= learning_rate_decay

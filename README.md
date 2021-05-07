# FedAvg with non-IID dataset of Dirichlet distribution

Some codes refers to https://github.com/WHDY/FedAvg

### Environment

1.python3.7.6

2.pytorch1.8.0

3.cuda10.0

4.cudnn7


### Input

Example:
```asp
python server.py -nc 100 -pf 0.1 -e 5 -b 10 -comm 400 -alpha 1 -lr 0.01  -g 0
```

- -nc: number of total clients
- -pf: number of seleted participants in each round
- -e: local epoches
- -b: local batch size
- -comm: total communication round
- -alpha: if alpha=0, use IID dataset, if alpha>0, use non-IID dataset, alpha is the Dirichlet concentration parameters, with alpha increase, the non-IID level decrease
- -lr: learning rate
- -g: GPU which you want to use


Above code means: 
1. Total 100 clients, in each communicating round randomly select 10 clients as participants. 
2. In a round, a participant trains the local model for 5 epochs, and the local training data batch is 10. 
3. alpha is 1, which means data in participants with high different
4. The learning rate is 0.01, training stops after 200 rounds.

### Output results

The training result is exported into csv files, you can find it at ./result

### Reference

[1] Mcmahan H B , Moore E , Ramage D , et al. Communication-Efficient Learning of Deep Networks from Decentralized Data[J]. 2016.

[2] Hsu, T. M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335.

[3]  Li, X., Huang, K., Yang, W., Wang, S., \& Zhang, Z. (2019). On the convergence of fedavg on non-iid data. arXiv preprint arXiv:1907.02189.

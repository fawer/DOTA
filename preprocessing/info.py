import numpy as np
import pandas as pd


def read():
    with open('../dataset/disease_dict') as f, open('../dataset/AD-osc-gene-list-072721.txt') as f1, \
            open("../dataset/drugNets/drugProtein.txt") as f2, open('../dataset/drug_dict') as f3:
        # prepare disease_dict
        de_d = f.read().split('\n')
        de_d = [i.split(":") for i in de_d[:-1]]
        de_d = list(np.array(de_d)[:, 1])
        de_dic = {k: i for i, k in enumerate(de_d)}
        # prepare drug dictionary
        d_d = f3.read().split('\n')
        d_dic = {k: i for i, k in enumerate(d_d)}
        d_i_dic = {i: k for i, k in enumerate(d_d)}
        return de_dic, d_dic, d_i_dic


if __name__ == '__main__':
    # get disease and drug dic
    de_dic, d_dic, d_i_dic = read()
    # get ad index
    ad_index = de_dic["Alzheimer's Disease"]
    d_index = d_dic['DB01132:Pioglitazone']
    r_index = d_dic['DB01197:Captopril']
    # prepare circadian dataframe
    # column ['Drug_ID', 'Circadian', 'Alzheimers', 'Target']
    df = pd.read_csv('../drug_cir.csv', delimiter=',')
    # load old drug-disease net
    data = np.loadtxt('../dataset/drugDisease.txt')
    data = data.transpose()
    # validate Alzheimer is loc
    for index, item in enumerate(data[ad_index]):
        if item == 1:
            print(d_i_dic[index])

    # find cir and Alz drug
    k = df.loc[df['Circadian'] == 1].loc[df['Alzheimers'] == 1]['Drug_ID'].values
    k = [d_dic[i] for i in k]
    k.append(d_index)
    for i in k:
        data[ad_index][i] = 2

    temp, temp1 = [], []
    with open("../dataset/drugNets/drugProtein.txt") as f:
        da = f.read().split('\n')
        de_d = da[d_index].split()
        de_d = np.array(de_d, dtype=int)
        temp = np.where(de_d == 1)
        de_d = da[r_index].split()
        de_d = np.array(de_d, dtype=int)
        temp1 = np.where(de_d == 1)
    print(temp)
    print(temp1)
    C = np.intersect1d(temp, temp1)
    print(C)
    # np.save('./dataset/new_dd.npy', data[ad_index])






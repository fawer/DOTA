import numpy as np
import csv

# get drug dictionary

def read():
    with open('./dataset/disease_dict') as f, open('./dataset/AD-osc-gene-list-072721.txt') as f1, \
            open("./dataset/drugNets/drugProtein.txt") as f2, open('./dataset/drug_dict') as f3:
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


def evl(feature, index):
    drug_index = [742, 819, 1369]
    ans = []
    index_norm = np.linalg.norm(feature[index])
    for fea in drug_index:
        fea_norm = np.linalg.norm(feature[fea])
        ans.append(np.dot(feature[fea], feature[index].reshape(-1, 1))/(index_norm * fea_norm))
    return sum(ans)/3


def dis_evl(index):
    with open('./dataset/protein_dict') as p_f, open('./dataset/AD-osc-gene-list-072721.txt') as f1,\
            open("./dataset/drugNets/drugProtein.txt") as f2:
        p_d = p_f.read().split('\n')
        p_d = {k.split(':')[-1]: i for i, k in enumerate(p_d)}
        a_p = f1.read().split('\n')
        ans = []
        for i in a_p:
            if i in p_d.keys():
                ans.append(p_d[i])
        d_p_n = f2.read().split('\n')
        focus = d_p_n[index].split('\t')
        if any([focus[i] == '1' for i in ans]):
            print(index)


def get_target(index):
    with open('./dataset/protein_dict') as p_f, open('./dataset/disease_dict') as f, open('./dataset/se_dict') as f1:
        p_d = p_f.read().split('\n')
        p_d = {i: k.split(':')[-1]for i, k in enumerate(p_d)}

        d_d = f.read().split('\n')
        d_d = {i: k.split(':')[-1] for i, k in enumerate(d_d)}

        s_d = f1.read().split('\n')
        s_d = {i: k.split(':')[-1] for i, k in enumerate(s_d)}

        dd = np.loadtxt('./dataset/drugDisease.txt').astype(np.float32)
        dp = np.loadtxt("./dataset/drugNets/drugProtein.txt").astype(np.float32)
        sd = np.loadtxt("./dataset/drugNets/drugsideEffect.txt").astype(np.float32)
        temps = dp[index]
        d_index = dd[index]
        target, diseases, re_target = [], [], []
        target = [p_d[i] for i in np.where(temps == 1)[0].tolist()]
        se = [s_d[i] for i in np.where(sd[index] == 1)[0].tolist()]
        diseases = np.where(d_index == 1)[0]
        # re_drug = np.where(dd[:, diseases] == 1)[0]
        # re_drug = set(re_drug)
        # for i in re_drug:
        #     re_target.append(np.where(dp[i] == 1))

        return target, [d_d[i] for i in diseases.tolist()], se


if __name__ == '__main__':

    k = np.load('predict/score_wl_0_oaz.npy').reshape(1, -1)
    print(k.max(), k.min())
    de_dic, d_dic, dr_dic = read()

    r = 'DB00734:Risperidone'

    a = 'DB01238:Aripiprazole'

    s = 'DB00203:Sildenafil'

    r_index, a_index, s_index = d_dic[r], d_dic[a], d_dic[s]

    ad_index = de_dic["Alzheimer's Disease"]

    results = [1193, 805, 1043, 1108, 103, 107, 853, 1249, 1220, 877]

    # temp = np.argpartition(k[ad_index], -20)[-20:]

    # get drug feature
    features = np.loadtxt('./drugmdaFeatures.txt')
    head = ['name', 'score', 'target', 'diseases', 'side effect']
    # get results from w_vae
    temp = np.argpartition(k[0], -20)[-20:]
    print(min(temp))
    candi = []
    for i in temp:
        # relate = evl(features, i)
        tar, dis, se = get_target(i)
        candi.append({'name': dr_dic[i], 'score': str(k[0][i])[:4], 'target': tar, 'diseases': dis, 'side effect': se})

    for i in results:
        print(str(k[0][i])[:4])

    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=head)

        writer.writeheader()
        writer.writerows(candi)





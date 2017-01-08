import numpy as np

from s2v import J2V

def get_data():
    with open('./data/sentiment.txt', 'r') as f:
        datas =  f.readlines()
        s2v = J2V('alphabet', max_len=2000, length=1024)
        datasets = []
        for data in datas:
            line, t = data.strip().split(',')
            t_set = [0] * 2
            t_set[int(t)] = 1
            d = s2v.encode_sentences([line])[0]
            d_set = []
            for x in d:
                d_set += x
            # datasets.append(tuple([np.array(d_set, dtype=np.float32), int(t)]))
            datasets.append(tuple([np.array(d_set, dtype=np.float32), t_set]))


    return [ datasets[:10], datasets[10:]]

if __name__ == '__main__':
    train, test = get_data()

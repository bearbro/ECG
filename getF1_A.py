from config import config


def read_labels(filename):
    with open(filename, 'r') as f:
        readlines = f.readlines()
        id2labels = {}
        for line in readlines:
            value = list(line[:-1].split('\t'))
            if value[-1] == '':
                value = value[:-1]
            id2labels[value[0]] = value[3:]
        f.close()
    return id2labels


def get_scoreA(args):
    # name_1='subA_202003071034.txt'
    print(args.subA)
    name_1 = args.subA
    save = args.save
    pred = read_labels(name_1)
    val = read_labels('data/hefei_round1_ansA_20191008.txt')
    true_n = {}  # val有pred有
    false_n = {}  # val有pred无
    list_name = []
    for line in open(config.arrythmia, encoding='utf-8'):
        list_name.append(line.strip())
    indx2name = {i: name for i, name in enumerate(list_name)}
    for k, v in pred.items():
        v_val = val[k]
        for i in v_val:
            if args.tag:
                x = [indx2name[ii] for ii in config.top4_tag_list]
                if i not in x:
                    continue
            if i in v:
                true_n[i] = true_n.get(i, 0) + 1
            else:
                false_n[i] = false_n.get(i, 0) + 1
    true_n2 = {}  # val有pred有
    false_n2 = {}  # val无pred有
    for k, v in pred.items():
        v_val = val[k]
        for i in v:
            if i in v_val:
                true_n2[i] = true_n2.get(i, 0) + 1
            else:
                false_n2[i] = false_n2.get(i, 0) + 1

    TP = sum(list(true_n.values()))
    FN = sum(list(false_n.values()))
    FP = sum(list(false_n2.values()))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('\tval有pred有\tval有pred无\tval有pred有\tval无pred有\n')
    print('\tTP\tFN\tFP\tP\tR\t%.4f\n' % F1)
    kk = list(set(true_n) | set(false_n))
    kk.sort()
    for k in kk:
        TP = true_n.get(k, 0)
        FN = false_n.get(k, 0)
        FP = false_n2.get(k, 0)
        if (TP + FP) == 0:
            P = 0
        else:
            P = TP / (TP + FP)
        if (TP + FN) == 0:
            R = 0
        else:
            R = TP / (TP + FN)
        if P + R == 0:
            F1 = 0
        else:
            F1 = 2 * P * R / (P + R)
        print('%5d\t%5d\t%5d\t%.3f\t%.3f\t%.3f\t%s\n' % (TP, FN, FP, P, R, F1, k))
    if save:
        with open('different_%s.txt' % name_1, 'w') as f:
            TP = sum(list(true_n.values()))
            FN = sum(list(false_n.values()))
            FP = sum(list(false_n2.values()))
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            f.write('\tval有pred有\tval有pred无\tval有pred有\tval无pred有\n')
            f.write('\tTP\tFN\tFP\tP\tR\t%.4f\n' % F1)
            kk = list(set(true_n) | set(false_n))
            kk.sort()
            for k in kk:
                TP = true_n.get(k, 0)
                FN = false_n.get(k, 0)
                FP = false_n2.get(k, 0)
                if (TP + FP) == 0:
                    P = 0
                else:
                    P = TP / (TP + FP)
                if (TP + FN) == 0:
                    R = 0
                else:
                    R = TP / (TP + FN)
                if P + R == 0:
                    F1 = 0
                else:
                    F1 = 2 * P * R / (P + R)
                f.write('%5d\t%5d\t%5d\t%.3f\t%.3f\t%.3f\t%s\n' % (TP, FN, FP, P, R, F1, k))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subA", type=str, help="the path of model weight file")
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--tag", action='store_true', default=False)  # 只考虑部分
    args = parser.parse_args()
    get_scoreA(args)

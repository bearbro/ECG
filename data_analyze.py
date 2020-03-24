import torch

from config import config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dd = torch.load(config.train_data)
    # dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx, 'wc': wc, 'file2age': file2age,
    #       'file2sex': file2sex}

    plt.bar(range(len(dd['wc'])), dd['wc'])
    plt.savefig('fig/all_wc')
    plt.show()

    plt.bar(range(len(dd['wc']))[20:], dd['wc'][20:])
    plt.savefig('fig/20_wc')  # 20 204 0.005
    plt.show()
    plt.bar(range(len(dd['wc']))[10:], dd['wc'][10:])
    plt.savefig('fig/10_wc')  # 30 43 0.001
    plt.show()
    # 33
    tag_list1 = list(range(0, 31 + 1)) + [46]  # 30 43 0.001
    plt.bar(tag_list1, dd['wc'][tag_list1])
    plt.savefig('fig/tag1_wc')
    plt.show()
    # 20
    tag_list2 = list(range(0, 20))  # 18 350 0.0086
    plt.bar(tag_list2, dd['wc'][tag_list2])
    plt.savefig('fig/tag2_wc')
    plt.show()

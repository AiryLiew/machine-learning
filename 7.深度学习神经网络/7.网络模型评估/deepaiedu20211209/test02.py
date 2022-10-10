import numpy as np

def voc_ap(rec, prec, use_07_metric=False):
    """
    Ap = voc_ap(rec, prec， [use_07_metric])
    计算给定精度和召回率的VOC AP。
    如果use_07_metric为true，则使用
    VOC 07 11点方法(默认值:False)。
    """

    # 针对2007年以前的计算方法，针对VOC数据集，使用的11个点计算AP，现在不使用
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 2007年以后的AP计算方法
        # 在rec和pre的前后添加标记值，形成闭合的值域
        mrec = np.concatenate(([0.], rec, [1.]))  #[0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
        mpre = np.concatenate(([0.], prec, [0.])) #[0.  1.,     0.6666, 0.4285, 0.3043,  0.]

        # 计算精度包络线
        # 计算出precision的各个断点(折线点)，找出区域内最大的pre值代表这个区域
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #[0.     1.     0.6666 0.4285 0.3043 0.    ]

        # 获取召回率不同的值，去除重复的点
        i = np.where(mrec[1:] != mrec[:-1])[0]  #前后两个值不一样的点
        # print(mrec[1:], "\n",mrec[:-1])
        # print(i) #[0, 1, 3, 4, 5]

        # AP= AP1 + AP2+ AP3+ AP4，
        # 获取每个区间段的召回率和对应的准确率，做乘积，求取当前区域的AP值，最后累加所有区域的AP值
        # print((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == '__main__':

    rec = [0.0666, 0.1333, 0.4   , 0.4666]
    pre = [1.,     0.6666, 0.4285, 0.3043]
    ap = voc_ap(rec, pre)

    print(ap) #输出：0.2456
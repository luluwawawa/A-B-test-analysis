import random
import numpy as np


def sharp_null_hypothesis_pvalue(data, mc_num=10000):
    '''Compute pvalue in sharp null hypothesis.

    Parameters
    ----------
    data_raw : dataframe
        a dataframe contains numerator, denominator and groupid
    mc_num : int
        numeber of monte carlo

    Returns
    ----------
    pvalue : float
        pvalue of hypothesis
    '''
    # Step 1:计算实验观测到的对照组和实验组的平均值的差
    obs_statistic = (
        data[data.groupid == 1].numerator.sum() / data[data.groupid == 1].denominator.sum() - 
        data[data.groupid == 0].numerator.sum() / data[data.groupid == 0].denominator.sum()
    )
    
    # Step 2: 重采样
    mc_statistic = [0 for _ in range(10000)]
    for _ in range(mc_num):
        # 无放回采样
        data["mc_groupid"] = random.sample(list(data.groupid.values), len(data))
        mc_statistic.append(
            data[data.mc_groupid == 1].numerator.sum() / data[data.mc_groupid == 1].denominator.sum() - 
            data[data.mc_groupid == 0].numerator.sum() / data[data.mc_groupid == 0].denominator.sum()
        )

    # Step 3: 计算p值 
    p_value =  np.mean([abs(val) > abs(obs_statistic) for val in mc_statistic])
    return p_value
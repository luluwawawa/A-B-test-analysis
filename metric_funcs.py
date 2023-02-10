#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Base classes for metric using buckets.'''
import math
import numpy as np
from typing import List
from scipy import stats


class Metric:
    '''A class that contains information of bucket data.

    A Metric contains 2 lists: sums and cnts. The sums contains the summation
    of numerator in each bucket. The cnts contains the summation of denominator
    in each bucket.
    '''
    def __init__(self, sums: List[float], cnts: List[float]):
        '''Initialize metric class.
        Compute the sums of numerator, denominator and cluster number.
        Compute the mean.

        Parameters
        ----------
        sums: list
            numerator in each bucket, usually is the metric
        cnts: list
            denominator in each bucket, usually is the total size
        '''
        self.sums = sums
        self.cnts = cnts
        # calculate the sum of numerator
        self.cnt = sum(cnts)
        # calculate the sum of denominator
        self.sum = sum(sums)
        # calculate the sample mean
        self.mean = self.sum / self.cnt

    def cov(self, list1, list2):
        '''Compute the covariance of two lists.

        Parameters
        ----------
        list1 : list
            the first list
        list2 : list
            the second list

        Returns
        ----------
        covariance : float
            the covariance of two lists
        '''
        assert(len(list1) == len(list2))
        size = len(list1)
        covar = (
            1.0 * size / (size-1) *
            (sum(map(lambda x: x[0]*x[1], zip(list1, list2)))/size -
             sum(list1)/size*sum(list2)/size)
        )
        return covar

    def delta_var(self):
        '''Compute the variance of the sample mean by delta method.'''
        num_bucket = len(self.sums)
        mu_d1, mu_d2 = sum(self.cnts)/num_bucket, sum(self.cnts)/num_bucket
        mu_n1, mu_n2 = sum(self.sums)/num_bucket, sum(self.sums)/num_bucket

        cov_n1_n2 = self.cov(self.sums, self.sums)
        cov_d1_d2 = self.cov(self.cnts, self.cnts)
        cov_n1_d2 = self.cov(self.sums, self.cnts)
        cov_n2_d1 = self.cov(self.cnts, self.sums)
    
        results = (
            1.0 / num_bucket *
            (
                cov_n1_n2 / mu_d1 / mu_d2 +
                cov_d1_d2 * mu_n1 * mu_n2 / pow(mu_d1*mu_d2, 2.0) -
                cov_n1_d2 * mu_n2 / mu_d1 / pow(mu_d2, 2.0) -
                cov_n2_d1 * mu_n1 / mu_d2 / pow(mu_d1, 2.0)
            )
        )

        return results

    def delta_std(self):
        '''Compute the variance of the population standard deviation by
        delta method.'''
        return math.sqrt(self.delta_var() * sum(self.cnts))

    def jackknife_var(self):
        '''Compute the variance of the sample mean by jackknife method.'''
        num_bucket = len(self.sums)
        sum_sums = sum(self.sums)
        sum_cnts = sum(self.cnts)
        del_one = np.zeros(num_bucket)
        for iter in range(num_bucket):
            del_one[iter] = (
                (sum_sums-self.sums[iter]) / (sum_cnts-self.cnts[iter])
            )
        del_one_mean = np.mean(del_one)
        jackknife_var = (
            (sum(del_one**2) - num_bucket*del_one_mean**2) *
            (num_bucket-1) / num_bucket
        )

        return jackknife_var

    def jackknife_std(self):
        ''''
        Compute the variance of the population standard deviation by
        jackknife method.
        '''
        return math.sqrt(self.jackknife_var() * sum(self.cnts))


class Ttest:
    '''A class that contains information of two-group two-side ratio t test.
    H0: (mu1 - mu0)/ mu0 = 0'''
    def __init__(self, x0, x1, alpha=0.05, power=0.8, delta=0.01):
        '''
        Initialize data and t test settings. x0 and x1 are of Metric class which
        contains bucket data. alpha, power and delta are parameters in t-test.

        After receiving parameters, compute the means in control group and
        treatment group, the absolute difference and relative difference of the
        two means, the population standard deviations of two groups, the
        adjusted variance of the absolute difference, and the standard error of
        the absolute difference.

        Parameters:
        ----------
        x0 : Metric class
            buckets of control group
        x1 : Metric class
            buckets of treatment group
        alpha : double
            significance level.
        power : double
            power of test, 1 - type II error
        delta : double
            sensitiveness, relative difference that want to test.
        '''
        self.x0 = x0
        self.x1 = x1
        self.alpha = alpha
        self.power = power
        self.delta = delta

        # calculate mean of each group
        self.p0 = self.x0.mean
        self.p1 = self.x1.mean
        self.p0_ori = self.p0
        self.p1_ori = self.p1
        self.abs_diff = self.p1 - self.p0
        self.abs_diff_ori = self.abs_diff
        if self.p0 != 0:
            self.rela_diff = self.abs_diff / abs(self.p0)
        else:
            self.rela_diff = float("inf")
        self.rela_diff_ori = self.rela_diff

        # calculate standard error
        self.std0 = self.x0.delta_std()
        self.std1 = self.x1.delta_std()
#         self.std0 = self.x0.jackknife_std()
#         self.std1 = self.x1.jackknife_std()
        self.std0_ori = self.std0
        self.std1_ori = self.std1
        if self.p0 != 0:
            self.var = self.x0.delta_var() + self.x1.delta_var()
#             self.var = self.x0.jackknife_var() + self.x1.jackknife_var()
        else:
            self.var = float("inf")
        self.se = math.sqrt(self.var)

    def pvalue(self):
        '''Calculate p value of t test.'''
        t_stat = (self.p1-self.p0) / self.se
        p_value = (1 - stats.norm.cdf(abs(t_stat))) * 2
        return p_value

    def test_power(self):
        '''Calculate power of t test.'''
        temp_value = self.p0_ori * self.delta / self.se
        cdf_1 = stats.norm.cdf(stats.norm.ppf(1-self.alpha/2) - temp_value)
        cdf_2 = stats.norm.cdf(stats.norm.ppf(self.alpha/2) - temp_value)
        power_test = 1 - cdf_1 + cdf_2
        return power_test

    def recom_sample_size(self):
        '''
        Calculate recommend sample size to detect a specific delta with
        specific power.
        '''
        std0 = self.std0
        std1 = self.std1
        b = std0 / std1
        a = self.x0.cnt / self.x1.cnt
        alpha_power = (
            stats.norm.ppf(1 - self.alpha/2) - stats.norm.ppf(1 - self.power)
        )
        recomsamplesize = (
            ((b**2 + a) / a) * (std1/self.p0_ori)**2 * alpha_power**2 /
            self.delta**2
        )
        return int(np.ceil(recomsamplesize))

    def confidence_interval(self):
        '''Compute confidence invervel of relative difference.'''
        ci_width = stats.norm.ppf(1-self.alpha/2) * self.se / abs(self.p0_ori)
        rela_diff = str(round(self.rela_diff*100, 3)) + '%'
        ci_width = str(round(ci_width*100, 3)) + '%'
        ci_str = rela_diff + ' +- ' + ci_width
        return ci_str

    def abs_confidence_interval(self):
        '''Compute confidence invervel of absolute difference.'''
        ci_width = stats.norm.ppf(1-self.alpha/2) * self.se
        abs_diff = str(round(self.abs_diff, 5))
        ci_width = str(round(ci_width, 5))
        ci_str = abs_diff + ' +- ' + ci_width
        return ci_str

#     def confidence_interval2(self):
#         '''Compute confidence invervel of relative difference.'''
#         ci_width = stats.norm.ppf(1-self.alpha/2) * self.se / abs(self.p0_ori)
#         left = self.rela_diff - ci_width
#         left = str(round(left*100, 3)) + '%'
#         right = self.rela_diff + ci_width
#         right = str(round(right*100, 3)) + '%'
#         ci_str = '[' + left + ', ' + right + ']'
#         return ci_str

    def effect_size(self):
        '''Compute the effect size.'''
        pooled_std = np.sqrt((self.std0**2 + self.std1**2)/2)
        return self.abs_diff / pooled_std

    def add_control(
        self,
        y0_numerator,
        y0_denominator,
        y1_numerator,
        y1_denominator
    ):
        '''
        Add one control variable to test metric

        Parameters
        ----------
        y0_numerator : list of float
            numerator of control group of control variable in each bucket
        y0_denominator : list of float
            denominator of control group of control variable in each bucket
        y1_numerator : list of float
            numerator of treatment group of control variable in each bucket
        y1_denominator : list of float
            denominator of treatment group of control variable in each bucket
        '''
        x_metric = Metric(np.array(self.x0.sums) + np.array(self.x1.sums),
                          np.array(self.x0.cnts) + np.array(self.x1.cnts))
        y_metric = Metric(np.array(y0_numerator) + np.array(y1_numerator),
                          np.array(y0_denominator) + np.array(y1_denominator))
        self.beta = (
            metric_cov(x_metric, y_metric) / metric_cov(y_metric, y_metric)
        )
        # update metric with control variable
        self.y0 = Metric(y0_numerator, y0_denominator)
        self.y1 = Metric(y1_numerator, y1_denominator)
        self.p0 = self.x0.mean - self.beta*self.y0.mean
        self.p1 = self.x1.mean - self.beta*self.y1.mean
        self.var0 = (
            self.x0.delta_var() + self.beta**2*self.y0.delta_var() -
            2*self.beta*metric_cov(self.x0, self.y0)
        )
        self.var1 = (
            self.x1.delta_var() + self.beta**2*self.y1.delta_var() -
            2*self.beta*metric_cov(self.x1, self.y1)
        )
        self.std0 = np.sqrt(self.var0 * self.x0.cnt)
        self.std1 = np.sqrt(self.var1 * self.x1.cnt)
        self.var = self.var0 + self.var1
        self.se = np.sqrt(self.var)
        self.abs_diff = self.p1 - self.p0
        if self.p0_ori != 0:
            self.rela_diff = self.abs_diff / abs(self.p0_ori)
        else:
            self.rela_diff = float("inf")


def cov(list1, list2):
    '''Compute the covariance of two lists.

    Parameters
    ----------
    list1 : list
        the first list
    list2 : list
        the second list

    Returns
    ----------
    covariance : float
        the covariance of two lists
    '''
    assert(len(list1) == len(list2))
    size = len(list1)
    covar = (
        1.0 * size / (size-1) *
        (sum(map(lambda x: x[0]*x[1], zip(list1, list2)))/size -
            sum(list1)/size*sum(list2)/size)
    )

    return covar


def metric_cov(metric1, metric2):
    '''Compute covariance of two metric.

    Parameters
    ----------
    metric1 : Metric
        the first metric
    metric2 : Metric
        the second metric

    Returns
    ----------
    covariance : float
        the covariance of two metrics
    '''
    num_bucket = len(metric1.sums)
    mu_d1, mu_d2 = sum(metric1.cnts)/num_bucket, sum(metric2.cnts)/num_bucket
    mu_n1, mu_n2 = sum(metric1.sums)/num_bucket, sum(metric2.sums)/num_bucket

    cov_n1_n2 = cov(metric1.sums, metric2.sums)
    cov_d1_d2 = cov(metric1.cnts, metric2.cnts)
    cov_n1_d2 = cov(metric1.sums, metric2.cnts)
    cov_n2_d1 = cov(metric1.cnts, metric2.sums)

    results = (
        1.0 / num_bucket *
        (
            cov_n1_n2 / mu_d1 / mu_d2 +
            cov_d1_d2 * mu_n1 * mu_n2 / pow(mu_d1*mu_d2, 2.0) -
            cov_n1_d2 * mu_n2 / mu_d1 / pow(mu_d2, 2.0) -
            cov_n2_d1 * mu_n1 / mu_d2 / pow(mu_d1, 2.0)
        )
    )

    return results

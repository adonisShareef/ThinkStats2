import pandas as pd
import numpy as np
import sys
import nsfg
import thinkstats2
from thinkstats2 import Mean, MeanVar, Var, Std, Cov
import thinkplot
import matplotlib.pyplot as plt

def histograms(live,num_bins,filename):
    hist = live.plot.hist(bins=num_bins)
    hist.figure.savefig(filename)

def findMean(live):
    print('Customer Age mean: ',live['Customer_Age'].mean())
    print('Avg_Utilization_Ratio mean: ',live['Avg_Utilization_Ratio'].mean())
    print('Total_Revolving_Bal mean: ',live['Total_Revolving_Bal'].mean())
    print('Total_Trans_Ct mean: ',live['Total_Trans_Ct'].mean())
    print('Credit_Limit mean: ',live['Credit_Limit'].mean())
    print('Contacts_Count_12_mon mean: ',live['Contacts_Count_12_mon'].mean())

def findMode(live):
    print('Customer Age mode: ',live['Customer_Age'].mode())
    print('Avg_Utilization_Ratio mode: ',live['Avg_Utilization_Ratio'].mode())
    print('Total_Revolving_Bal mode: ',live['Total_Revolving_Bal'].mode())
    print('Total_Trans_Ct mode: ',live['Total_Trans_Ct'].mode())
    print('Credit_Limit mode: ',live['Credit_Limit'].mode())
    print('Contacts_Count_12_mon mode: ',live['Contacts_Count_12_mon'].mode())
def pmf(live):
    util_ratio = live.Avg_Utilization_Ratio
    pmf_util_ratio = thinkstats2.Pmf(util_ratio)
    print('pmf Avg_Utilization_Ratio mean: ', pmf_util_ratio.Mean())
    print('pmf Avg_Utilization_Ratio variance: ', pmf_util_ratio.Var())

    contacts = live.Contacts_Count_12_mon
    pmf_contacts = thinkstats2.Pmf(contacts)
    print('pmf Contacts_Count_12_mon mean: ', pmf_contacts.Mean())
    print('pmf Contacts_Count_12_mon variance: ', pmf_contacts.Var())

def Scatterplots(live):
    ages = live.Customer_Age
    transactions = live.Total_Trans_Ct
    #correlations
    print('thinkstats2 Corr ages', thinkstats2.Corr(ages, transactions))
    print('thinkstats2 SpearmanCorr ages',
          thinkstats2.SpearmanCorr(ages, transactions))

    #use ScatterPlot and save to jpg format
    thinkplot.Scatter(ages, transactions, alpha=0.5)
    thinkplot.Config(xlabel='age (years)',
                     ylabel='transactions',
                     xlim=[10, 45],
                     ylim=[0, 15],
                     legend=False)
    thinkplot.Save(root='Customer_Age_Transactions',
                   legend=False,
                   formats=['jpg'])

    bal = live.Total_Revolving_Bal
    ages = live.Customer_Age
    #correlations
    print('thinkstats2 Corr balances', thinkstats2.Corr(bal, ages))
    print('thinkstats2 SpearmanCorr balances',
          thinkstats2.SpearmanCorr(bal, ages))

    #use ScatterPlot and save to jpg format
    thinkplot.Scatter(ages, bal, alpha=0.5)
    thinkplot.Config(ylabel='Total_Revolving_Bal ($USD)',
                     xlabel='Customer_Age (years)',
                     ylim=[0, 5000],
                     xlim=[0, 100],
                     legend=False)
    thinkplot.Save(root='Total_balance_Ages',
                   legend=False,
                   formats=['jpg'])

def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res
def LeastSquares(xs, ys):
    #interpret the intercept and slope
    meanx, varx = MeanVar(xs)
    meany = Mean(ys)

    slope = Cov(xs, ys, meanx, meany) / varx
    inter = meany - slope * meanx

    return inter, slope

def CoefDetermination(ys, res):
    #compares the variance of the residuals to the variance of the dependent variable
    return 1 - Var(res) / Var(ys)

def main():
    live = pd.read_csv("BankChurners.csv")
    live.fillna(live.mean())

    #histograms
    histograms( live['Customer_Age'],5,'customer_age_hist.jpg')
    histograms( live['Avg_Utilization_Ratio'],5,'Avg_Utilization_Ratio.jpg')
    histograms( live['Total_Revolving_Bal'],5,'Total_Revolving_Bal.jpg')
    histograms( live['Total_Trans_Ct'],5,'Total_Trans_Ct.jpg')
    histograms( live['Credit_Limit'],5,'Credit_Limit.jpg')
    histograms( live['Contacts_Count_12_mon'],5,'Contacts_Count_12_mon.jpg')

    findMean(live)
    print('-'*36)
    findMode(live)
    print('-'*36)
    #pmf
    pmf(live)
    print('-'*36)

    #cdf
    cdf = thinkstats2.Cdf(live['Credit_Limit'])
     #range check
    for k,v in cdf.Items():
        assert(v >=0 and v <=1)
    thinkplot.Cdf(cdf)
    thinkplot.Show(xlabel='Credit_Limit', ylabel='CDF')

    #scatter plots
    Scatterplots(live)

    avg_util, age = live.Avg_Utilization_Ratio, live.Customer_Age


    inter, slope = LeastSquares(avg_util, age)
    #easier to interpret
    print('intercept:', inter + slope * 25)
    print('slope', slope * 10 )

    res = Residuals(avg_util, age, inter, slope)
    r2 = CoefDetermination(age, res)
    print('rho', thinkstats2.Corr(avg_util, age))
    print('R', np.sqrt(r2))
    print('Std(ys)', Std(age))
    print('Std(res)', Std(res))

if __name__ == '__main__':
    main()

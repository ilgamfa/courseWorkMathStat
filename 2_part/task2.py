# Гипотеза H0: выборка имеет показательное распредление с параметром lambda = 1.5
import matplotlib.pyplot as plt
import csv
import numpy as np
import collections
import math
from scipy.special import kolmogorov
import pandas as pd

alf = 0.01
l = 2  # Лямбда для показательного распределения
eps = 1e-8


with open('r2z2.csv') as file:
    reader = csv.reader(file)
    sample = []
    for row in reader:
        sample.extend(row)
sample.remove('X')
sample = [float(val) for val in sample]


def new_efr(sample):
    hist, limits = np.histogram(sample, bins=len(sample))
    amount = [hist[0]]
    for i in range(1, len(hist)):
        tmp = np.sum(hist[:i + 1])
        amount.append(tmp)
    amount = [cut / len(sample) for cut in amount]

    def result(x):
        if x <= limits[0]:
            return 0

        for i in range(len(amount)):
            if limits[i] < x <= limits[i + 1]:
                return amount[i]
        if x > limits[-1]:
            return 1

    return result


# эмпирическая функция распределения
def efr(sample):
    sample = np.sort(sample)

    def result(x):
        return sample[sample < x].size / sample.size

    return result


# функция показательного распределения
def exp_fr(l, x):
    return (1 - math.exp(-l * x))


# квантиль
def kolm_quantil(quant):
    return math.sqrt((-1 / 2) * (math.log(quant / 2)))


# построение эфр выборки с наложением функции показательного распределения
def graph(sample):
    distribution = sorted(collections.Counter(sample).most_common(), key=lambda elem: elem[0])
    x = []
    y = []
    sum_for_y = 0
    for pair in distribution:
        if pair != distribution[0]:
            x.append(prev[0])
            y.append(sum_for_y / len(sample))
        x.append(pair[0])
        y.append(sum_for_y / len(sample))
        prev = pair
        sum_for_y += pair[1]
    plt.plot(x, y)
    plt.suptitle('Эмпирическая функция распределения')
    plt.xlabel('X')
    plt.ylabel('F(X)')

    plt.plot(sorted(sample), [exp_fr(l, x) for x in sorted(sample)], color='green', lw=4)
    plt.xlabel('X')
    plt.ylabel('F(X)')
    plt.show()


cdf = efr(sample)
new_cdf = new_efr(sample)

graph(sample)

exp_df = [exp_fr(l, val) for val in sorted(sample)]
exp_df = np.array(exp_df)
df = [cdf(val) for val in sorted(sample)]
df = np.array(df)
exp_df_eps = [exp_fr(l, val + eps) for val in sorted(sample)]
exp_df_eps = np.array(exp_df_eps)
df_eps = [cdf(val + eps) for val in sorted(sample)]
df_eps = np.array(df_eps)

D_n = max(abs(exp_df - df))
D_n_1 = max(abs(exp_df_eps - df_eps))
D_n = max(D_n, D_n_1)
statistic = D_n * math.sqrt(len(sample))
k_quantil = kolm_quantil(alf)

p_value = kolmogorov(statistic)

print(f"D_N = {D_n}")
print(f"Критическая область имеет вид: Statistic > {k_quantil}")
print(f"Критическая константа = {k_quantil}")
print(f"Статистика = {statistic}")

if statistic > k_quantil:
    print("Гипотеза H0 отклоняется")
else:
    print("Гипотеза H0 принимается")

print(f"P-value = {p_value}")

from scipy import stats
import csv


with open('r3z2.csv') as file:
    reader = csv.reader(file)
    df = []
    for row in reader:
        df.extend(row)
df.remove('X')
df = [float(val) for val in df]


n = len(df)
mean = sum(df) / n
q = 0.9
alfa = 1 - q

print(f"Объём выборки: {n}")
print(f"Выборочное среднее: {mean}")

deviations = [(x - mean) ** 2 for x in df]
variance_sm = sum(deviations) / n
print("Смещеная дисперсия =", variance_sm)

er_average = variance_sm**0.5 / ((n-1)**0.5)
print(f"Стандартная ошибка среднего: {er_average}")

q_quantil = stats.t(n-1).ppf(1-alfa)
up_limit = er_average * q_quantil + mean

print(f"Верхняя доверительная граница: {up_limit}")
print(f"Доверительный интервал имеет вид: -infinity; {up_limit}")
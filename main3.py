# Определим функцию f(x) для уравнения
def f(x):
    return x ** 3 - 3.125 * x ** 2 - 3.5 * x + 2.458


# Преобразуем уравнение в итерационную форму
def g(x):
    return x+(x ** 3 - 3.125 * x ** 2 - 3.5 * x + 2.458)/(-19.5)


# Метод простой итерации
def iteration_method(x0, tol=1e-2, max_iter=100):
    iterations = []
    x_k = x0
    for k in range(max_iter):
        x_k1 = g(x_k)  # Находим новое значение по итерационной формуле
        f_xk1 = f(x_k1)
        error = abs(x_k1 - x_k)

        iterations.append((k + 1, x_k, x_k1, f_xk1, error))

        if error < tol:
            break

        x_k = x_k1

    return iterations


# Запускаем метод итерации на интервале (3, 4)
iterations = iteration_method(x0=3.0, tol=1e-2)


# Выводим таблицу результатов
for s in iterations:
    print(s)

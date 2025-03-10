# Определяем функцию f(x)
def f(x):
    return x ** 3 - 3.125 * x ** 2 - 3.5 * x + 2.458


# Метод половинного деления (бисекции)
def bisection_method(a, b, tol=1e-2, max_iter=100):
    iterations = []
    for k in range(max_iter):
        x = (a + b) / 2  # Средняя точка
        f_a, f_b, f_x = f(a), f(b), f(x)
        error = abs(a - b)

        iterations.append((k + 1, a, b, x, f_a, f_b, f_x, error))

        if error < tol:
            break

        if f_a * f_x < 0:
            b = x  # Корень в левом подотрезке
        else:
            a = x  # Корень в правом подотрезке

    return iterations


# Запускаем метод бисекции на интервале (0,1)
iterations = bisection_method(a=0, b=1, tol=1e-2)

# Выводим таблицу результатов
for s in iterations:
    print(s)

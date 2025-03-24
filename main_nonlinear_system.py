from logging import error

import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления численной производной по конкретной переменной
def df(x, f, j, h=1e-6):
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[j] += h
    x_minus[j] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)

# Функция для вычисления второй производной
def d2f(x, f, j, h=1e-6):
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[j] += h
    x_minus[j] -= h
    f_plus = f(x_plus)
    f_minus = f(x_minus)
    return (f_plus - 2 * f(x) + f_minus) / (h ** 2)

# Метод Ньютона для системы уравнений
def jacobian(x, funcs, n):
    J = np.zeros((n, n))  # Создаем матрицу Якобиана размером n x n
    for i in range(n):
        for j in range(n):
            J[i, j] = df(x, funcs[i], j)  # Вычисляем частную производную функции i по переменной j
    return J


# Метод для нахождения решения системы методом Ньютона
def newton_method_system(funcs, jacobian, x0, n, epsilon, max_iterations=100, verbose=False):
    x = np.array(x0)  # Преобразуем начальное приближение в массив numpy
    delta_x = []
    max_delta_x = 0

    for i in range(max_iterations):
        # Вычисление значений функций в текущей точке
        F = np.array([func(x) for func in funcs])
        J = jacobian(x, funcs, n)  # Якобиан

        # Проверка детерминанта Якобиана
        det_J = np.linalg.det(J)
        if np.abs(det_J) < 1e-6:
            print(f"Детерминант Якобиана слишком мал: {det_J}")
            continue_solution = input("Детерминант Якобиана слишком мал. Хотите продолжить решение (y/n)? ")
            if continue_solution.lower() != 'y':
                print("Прерывание решения.")
                return None, i + 1, delta_x, max_delta_x


        try:
            delta_x = np.linalg.solve(J, -F)  # Решение системы J*delta_x = -F
        except np.linalg.LinAlgError:
            print(f"Ошибка: Якобиан вырожден на итерации {i + 1}")
            return None, i + 1, delta_x, max_delta_x

        x_new = x + delta_x  # Новый вектор
        max_delta_x = np.max(np.abs(delta_x))  # Отклонение

        if verbose:
            # Выводим шапку таблицы
            if i == 0:
                print(f"{'№ Итерации':<12}{'xk':<12}{'yk':<12}{'xk+1':<12}{'yk+1':<12}{'xk+1 - xk':<12}")
                print("-" * 72)

            # Выводим шаги в требуемом формате
            print(f"{i + 1:<12}{x[0]:<12.6f}{x[1]:<12.6f}{x_new[0]:<12.6f}{x_new[1]:<12.6f}{np.linalg.norm(delta_x):<12.6f}")

        if max_delta_x < epsilon:  # Если погрешность меньше требуемой, возвращаем результат
            return x_new, i + 1, delta_x, max_delta_x

        x = x_new  # Обновляем значение переменной

    return x, max_iterations, delta_x, max_delta_x


# Функции для системы уравнений
def f1(x):
    return np.sin(x[0]) + 0.5 - x[1] - 1  # sin(x) + 0.5 - y = 1

def f2(x):
    return np.cos(x[1]) - 2 + x[0]  # cos(y) - 2 + x = 0


# Графики функций
def plot_graph(funcs, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z1 = np.sin(X) + 0.5 - Y - 1  # f1(x, y)
    Z2 = np.cos(Y) - 2 + X  # f2(x, y)

    plt.figure(figsize=(8, 6))

    # График f1
    contour1 = plt.contour(X, Y, Z1, levels=[0], colors='blue')
    # График f2
    contour2 = plt.contour(X, Y, Z2, levels=[0], colors='red')

    # Добавление подписей на контурах
    plt.clabel(contour1, inline=True, fontsize=8, fmt={0: r'$f_1(x, y) = 0$'})
    plt.clabel(contour2, inline=True, fontsize=8, fmt={0: r'$f_2(x, y) = 0$'})

    # Построение графика с легендой
    plt.title('Графики функций системы уравнений')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)

    plt.show()


# Основная программа
def main():
    # Захардкоженные функции для вашей системы уравнений
    funcs = [f1, f2]
    n = len(funcs)  # Размерность системы

    # График функций до нахождения решения
    print("Построение графиков функций системы...")
    plot_graph(funcs, x_range=(-2, 2), y_range=(-2, 2))  # Установим диапазон для x и y

    # Ввод точности
    while True:
        try:
            epsilon = float(input("Введите точность (epsilon): "))
            if epsilon <= 0:
                print("Точность должна быть положительным числом. Попробуйте снова.")
            else:
                break
        except ValueError:
            print("Ошибка! Пожалуйста, введите число.")

    # Ввод максимального числа итераций
    while True:
        try:
            max_iterations = int(input("Введите максимальное количество итераций: "))
            if max_iterations <= 0:
                print("Максимальное количество итераций должно быть положительным числом. Попробуйте снова.")
            else:
                break
        except ValueError:
            print("Ошибка! Пожалуйста, введите число.")

    # Ввод начальных приближений с клавиатуры
    x0 = []
    for i in range(n):
        while True:
            try:
                x_val = float(input(f"Введите начальное приближение для x{i + 1}: "))
                x0.append(x_val)
                break
            except ValueError:
                print("Ошибка! Пожалуйста, введите число.")

    # Опционально выводить шаги
    verbose = input("Хотите выводить шаги итераций (y/n)? ").lower() == 'y'

    print("Решение системы уравнений методом Ньютона...")
    solution, iterations, delta_x, error  = newton_method_system(funcs, jacobian, x0, n, epsilon, max_iterations, verbose)

    if solution is not None:
        print(f"\nРешение системы: {solution}")
        print(f"Количество итераций: {iterations}")
        print(f"Вектор погрешностей: {delta_x}")
        print(f"Максимальное отклонение по координате: {error}")

    else:
        print("Не удалось найти решение")


if __name__ == "__main__":
    main()

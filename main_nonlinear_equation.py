import numpy as np
import matplotlib.pyplot as plt
import os


# Метод для вычисления численной производной (приближённо)
def numerical_derivative(f, x, h=1e-5):
    try:
        return (f(x + h) - f(x - h)) / (2 * h)
    except OverflowError:
        print(f"Ошибка переполнения при вычислении производной в точке x = {x}")
        return None

# Метод для вычисления второй численной производной
def second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
# Метод хорд с выбором фиксированного конца по знаку второй производной
def chord_method_fixed(f, a, b, epsilon, max_iterations=1000):
    if f(a) * f(b) > 0:
        print("Нет корней на данном интервале!")
        return None, 0

    # Вычисляем вторые производные на концах
    fpp_a = second_derivative(f, a)
    fpp_b = second_derivative(f, b)

    # Выбираем фиксированный конец
    if np.sign(f(a)) == np.sign(fpp_a):
        fixed = 'a'
    else:
        fixed = 'b'

    x_next = a if fixed == 'a' else b
    for i in range(max_iterations):
        if fixed == 'a':
            x_next = b - (f(b) * (b - a)) / (f(b) - f(a))
        else:
            x_next = a - (f(a) * (a - b)) / (f(a) - f(b))

        if abs(f(x_next)) < epsilon:
            return x_next, i + 1

        if fixed == 'a':
            b = x_next
        else:
            a = x_next

    return x_next, max_iterations

# Метод секущих
def secant_method(f, x0, x1, epsilon, max_iterations=1000):
    try:
        for i in range(max_iterations):
            fx0 = f(x0)
            fx1 = f(x1)
            if np.abs(fx1 - fx0) < 1e-10:  # Чтобы избежать деления на ноль
                print("Ошибка: разность значений функции слишком мала")
                break
            x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if np.abs(x_next - x1) < epsilon:
                return x_next, i + 1
            x0, x1 = x1, x_next

        return x1, max_iterations
    except OverflowError:
        print("Ошибка переполнения при вычислениях метода секущих.")
        return None, max_iterations
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None, max_iterations


# Метод простой итерации с использованием phi(x) = x + lambda * f(x)
def simple_iteration_method(f, g, x0, epsilon, max_iterations=1000):
    try:
        x = x0
        for i in range(max_iterations):
            x_next = g(x)
            if np.abs(x_next - x) < epsilon:
                return x_next, i + 1
            x = x_next

        return x, max_iterations
    except OverflowError:
        print("Ошибка переполнения при вычислениях метода простой итерации.")
        return None, max_iterations
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None, max_iterations


# Проверка на наличие корней в интервале
def check_roots(f, a, b):
    if f(a) * f(b) > 0:
        return False
    return True


# Ввод уравнения и параметров
def input_nonlinear_equation():
    try:
        choice = input("Хотите ввести уравнение из файла (f) или с клавиатуры (k)? (Введите f или k): ").strip().lower()
        if choice == 'f':
            filename = input("Введите имя файла: ")
            if not os.path.exists(filename):
                print(f"Файл {filename} не существует.")
                return None, None, None, None, None
            with open(filename, 'r') as file:
                f_str = file.read().strip()
                print(f"Уравнение из файла: {f_str}")
        elif choice == 'k':
            f_str = input("Введите нелинейное уравнение f(x) в виде строки (например, 'x**3 - 4*x**2 + x - 1'): ")
        else:
            print("Неверный выбор. Выход из программы.")
            return None, None, None, None, None

        f = lambda x: eval(f_str)

        # Численное вычисление производной
        df = lambda x: numerical_derivative(f, x)

        # Рисуем график функции
        plot_function(f)

        a = float(input("Введите начало интервала a: "))
        b = float(input("Введите конец интервала b: "))
        epsilon = float(input("Введите точность вычислений: "))

        if not check_roots(f, a, b):
            print(f"На интервале [{a}, {b}] нет корней.")
            return None, None, None, None, None

        return f, df, a, b, epsilon
    except Exception as e:
        print(f"Ошибка ввода: {e}")
        return None, None, None, None, None


def input_method_choice():
    try:
        print("Выберите метод решения:")
        print("1. Метод хорд")
        print("2. Метод секущих")
        print("3. Метод простой итерации")
        choice = input("Введите номер метода (1, 2, 3): ")
        if choice not in ['1', '2', '3']:
            raise ValueError("Неверный выбор метода.")
        return choice
    except Exception as e:
        print(f"Ошибка ввода метода: {e}")
        return None


# Метод для вычисления phi(x) для итераций
def phi(x, f, lambda_val):
    return x + lambda_val * f(x)


# Метод для вычисления численной производной (приближённо)
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


# Вычисление лямбды на основе производной функции φ'(x)
def calculate_lambda(f, a, b):
    # Сначала находим производную функции f(x)
    def f_prime(x):
        return numerical_derivative(f, x)

    # Дискретизация интервала [a, b] для вычисления максимальной производной
    step = (b - a) / 100  # Шаг для оценки на интервале
    x_vals = np.arange(a, b, step)
    f_prime_vals = [f_prime(x) for x in x_vals]

    # Находим максимальное значение производной по модулю
    max_f_prime = max(abs(val) for val in f_prime_vals)

    # Вычисляем lambda
    lambda_val = -2 / max_f_prime if max_f_prime != 0 else 0

    return lambda_val


# Функция для рисования графика функции
def plot_function(f, x_range=(-10, 10), step=0.1):
    try:
        x_vals = np.arange(x_range[0], x_range[1], step)
        y_vals = [f(x) for x in x_vals]
        plt.plot(x_vals, y_vals, label="f(x)")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("График функции f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Ошибка при рисовании графика: {e}")


# Основная программа
def main():
    try:
        f, df, a, b, epsilon = input_nonlinear_equation()
        if f is None:
            return

        choice = input_method_choice()
        if choice is None:
            return
        if choice == '3':
            lambda_val = calculate_lambda(f, a, b)
            print(f"Автоматически вычисленное значение lambda: {lambda_val}")

        if choice == '1':
            print("\nИспользуется метод хорд.")
            solution, iterations = chord_method_fixed(f, a, b, epsilon)
        elif choice == '2':
            print("\nИспользуется метод секущих.")
            x1 = float(input("Введите второе приближение x1: "))
            solution, iterations = secant_method(f, a, x1, epsilon)
        elif choice == '3':
            print("\nИспользуется метод простой итерации.")
            g = lambda x: phi(x, f, lambda_val)
            solution, iterations = simple_iteration_method(f, g, a, epsilon)
        else:
            print("Неверный выбор метода.")
            return

        if solution is not None:
            print(f"\nРешение уравнения: x = {solution:.10f}")
            print(f"f(x) = {f(solution):.10f}")  # Выводим значение функции в решении
            print(f"Количество итераций: {iterations}")
            output_choice = input("Вы хотите вывести результат в файл? (y/n): ").strip().lower()
            if output_choice == 'y':
                filename = input("Введите имя файла для сохранения результатов: ")
                with open(filename, 'w') as file:
                    file.write(f"Решение уравнения: x = {solution:.10f}\n")
                    file.write(f"f(x) = {f(solution):.10f}\n")
                    file.write(f"Количество итераций: {iterations}\n")
        else:
            print("Не удалось найти решение.")

    except Exception as e:
        print(f"Произошла ошибка в программе: {e}")


if __name__ == "__main__":
    main()

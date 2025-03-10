import math

# Инициализация значений
x0 = 0
y0 = 0

# Устанавливаем точность
tolerance = 0.01

# Массив для хранения результатов
iterations = []

# Выполнение итераций
iteration_count = 0
while True:
    # Вычисляем новые значения y и x через φ(x) и φ(y)
    y1 = math.sin(x0 + 0.5) - 1  # φ(x) для y
    x1 = -math.cos(y0 - 2)      # φ(y) для x

    # Сохраняем результаты
    iterations.append((iteration_count, x0, y0, x1, y1, x1-x0))

    # Проверяем, достигнута ли требуемая точность
    if abs(x1 - x0) < tolerance and abs(y1 - y0) < tolerance:
        break

    # Обновляем значения для следующей итерации
    x0, y0 = x1, y1
    iteration_count += 1

# Вывод результатов итераций
for s in iterations:
    print(s)
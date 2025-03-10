import sympy as sp

x = sp.symbols('x')
phi = x + (x**3 - 3.125*x**2 - 3.5*x + 2.458)/(-19.5)

phi_prime = sp.diff(phi, x)

abs_phi_prime = sp.Abs(phi_prime)

value_at_3 = abs_phi_prime.subs(x, 3)
value_at_4 = abs_phi_prime.subs(x, 4)

critical_points = sp.solveset(phi_prime, x, domain=sp.Interval(3, 4))

critical_points = list(critical_points) + [3, 4]

values_at_critical_points = [abs_phi_prime.subs(x, point) for point in critical_points]

max_value = max(values_at_critical_points)

print("Максимальное значение |phi'(x)| на интервале [3, 4]:", max_value)

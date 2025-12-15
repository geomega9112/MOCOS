import numpy as np
import matplotlib.pyplot as plt

# Вихідні дані
n = 5  # ваш номер варіанту
N = 100 * n  # межа інтегрування
T_values = [4, 8, 16, 32, 64, 128]
k_max = 20  # кількість членів ряду


# Функція сигналу f(t)
def f(t):
    return t ** (2 * n)  # t^(2n) = t^10


# Підпрограма для обчислення k-го члена інтегралу Фур’є
def fourier_coefficient(k, T, t):
    w_k = 2 * np.pi * k / T  # кутова частота ωk

    cos_part = np.cos(w_k * t)
    sin_part = np.sin(w_k * t)
    dt = t[1] - t[0]

    Re = np.sum(f(t) * cos_part) * dt  # дійсна частина
    Im = -np.sum(f(t) * sin_part) * dt  # уявна частина (з мінусом)

    return Re, Im


# Підпрограма для амплітудного спектра
def amplitude(Re, Im):
    return np.sqrt(Re ** 2 + Im ** 2)


# Формуємо сітку t для інтегрування
t = np.linspace(-N, N, 20001)

# Основний розрахунок і побудова графіків
for T in T_values:
    Re_list = []
    Im_list = []
    Amp_list = []

    print(f"\nT = {T}")
    print("k\tRe(F)\t\tIm(F)\t\t|F|")

    for k in range(k_max + 1):
        Re, Im = fourier_coefficient(k, T, t)
        A = amplitude(Re, Im)

        Re_list.append(Re)
        Im_list.append(Im)
        Amp_list.append(A)

        print(f"{k}\t{Re:.5e}\t{Im:.5e}\t{A:.5e}")

    # Графік Re(F(w_k))
    plt.figure()
    plt.plot(range(k_max + 1), Re_list, 'b-o')
    plt.title(f"Re(F(ωₖ)) для T = {T}")
    plt.xlabel("k")
    plt.ylabel("Re(F)")
    plt.grid(True)
    plt.show()

    # Графік амплітуди |F|
    plt.figure()
    plt.plot(range(k_max + 1), Amp_list, 'r-o')
    plt.title(f"|F(ωₖ)| для T = {T}")
    plt.xlabel("k")
    plt.ylabel("|F|")
    plt.grid(True)
    plt.show()

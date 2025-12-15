import numpy as np
import matplotlib.pyplot as plt

PI = np.pi

def f(x):
    return 5 * np.sin(5 * PI * x)

def simpson(func, a, b, eps):
    N = 20000
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    fx = func(x)
    I_prev = (h/3) * (fx[0] + 4 * np.sum(fx[1:N:2]) + 2 * np.sum(fx[2:N-1:2]) + fx[N])

    while True:
        N *= 2
        h = (b - a) / N
        x = np.linspace(a, b, N+1)
        fx = func(x)
        I_curr = (h/3) * (fx[0] + 4 * np.sum(fx[1:N:2]) + 2 * np.sum(fx[2:N-1:2]) + fx[N])
        if abs(I_curr - I_prev) < eps:
            return I_curr
        I_prev = I_curr

def compute_a0(eps):
    return (1/PI) * simpson(f, 0, PI, eps)

def compute_ak(k, eps):
    return (1/PI) * simpson(lambda x: f(x)*np.cos(k*x), 0, PI, eps)

def compute_bk(k, eps):
    return (1/PI) * simpson(lambda x: f(x)*np.sin(k*x), 0, PI, eps)

def fourier_series(x, N, a0, ak, bk):
    s = a0/2
    for k in range(1, N+1):
        s += ak[k]*np.cos(k*x) + bk[k]*np.sin(k*x)
    return s

def main():
    eps = 1e-8
    N = 40
    a0 = compute_a0(eps)
    ak = [0]*(N+1)
    bk = [0]*(N+1)

    for k in range(1, N+1):
        ak[k] = compute_ak(k, eps)
        bk[k] = compute_bk(k, eps)

    # Збереження результатів
    with open("results.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f"Порядок N = {N}\n")
        f_out.write(f"a0 = {a0:.6f}\n")
        for k in range(1, N+1):
            f_out.write(f"k={k}: a_k={ak[k]:.6f}, b_k={bk[k]:.6f}\n")

    x = np.linspace(0, PI, 500)

    # --- Графіки гармонік + початкова функція ---
    plt.figure(figsize=(10,6))
    # Гармоніки
    #for k in range(1, N+1):
    # Остання часткова сума (до N гармонік)
    fourier_sum = [fourier_series(xi, N, a0, ak, bk) for xi in x]
    plt.plot(x, fourier_sum, 'r', linewidth=2, label=f"F_{N}(x)")
    # Початкова функція
    plt.plot(x, f(x), 'k', linewidth=2.5, label="f(x)")

    # Координатні осі
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.ylim(-5, 5)
    plt.xlim(0, 2/5)
    plt.title("Гармоніки та початкова функція")
    plt.xlabel("x")
    plt.ylabel("Амплітуда")
    plt.legend(loc='upper right', fontsize=9)
    plt.grid()
    plt.show()

    # --- Графіки a(k) та b(k) ---
    k_vals = np.arange(1, N+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.stem(k_vals, [ak[k] for k in k_vals], basefmt=" ")
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Коефіцієнти a(k)")
    plt.xlabel("k")
    plt.ylabel("a(k)")
    plt.grid()

    plt.subplot(1,2,2)
    plt.stem(k_vals, [bk[k] for k in k_vals], basefmt=" ")
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Коефіцієнти b(k)")
    plt.xlabel("k")
    plt.ylabel("b(k)")
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Перевірка апроксимації
    x_val = float(input("Введіть x: "))
    f_val = f(x_val)
    approx = fourier_series(x_val, N, a0, ak, bk)
    error = abs(f_val - approx)/abs(f_val)*100 if f_val != 0 else abs(f_val - approx)
    print(f"x={x_val:.3f}, f(x)={f_val:.6f}, F_N(x)={approx:.6f}, похибка={error:.6f}%")

if __name__ == "__main__":
    main()

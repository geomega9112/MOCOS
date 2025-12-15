import numpy as np
import time

n = 5
np.random.seed(0)

def compute_dft_with_metrics(f_input):

    N = len(f_input)
    total_mult = 0
    total_add = 0
    F = np.zeros(N, dtype=complex)

    start = time.perf_counter()
    for k in range(N):
        sum_val = 0
        for m in range(N):
            angle = -2j * np.pi * k * m / N
            sum_val += f_input[m] * np.exp(angle)
            total_mult += 1
            total_add += 1
        F[k] = sum_val
    elapsed = time.perf_counter() - start
    return F, elapsed, total_mult, total_add

def compute_fft_with_metrics(f_input):

    N_original = len(f_input)
    N_fft = 2 ** int(np.ceil(np.log2(N_original)))
    f_padded = np.zeros(N_fft)
    f_padded[:N_original] = f_input

    start = time.perf_counter()
    F = np.fft.fft(f_padded)
    elapsed = time.perf_counter() - start

    log2N = np.log2(N_fft)
    mul_count = int((N_fft / 2) * log2N)
    add_count = int(N_fft * log2N)

    return F, elapsed, mul_count, add_count

def compare_dft_fft(f_input, N_runs=3):
    N = len(f_input)
    print(f"\n===== ПОРІВНЯННЯ ДПФ vs. ШПФ (N = {N}) =====\n")

    dft_time_total = 0
    for _ in range(N_runs):
        dft_result, dft_time, dft_mul, dft_add = compute_dft_with_metrics(f_input)
        dft_time_total += dft_time
    dft_time_avg = dft_time_total / N_runs

    fft_time_total = 0
    for _ in range(N_runs):
        fft_result, fft_time, fft_mul, fft_add = compute_fft_with_metrics(f_input)
        fft_time_total += fft_time
    fft_time_avg = fft_time_total / N_runs

    if fft_time_avg == 0:
        fft_time_avg = 1e-9

    print(f"Кількість відліків N (для ДПФ): {N}")
    print(f"Кількість відліків N_FFT (для ШПФ): {len(fft_result)}")
    print("-" * 60)
    print("                 |   ДПФ (ЛР2)   |   ШПФ (ЛР3)   |   Прискорення")
    print("-----------------|---------------|---------------|---------------")
    print(f"Час (середній)  | {dft_time_avg:.6f} с | {fft_time_avg:.6f} с | {dft_time_avg / fft_time_avg:.1f}x")
    print(f"Множення         | {dft_mul:<13d}| {fft_mul:<13d}| {dft_mul / fft_mul:.1f}x")
    print(f"Додавання        | {dft_add:<13d}| {fft_add:<13d}| {dft_add / fft_add:.1f}x")
    print("-" * 60)
    print("Висновок: ШПФ значно швидше та потребує менше операцій.")

    return dft_result, fft_result

def main_lr3():
    N_part1 = 10 + n
    f_part1 = np.sin(2 * np.pi * np.arange(N_part1) / N_part1) + 0.5 * np.random.randn(N_part1)

    N_part2 = 96 + n
    f_part2 = np.sin(2 * np.pi * np.arange(N_part2) / N_part2) + 0.5 * np.random.randn(N_part2)

    compare_dft_fft(f_part1, N_runs=3)
    compare_dft_fft(f_part2, N_runs=3)

if __name__ == "__main__":
    main_lr3()

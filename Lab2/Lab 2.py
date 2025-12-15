import numpy as np
import matplotlib.pyplot as plt
import time

n=5
N=10+n
M=96+n

print("=== Частина І ===")

f=np.sin(2*np.pi*np.arange(N)/N)+0.5*np.random.randn(N)

def fourier_term(f,k,N):
    Ak=0
    Bk=0
    mult_ops=0
    add_ops=0
    for n in range(N):
        Ak+=f[n]*np.cos(2*np.pi*k*n/N)
        Bk-=f[n]*np.sin(2*np.pi*k*n/N)
        mult_ops+=2
        add_ops+=2
    return Ak,Bk,mult_ops,add_ops

C=[]
total_mult=0
total_add=0
t0=time.time()
for k in range(N):
    Ak,Bk,m,a=fourier_term(f,k,N)
    total_mult+=m
    total_add+=a
    C.append(complex(Ak,Bk))
t1=time.time()

print(f"Час обчислення:{t1-t0:.6f}сек")
print(f"Кількість множень:{total_mult},кількість додавань:{total_add}")

ampl=np.abs(C)
phase=np.angle(C)

plt.figure(figsize=(10,4))
plt.stem(ampl)
plt.title("Амплітудний спектр (N=15)")
plt.xlabel("k")
plt.ylabel("|Ck|")
plt.grid()
plt.show() 

plt.figure(figsize=(10,4))
plt.stem(phase)
plt.title("Фазовий спектр (N=15)")
plt.xlabel("k")
plt.ylabel("arg(Ck)")
plt.grid()
plt.show()

print("\n=== Частина ІІ ===")

binM=list(bin(M)[2:].zfill(8))
if n%2==1:
    binM[0]='1'
else:
    binM[0]='0'
samples=np.array([int(x)for x in binM],dtype=float)

def compact(arr):
    result='['
    for x in arr:
        if isinstance(x, complex):
            result+=f"{x.real:+.4f}{x.imag:+.4f}j "
        else:
            result+=f"{x:.4f}"
    return result+']'

print("Вхідні відліки s(nTδ):",compact(samples))

C2=np.fft.fft(samples)
ampl2=np.abs(C2)
phase2=np.angle(C2)

print("Коефіцієнти Cn:",compact(C2))
print("Модулі |Cn|:",compact(ampl2))
print("Фази arg(Cn):",compact(phase2))

t=np.linspace(0,1,1000)
reconstructed=np.zeros_like(t,dtype=complex)
for k in range(len(C2)):
    reconstructed+=(C2[k]/len(C2))*np.exp(1j*2*np.pi*k*t)

plt.figure(figsize=(10,4))
plt.plot(t,reconstructed.real)
plt.title("Відновлений аналоговий сигнал s(t)")
plt.xlabel("t")
plt.ylabel("s(t)")
plt.grid()
plt.show()

print("\n=== Частина ІІІ ===")

s_recovered=np.fft.ifft(C2)
print("Відліки s(nTδ),n=0..7:",compact(s_recovered))

s0=(1/len(C2))*sum(C2[k]for k in range(len(C2)))
s1=(1/len(C2))*sum(C2[k]*np.exp(1j*2*np.pi*k*1/len(C2))for k in range(len(C2)))
print(f"s(0Tδ)={s0:.4f}")
print(f"s(1Tδ)={s1:.4f}")

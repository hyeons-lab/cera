import subprocess,re,statistics as st,sys,time
A,B=sys.argv[1],sys.argv[2]; reps=int(sys.argv[3]); n=sys.argv[4]
M="/data/local/tmp/Llama-3.2-1B-Q4_0.gguf"
def ap():
    o=subprocess.run(["adb","shell","dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"],capture_output=True,text=True).stdout
    m=re.search(r"mValue=([0-9.]+)",o); return float(m.group(1)) if m else 99
def run(b):
    while ap()>36: time.sleep(15)
    o=subprocess.run(["adb","shell",f"cd /data/local/tmp/cmp-cera && ./{b} bench -m {M} --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1"],capture_output=True,text=True).stdout
    m=re.search(r"prefill tok/s: p50=([0-9.]+)",o); return float(m.group(1)) if m else 0
r={A:[],B:[]}
for i in range(reps):
    for b in ([A,B] if i%2==0 else [B,A]): r[b].append(run(b))
for b in (A,B): print(b,"p50",round(st.median(r[b]),1),r[b])
print("change %+.1f%%"%((st.median(r[B])/st.median(r[A])-1)*100))

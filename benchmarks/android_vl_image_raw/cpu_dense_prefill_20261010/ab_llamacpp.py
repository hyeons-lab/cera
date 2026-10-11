import subprocess,re,statistics as st,sys,time
reps=int(sys.argv[1]); n=sys.argv[2]; M="Llama-3.2-1B-Q4_0"
def ap():
    o=subprocess.run(["adb","shell","dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"],capture_output=True,text=True).stdout
    m=re.search(r"mValue=([0-9.]+)",o); return float(m.group(1)) if m else 99
def cera():
    while ap()>36: time.sleep(15)
    o=subprocess.run(["adb","shell",f"cd /data/local/tmp/cmp-cera && ./cera-q bench -m /data/local/tmp/{M}.gguf --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1"],capture_output=True,text=True).stdout
    return float(re.search(r"prefill tok/s: p50=([0-9.]+)",o).group(1))
def llama():
    while ap()>36: time.sleep(15)
    o=subprocess.run(["adb","shell",f"cd /data/local/tmp/cmp-llamacpp && LD_LIBRARY_PATH=lib ./bin/llama-bench -m /data/local/tmp/{M}.gguf -t 8 -p {n} -n 0 -r 3 -ngl 0 --device none 2>&1"],capture_output=True,text=True).stdout
    m=re.search(r"pp"+n+r"\s*\|\s*([0-9.]+)",o); return float(m.group(1)) if m else 0
c=[];l=[]
for i in range(reps):
    if i%2==0: c.append(cera()); l.append(llama())
    else: l.append(llama()); c.append(cera())
print(f"n={n} cera p50 {st.median(c):.0f} {c}\nn={n} llama.cpp p50 {st.median(l):.0f} {l}\nratio cera/llama {st.median(c)/st.median(l):.3f}")

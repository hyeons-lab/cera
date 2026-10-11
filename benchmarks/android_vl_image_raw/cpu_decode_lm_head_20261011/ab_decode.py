import subprocess,re,statistics as st,sys,time
# usage: ab_dec.py MODEL REPS PROMPT_TOKENS "[ENV=v ]binA" "[ENV=v ]binB"  -> decode tok/s p50
M,reps,n,A,B=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
def ap():
    o=subprocess.run(["adb","shell","dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"],capture_output=True,text=True).stdout
    m=re.search(r"mValue=([0-9.]+)",o); return float(m.group(1)) if m else 99
def run(v):
    while ap()>36: time.sleep(15)
    env,bn=(v.rsplit(' ',1)+[''])[:2] if ' ' in v else ('',v)
    if ' ' in v: env,bn=v.rsplit(' ',1)
    o=subprocess.run(["adb","shell",f"cd /data/local/tmp/cmp-cera && {env} ./{bn} bench -m /data/local/tmp/{M}.gguf --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {n} --max-tokens 128 2>&1"],capture_output=True,text=True).stdout
    m=re.search(r"decode tok/s: p50=([0-9.]+)",o); return float(m.group(1)) if m else 0
r={A:[],B:[]}
for i in range(reps):
    for v in ([A,B] if i%2==0 else [B,A]): r[v].append(run(v))
for v in (A,B): print(f"{M} n={n} [{v}] p50 {st.median(r[v]):.1f}",r[v])
print("change %+.1f%%"%((st.median(r[B])/st.median(r[A])-1)*100))

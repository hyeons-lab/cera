# Alternates the binary `cera-attnpf` with CERA_ATTN_PREFETCH=0 CERA_ATTN_PREFETCH_L2=0 (off) and its defaults (on); decode tok/s p50 of 3 runs.
# usage: SERIAL=<adb serial> ab_prefetch.py REPS model1,model2 depth1,depth2 (AP gate 36 C: the phone idles at 34 C when warm)
import subprocess,re,statistics as st,sys,time,os
SERIAL=os.environ.get("SERIAL","R5CXC3C2LNT")
def sh(c,t=900): 
    r=subprocess.run(["adb","-s",SERIAL,"shell",c],capture_output=True,text=True,errors="replace",timeout=t); return (r.stdout+r.stderr)
def ap():
    m=re.search(r"mValue=([\d.]+)",sh("dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'")); return float(m.group(1)) if m else None
def cool(mx=36.0):
    t0=time.time()
    while time.time()-t0<600:
        t=ap()
        if t is not None and t<=mx: return
        time.sleep(4)
def dec(model,depth,env):
    o=sh(f"cd /data/local/tmp/cmp-cera && {env} ./cera-attnpf bench -m /data/local/tmp/{model}.gguf --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {depth} --max-tokens 48 2>&1")
    m=re.search(r"decode tok/s: p50=([\d.]+)",o); return float(m.group(1)) if m else None
REPS=int(sys.argv[1]); cases=[(m,d) for m in sys.argv[2].split(",") for d in map(int,sys.argv[3].split(","))]
OFF="CERA_ATTN_PREFETCH=0 CERA_ATTN_PREFETCH_L2=0"
print(f"{'model':24s} {'depth':>5s} | {'off':>7s} {'on':>7s} {'change':>7s} | raw off / on")
for m,d in cases:
    r={"off":[],"on":[]}
    for i in range(REPS):
        order=["off","on"] if i%2==0 else ["on","off"]
        for mode in order:
            cool(); r[mode].append(dec(m,d,OFF if mode=="off" else ""))
    o,n=st.median(r["off"]),st.median(r["on"])
    print(f"{m:24s} {d:5d} | {o:7.1f} {n:7.1f} {(n/o-1)*100:+6.0f}% | {r['off']} / {r['on']}",flush=True)

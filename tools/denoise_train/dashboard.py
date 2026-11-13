"""Training statistics window for the stable denoiser.

  python dashboard.py [--runs D:/denoise_data/runs] [--port 8765] [--open]

Serves a page (http://localhost:8765) that reads the run folders train.py writes
(status, metrics, validation, sample images) and refreshes itself every few seconds:
progress and ETA, loss, quality gain per data source (training and validation),
speed and how long the GPU waits for data, learning rate, memory, gradient norm, and
the validation sample pictures (noisy | output | clean) at any step. Pause, Resume and
Stop write the run's control files. --open shows it in its own window (Edge app mode).
Read-only apart from those control files; only listens on this computer.
"""
import argparse, json, os, shutil, subprocess, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

RUNS = "D:/denoise_data/runs"


def jsonl(path, start=0):
    if not os.path.exists(path):
        return [], 0
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for l in lines[start:]:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass  # a line being written
    return out, len(lines)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def run_dir(self, name):
        d = os.path.abspath(os.path.join(RUNS, name))
        if not d.startswith(os.path.abspath(RUNS)) or not os.path.isdir(d):
            return None
        return d

    def do_GET(self):
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
        q = parse_qs(u.query)
        if not parts:
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if parts == ["api", "runs"]:
            runs = []
            for n in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
                p = os.path.join(RUNS, n, "status.json")
                if os.path.exists(p):
                    try:
                        st = json.load(open(p))
                    except ValueError:
                        st = {}
                    runs.append({"name": n, "state": st.get("state"), "updated": st.get("updated", 0)})
            runs.sort(key=lambda r: -r["updated"])
            return self.send(200, runs)
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "run":
            d = self.run_dir(parts[2])
            if not d:
                return self.send(404, {"error": "no such run"})
            what = parts[3] if len(parts) > 3 else "status"
            if what == "status":
                st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
                st["paused_file"] = os.path.exists(os.path.join(d, "pause"))
                cfg = os.path.join(d, "config.json")
                st["config"] = json.load(open(cfg)) if os.path.exists(cfg) else {}
                return self.send(200, st)
            if what in ("metrics", "val"):
                start = int(q.get("from", ["0"])[0])
                rows, n = jsonl(os.path.join(d, f"{what}.jsonl"), start)
                return self.send(200, {"rows": rows, "next": n})
            if what == "realtest":
                rows, _ = jsonl(os.path.join(d, "realtest.jsonl"))
                sd = os.path.join(d, "realtest")
                files = sorted(f for f in os.listdir(sd) if f.endswith(".png")) if os.path.isdir(sd) else []
                return self.send(200, {"rows": rows, "files": files})
            if what == "realimg" and len(parts) > 4:
                p = os.path.join(d, "realtest", os.path.basename(parts[4]))
                if os.path.exists(p):
                    return self.send(200, open(p, "rb").read(), "image/png")
            if what == "samples":
                sd = os.path.join(d, "samples")
                return self.send(200, sorted(f for f in os.listdir(sd) if f.endswith(".png")) if os.path.isdir(sd) else [])
            if what == "sample" and len(parts) > 4:
                p = os.path.join(d, "samples", os.path.basename(parts[4]))
                if os.path.exists(p):
                    return self.send(200, open(p, "rb").read(), "image/png")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        if len(parts) == 4 and parts[:2] == ["api", "run"]:
            d = self.run_dir(parts[2])
            if not d:
                return self.send(404, {"error": "no such run"})
            act = parts[3]
            if act == "pause":
                open(os.path.join(d, "pause"), "w").close()
            elif act == "resume":
                if os.path.exists(os.path.join(d, "pause")):
                    os.remove(os.path.join(d, "pause"))
            elif act == "stop":
                open(os.path.join(d, "stop"), "w").close()
            else:
                return self.send(400, {"error": "unknown action"})
            return self.send(200, {"ok": True})
        self.send(404, {"error": "not found"})


def open_window(url):
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(p):
            subprocess.Popen([p, f"--app={url}", "--window-size=1500,1000"])
            return
    webbrowser.open(url)


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    RUNS = a.runs
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://localhost:{a.port}/"
    print(f"dashboard: {url} (runs in {RUNS})", flush=True)
    if a.open:
        threading.Timer(0.5, open_window, (url,)).start()
    srv.serve_forever()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Denoiser Training</title>
<style>
:root{--bg:#0d0f15;--card:#151821;--line:#232838;--text:#e6e8ef;--dim:#8a90a2;--accent:#4f8cff;--a2:#8a5cff;--ok:#3ecf8e;--warn:#f5a524;--bad:#ff5d6c;
--c1:#4f8cff;--c2:#3ecf8e;--c3:#f5a524;--c4:#d36cff;--c5:#ff5d6c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 "Segoe UI",system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:16px;margin:0;font-weight:600}select,button{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}.badge{padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;background:var(--line)}
.running{background:#123a26;color:var(--ok)}.crashed{background:#3d1218;color:var(--bad)}.paused{background:#3d2e0e;color:var(--warn)}.stopped,.finished{background:#262b3a;color:var(--dim)}.starting{background:#132449;color:var(--accent)}
.stats{display:flex;gap:22px;flex-wrap:wrap;margin-left:auto}.stat b{display:block;font-size:18px;font-weight:600}.stat span{color:var(--dim);font-size:12px}
.bar{height:4px;background:var(--line)}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--a2));width:0}
main{padding:16px 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.card h2{font-size:13px;margin:0 0 6px;color:var(--dim);font-weight:600;display:flex;justify-content:space-between}
canvas{width:100%;height:220px;display:block}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:4px}.legend i{display:inline-block;width:10px;height:3px;margin-right:5px;vertical-align:middle;border-radius:2px}
.wide{grid-column:1/-1}#sample{display:block;width:100%;max-width:900px;margin:6px auto 0;image-rendering:pixelated;border-radius:8px;background:#000}
input[type=range]{width:100%}.empty{color:var(--dim);padding:40px;text-align:center}.cfg{font:12px ui-monospace,Consolas,monospace;color:var(--dim);white-space:pre-wrap}
@media(max-width:520px){main{grid-template-columns:1fr;padding:12px}header{padding:12px}}
</style></head><body>
<header><h1>Stable denoiser · training</h1><select id="run"></select><span id="state" class="badge">—</span>
<button id="pause">Pause</button><button id="resume">Resume</button><button id="stop">Stop</button>
<div class="stats"><div class="stat"><b id="step">—</b><span>step</span></div><div class="stat"><b id="speed">—</b><span>steps / s</span></div>
<div class="stat"><b id="eta">—</b><span>time left</span></div><div class="stat"><b id="vgain">—</b><span>validation gain</span></div><div class="stat"><b id="wait">—</b><span>waiting for data</span></div></div></header>
<div class="bar"><i id="prog"></i></div>
<main id="main"><div class="empty wide">No runs yet. Start one with <code>python train.py --run D:/denoise_data/runs/&lt;name&gt;</code>.</div></main>
<script>
const $=id=>document.getElementById(id);let run=null,M=[],V=[],mNext=0,vNext=0,samples=[];
const COL=['--c1','--c2','--c3','--c4','--c5'].map(v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim());
const fmtT=s=>{if(!isFinite(s))return'—';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h} h ${m} min`:`${m} min`};
function smooth(a,k){let o=[],e=null;for(const v of a){e=e===null?v:e*k+v*(1-k);o.push(e)}return o}
function chart(id,series,opt={}){const c=$(id);if(!c)return;const r=window.devicePixelRatio||1,W=c.clientWidth,H=c.clientHeight;c.width=W*r;c.height=H*r;
const g=c.getContext('2d');g.scale(r,r);g.clearRect(0,0,W,H);const L=52,R=10,T=8,B=22;let xs=[],ys=[];
for(const s of series)for(let i=0;i<s.x.length;i++){if(s.y[i]==null||!isFinite(s.y[i]))continue;xs.push(s.x[i]);ys.push(opt.log?Math.log10(Math.max(s.y[i],1e-9)):s.y[i])}
if(!xs.length){g.fillStyle='#8a90a2';g.fillText('waiting for data…',L,H/2);return}
let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);if(opt.zero)y0=Math.min(y0,0);if(x1==x0)x1=x0+1;if(y1==y0){y1+=1;y0-=1}const pad=(y1-y0)*.06;y0-=pad;y1+=pad;
const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);g.font='11px Segoe UI';g.strokeStyle='#232838';g.fillStyle='#8a90a2';g.lineWidth=1;
for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,yy=Y(v);g.beginPath();g.moveTo(L,yy);g.lineTo(W-R,yy);g.stroke();const lab=opt.log?Math.pow(10,v):v;g.fillText(opt.fmt?opt.fmt(lab):(Math.abs(lab)>=100?lab.toFixed(0):Math.abs(lab)>=1?lab.toFixed(2):lab.toPrecision(2)),4,yy+4)}
for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;g.fillText(v>=1000?(v/1000).toFixed(v>=1e4?0:1)+'k':v.toFixed(0),X(v)-10,H-6)}
if(opt.zero&&y0<0&&y1>0){g.strokeStyle='#3a4157';g.setLineDash([4,4]);g.beginPath();g.moveTo(L,Y(0));g.lineTo(W-R,Y(0));g.stroke();g.setLineDash([])}
series.forEach((s,si)=>{g.strokeStyle=s.color||COL[si%COL.length];g.globalAlpha=s.alpha||1;g.lineWidth=s.width||1.6;g.beginPath();let up=false;
for(let i=0;i<s.x.length;i++){let v=s.y[i];if(v==null||!isFinite(v)){up=false;continue}v=opt.log?Math.log10(Math.max(v,1e-9)):v;const px=X(s.x[i]),py=Y(v);up?g.lineTo(px,py):g.moveTo(px,py);up=true;
if(s.dots){g.stroke();g.beginPath();g.arc(px,py,2.5,0,7);g.fillStyle=g.strokeStyle;g.fill();g.beginPath();g.moveTo(px,py)}}g.stroke();g.globalAlpha=1});
const lg=$(id+'_lg');if(lg)lg.innerHTML=series.filter(s=>s.label).map((s,si)=>`<span><i style="background:${s.color||COL[series.indexOf(s)%COL.length]}"></i>${s.label}</span>`).join('')}
function layout(){$('main').innerHTML=`
<div class="card"><h2>Loss <span>Charbonnier + FFT, in PQ</span></h2><canvas id="c_loss"></canvas><div class="legend" id="c_loss_lg"></div></div>
<div class="card"><h2>Validation: quality gain over the input <span>dB, EMA weights</span></h2><canvas id="c_val"></canvas><div class="legend" id="c_val_lg"></div></div>
<div class="card"><h2>Training: quality gain by source <span>dB, smoothed</span></h2><canvas id="c_gain"></canvas><div class="legend" id="c_gain_lg"></div></div>
<div class="card"><h2>Validation PSNR <span>damaged clips: output vs input; clean clips: kept, dB</span></h2><canvas id="c_vpsnr"></canvas><div class="legend" id="c_vpsnr_lg"></div></div>
<div class="card"><h2>Validation: detail kept <span>output vs clean, fine and medium detail</span></h2><canvas id="c_detail"></canvas><div class="legend" id="c_detail_lg"></div></div>
<div class="card"><h2>Flicker (stage 2) <span>output change beyond the clean clip's, 8-bit levels</span></h2><canvas id="c_flick"></canvas><div class="legend" id="c_flick_lg"></div></div>
<div class="card"><h2>Temporal: alignment <span>clean previous frame moved onto the current one by the motion stage, PSNR (dB)</span></h2><canvas id="c_align"></canvas><div class="legend" id="c_align_lg"></div></div>
<div class="card"><h2>Temporal: neighbours used <span>output loss with another clip as neighbours (dB; about 0 = ignored)</span></h2><canvas id="c_fuse"></canvas><div class="legend" id="c_fuse_lg"></div></div>
<div class="card"><h2>Noise estimator: map size <span>estimated / true noise strength, validation (1 = exact)</span></h2><canvas id="c_est"></canvas><div class="legend" id="c_est_lg"></div></div>
<div class="card"><h2>Noise estimator: loss <span>error against the true map (under-estimates cost more)</span></h2><canvas id="c_estl"></canvas><div class="legend" id="c_estl_lg"></div></div>
<div class="card"><h2>Speed <span>steps / s</span></h2><canvas id="c_speed"></canvas><div class="legend" id="c_speed_lg"></div></div>
<div class="card"><h2>Waiting for data <span>share of time</span></h2><canvas id="c_wait"></canvas></div>
<div class="card"><h2>Learning rate</h2><canvas id="c_lr"></canvas></div>
<div class="card"><h2>GPU memory and gradient norm</h2><canvas id="c_mem"></canvas><div class="legend" id="c_mem_lg"></div></div>
<div class="card wide"><h2>Validation samples <span id="s_lab">noisy · output · clean</span></h2><input type="range" id="s_rng" min="0" max="0" value="0"><img id="sample" alt=""></div>
<div class="card wide"><h2>Real footage, no ground truth <span id="r_lab">input · output (dark clips brightened for viewing)</span></h2><div class="legend" id="r_tab"></div><input type="range" id="r_rng" min="0" max="0" value="0"><img id="realimg" alt="" style="display:block;width:100%;max-width:1100px;margin:6px auto 0;border-radius:8px;background:#000"></div>
<div class="card wide"><h2>Run settings</h2><div class="cfg" id="cfg"></div></div>`;
$('s_rng').oninput=()=>showSample(+$('s_rng').value);$('r_rng').oninput=()=>showReal(+$('r_rng').value)}
let real={rows:[],files:[]};
function showReal(i){if(!real.files.length)return;const f=real.files[i];$('realimg').src=`/api/run/${encodeURIComponent(run)}/realimg/${f}`;$('r_lab').textContent=`${f.replace('.png','')} · input · output (dark clips brightened for viewing)`;
const r=real.rows.find(x=>String(x.step).padStart(8,'0')===f.slice(0,8)&&(x.map_scale||1)==(parseFloat((f.split('_x')[1]||'1').replace('.png',''))||1))||real.rows[real.rows.length-1];
if(r){$('r_tab').innerHTML=Object.entries(r).filter(([k,v])=>typeof v==='object'&&v).map(([k,v])=>`<span>${k}: noise ${v.noise_in[0].toFixed(1)}/${v.noise_in[1].toFixed(1)} → ${v.noise_out[0].toFixed(1)}/${v.noise_out[1].toFixed(1)}, flicker ${v.flicker_in.toFixed(1)} → ${v.flicker_out.toFixed(1)}</span>`).join('')}}
function showSample(i){if(!samples.length)return;const f=samples[i];$('sample').src=`/api/run/${encodeURIComponent(run)}/sample/${f}`;$('s_lab').textContent=`step ${parseInt(f)} · noisy · output · clean`}
function draw(){const x=M.map(r=>r.step);
chart('c_loss',[{x,y:M.map(r=>r.loss),alpha:.25,width:1},{x,y:smooth(M.map(r=>r.loss),.9),label:'loss (smoothed)'}],{log:true});
const kinds=[...new Set(M.flatMap(r=>Object.keys(r.gain||{})))];
chart('c_gain',kinds.map(k=>({x,y:smooth(M.map(r=>r.gain&&r.gain[k]!=null?r.gain[k]:null),.9),label:k})).concat([{x,y:smooth(M.map(r=>r.psnr-r.psnr_in),.9),label:'all',color:'#e6e8ef',width:1}]),{zero:true});
const vx=V.map(r=>r.step),vk=[...new Set(V.flatMap(r=>Object.keys(r).filter(k=>k.startsWith('psnr_in_')).map(k=>k.slice(8))))];
chart('c_val',vk.map(k=>({x:vx,y:V.map(r=>r['psnr_'+k]-r['psnr_in_'+k]),label:k,dots:true})).concat([{x:vx,y:V.map(r=>r.psnr-r.psnr_in),label:'all',color:'#e6e8ef',dots:true}]),{zero:true});
chart('c_vpsnr',[{x:vx,y:V.map(r=>r.psnr),label:'output',dots:true},{x:vx,y:V.map(r=>r.psnr_in),label:'input',color:'#8a90a2',dots:true},{x:vx,y:V.map(r=>r.clean_psnr==null?null:Math.min(r.clean_psnr,60)),label:'clean in → out (leave alone)',color:COL[2],dots:true}]);
const dx=V.filter(r=>r.detail_fine!=null);chart('c_detail',[{x:dx.map(r=>r.step),y:dx.map(r=>100*r.detail_fine),label:'fine (0.7-1.5 px)',dots:true},{x:dx.map(r=>r.step),y:dx.map(r=>100*r.detail_med),label:'medium (1.5-3 px)',dots:true},{x:dx.map(r=>r.step),y:dx.map(()=>100),label:'clean = 100%',color:'#8a90a2',width:1}],{fmt:v=>v.toFixed(0)+'%'});
const fv=V.filter(r=>r.flicker!=null),fm=M.filter(r=>r.flicker!=null);chart('c_flick',[{x:fm.map(r=>r.step),y:smooth(fm.map(r=>r.flicker),.9),label:'training (smoothed)'},{x:fv.map(r=>r.step),y:fv.map(r=>r.flicker),label:'validation',color:'#e6e8ef',dots:true}]);
const av=V.filter(r=>r.align_psnr!=null);chart('c_align',[{x:av.map(r=>r.step),y:av.map(r=>r.align_psnr),label:'moved by the flow',dots:true},{x:av.map(r=>r.step),y:av.map(r=>r.align_psnr_nowarp),label:'not moved',color:'#8a90a2',dots:true}],{fmt:v=>v.toFixed(1)});
chart('c_fuse',[{x:av.map(r=>r.step),y:av.map(r=>r.fusion_gain),label:'validation',dots:true}],{fmt:v=>v.toFixed(2)});
const ev=V.filter(r=>r.est_ratio_median!=null);chart('c_est',[{x:ev.map(r=>r.step),y:ev.map(r=>r.est_ratio_median),label:'median',dots:true},{x:ev.map(r=>r.step),y:ev.map(r=>r.est_ratio_q25),label:'25%',color:COL[2],dots:true},{x:ev.map(r=>r.step),y:ev.map(r=>r.est_ratio_q75),label:'75%',color:COL[3],dots:true},{x:ev.map(r=>r.step),y:ev.map(()=>1),label:'exact = 1',color:'#8a90a2',width:1}],{fmt:v=>v.toFixed(2)+'x'});
const em=M.filter(r=>r.est_loss!=null);chart('c_estl',[{x:em.map(r=>r.step),y:smooth(em.map(r=>r.est_loss),.9),label:'training (smoothed)'}]);
chart('c_speed',[{x,y:M.map(r=>r.steps_per_s),alpha:.3,width:1},{x,y:smooth(M.map(r=>r.steps_per_s),.8),label:'steps / s'}]);
chart('c_wait',[{x,y:M.map(r=>100*r.data_wait),color:COL[2]}],{fmt:v=>v.toFixed(0)+'%'});
chart('c_lr',[{x,y:M.map(r=>r.lr),color:COL[3]}],{fmt:v=>v.toExponential(1)});
chart('c_mem',[{x,y:M.map(r=>r.gpu_mem_gb),label:'GPU memory (GB)'},{x,y:smooth(M.map(r=>r.grad_norm),.9),label:'gradient norm',color:COL[4]}])}
async function j(u,o){const r=await fetch(u,o);return r.json()}
async function poll(){try{const runs=await j('/api/runs');const sel=$('run');
if(!runs.length){$('main').innerHTML='<div class="empty wide">No runs yet. Start one with <code>python train.py --run D:/denoise_data/runs/&lt;name&gt;</code>.</div>';run=null;return}
const names=runs.map(r=>r.name);if(sel.options.length!==names.length||[...sel.options].some((o,i)=>o.value!==names[i])){sel.innerHTML=names.map(n=>`<option>${n}</option>`).join('')}
if(!run||!names.includes(run)){run=names[0];sel.value=run;M=[];V=[];mNext=0;vNext=0;layout()}
const st=await j(`/api/run/${encodeURIComponent(run)}/status`);const state=st.paused_file&&st.state==='running'?'pausing':st.state||'—';
$('state').textContent=state;$('state').className='badge '+(st.state||'');$('state').title=st.error||'';
// A run whose status has not changed for 5 minutes while "running" has probably died.
if(st.state==='running'&&st.updated&&Date.now()/1000-st.updated>300){$('state').textContent='not responding';$('state').className='badge crashed'}$('step').textContent=(st.step||0).toLocaleString()+' / '+(st.steps||0).toLocaleString();
$('speed').textContent=st.steps_per_s?st.steps_per_s.toFixed(1):'—';$('eta').textContent=st.state==='running'?fmtT(st.eta_s):'—';
$('vgain').textContent=st.val_gain!=null?(st.val_gain>=0?'+':'')+st.val_gain.toFixed(2)+' dB':'—';$('wait').textContent=st.data_wait!=null?Math.round(100*st.data_wait)+'%':'—';
$('prog').style.width=(100*(st.step||0)/(st.steps||1))+'%';$('cfg').textContent=JSON.stringify(Object.assign({gpu:st.gpu,parameters:st.params},st.config),null,1);
const m=await j(`/api/run/${encodeURIComponent(run)}/metrics?from=${mNext}`);M=M.concat(m.rows);mNext=m.next;
const v=await j(`/api/run/${encodeURIComponent(run)}/val?from=${vNext}`);V=V.concat(v.rows);vNext=v.next;
const s=await j(`/api/run/${encodeURIComponent(run)}/samples`);const atEnd=+$('s_rng').value>=samples.length-1;if(s.length!==samples.length){samples=s;$('s_rng').max=Math.max(0,s.length-1);if(atEnd){$('s_rng').value=s.length-1;showSample(s.length-1)}}
const rr=await j(`/api/run/${encodeURIComponent(run)}/realtest`);if(rr.files.length!==real.files.length){const atEnd=+$('r_rng').value>=real.files.length-1;real=rr;$('r_rng').max=Math.max(0,rr.files.length-1);if(atEnd){$('r_rng').value=rr.files.length-1;showReal(rr.files.length-1)}}
draw()}catch(e){console.error(e)}}
$('run').onchange=()=>{run=$('run').value;M=[];V=[];mNext=0;vNext=0;samples=[];layout();poll()};
for(const a of['pause','resume','stop'])$(a).onclick=async()=>{if(!run)return;if(a==='stop'&&!confirm('Stop this run? It saves a checkpoint and exits; --resume continues it.'))return;await j(`/api/run/${encodeURIComponent(run)}/${a}`,{method:'POST'});poll()};
window.onresize=()=>draw();layout();poll();setInterval(poll,4000);
</script></body></html>"""

if __name__ == "__main__":
    main()

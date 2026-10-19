// webui_dsv4.h — self-contained chat UI served at GET /. No CDN, no build step, one string.
//
// NOT include/webui.h: that one is gemma's, down to the title, the "118 tok/s" and the assumption
// that thinking is a single checkbox. This model has TWO orthogonal controls (thinking_mode and
// reasoning_effort, CHAT_FORMAT.md §2), streams reasoning separately as reasoning_content, and
// answers with DSML tool calls that need rendering as structured calls rather than as stray markup.
#pragma once
static const char* WEBUI_DSV4_HTML = R"WEBUI(<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek-V4-Flash-0731-REAP · DSpark · Thor</title>
<style>
:root{--bg:#0b0d12;--bg2:#12151d;--panel:#161a24;--line:#232838;--txt:#e6e9f0;--dim:#8b93a7;--acc:#6ea8fe;--acc2:#b98bfa;--user:#1c2333;--ok:#4ade80;--warn:#fbbf24;--code:#0d1017}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:radial-gradient(1200px 800px at 70% -10%,#161a2e 0%,var(--bg) 55%);color:var(--txt);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;display:flex;flex-direction:column;height:100vh}
header{display:flex;align-items:center;gap:12px;padding:12px 20px;border-bottom:1px solid var(--line);background:rgba(11,13,18,.7);backdrop-filter:blur(10px);position:sticky;top:0;z-index:5}
.logo{font-weight:700;letter-spacing:.3px;background:linear-gradient(90deg,var(--acc),var(--acc2));-webkit-background-clip:text;background-clip:text;color:transparent}
.badge{font-size:11px;color:var(--dim);border:1px solid var(--line);padding:2px 8px;border-radius:999px}
.spacer{flex:1}
.hbtn{background:var(--panel);border:1px solid var(--line);color:var(--txt);border-radius:9px;padding:7px 12px;cursor:pointer;font-size:13px;transition:.15s}
.hbtn:hover{border-color:var(--acc);color:var(--acc)}
#settings{display:none;flex-wrap:wrap;gap:16px;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--bg2);align-items:flex-start}
#settings.open{display:flex}
#settings label{font-size:12px;color:var(--dim);display:flex;flex-direction:column;gap:5px}
#settings input[type=text],#settings textarea,#settings select{background:var(--code);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:8px;font:13px ui-monospace,monospace}
#sys{width:min(520px,52vw);resize:vertical;min-height:38px}
#tools{width:min(520px,52vw);resize:vertical;min-height:38px}
.sw{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--txt)}
input[type=range]{accent-color:var(--acc)}
#chat{flex:1;overflow-y:auto;padding:26px 0}
.wrap{max-width:840px;margin:0 auto;padding:0 20px}
.msg{display:flex;gap:14px;padding:16px 0;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.av{width:30px;height:30px;border-radius:8px;flex:none;display:grid;place-items:center;font-size:11px;font-weight:700}
.av.u{background:var(--user);color:var(--acc)}
.av.a{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#0b0d12}
.body{flex:1;min-width:0;padding-top:3px}
.who{font-size:12px;color:var(--dim);margin-bottom:2px}
.think{border:1px solid var(--line);border-radius:10px;margin:8px 0;overflow:hidden;background:#0e1119}
.think>summary{cursor:pointer;padding:8px 12px;font-size:12.5px;color:var(--acc2);list-style:none;user-select:none;display:flex;align-items:center;gap:8px}
.think>summary::-webkit-details-marker{display:none}
.think>summary::before{content:"\25C6";font-size:10px}
.think .tc{padding:2px 14px 12px;color:var(--dim);font:12.5px/1.55 ui-monospace,monospace;white-space:pre-wrap;border-top:1px solid var(--line)}
.tool{border:1px solid #2b3550;border-radius:10px;margin:8px 0;background:#0e1220;overflow:hidden}
.tool .th{padding:7px 12px;font-size:12px;color:var(--warn);border-bottom:1px solid #2b3550;display:flex;gap:8px;align-items:center}
.tool pre{margin:0;padding:10px 12px;font:12.5px ui-monospace,monospace;overflow-x:auto;color:var(--txt)}
.content{overflow-wrap:anywhere}
.content p{margin:8px 0}
.content h1,.content h2,.content h3{margin:14px 0 6px;line-height:1.3}
.content ul,.content ol{margin:8px 0;padding-left:24px}
.content li{margin:3px 0}
.content blockquote{border-left:3px solid var(--acc);padding-left:12px;color:var(--dim);margin:8px 0}
.content a{color:var(--acc)}
.content table{border-collapse:collapse;margin:10px 0;display:block;overflow-x:auto}
.content th,.content td{border:1px solid var(--line);padding:5px 10px;text-align:left}
.content code.inl{background:var(--code);border:1px solid var(--line);padding:1px 6px;border-radius:6px;font:13px ui-monospace,monospace}
.cb{margin:10px 0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--code)}
.cb .cbh{display:flex;justify-content:space-between;align-items:center;padding:6px 12px;font-size:11px;color:var(--dim);border-bottom:1px solid var(--line)}
.cb .cpy{cursor:pointer;color:var(--dim);background:none;border:none;font-size:11px}
.cb .cpy:hover{color:var(--ok)}
.cb pre{margin:0;padding:12px;overflow-x:auto;font:13px/1.5 ui-monospace,monospace}
.cur::after{content:"\258A";color:var(--acc);animation:bl 1s steps(2) infinite}
@keyframes bl{50%{opacity:0}}
footer{border-top:1px solid var(--line);background:rgba(11,13,18,.7);backdrop-filter:blur(10px);padding:12px 0}
.ibar{max-width:840px;margin:0 auto;padding:0 20px;display:flex;gap:10px;align-items:flex-end}
#inp{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--txt);border-radius:12px;padding:12px 14px;font:15px inherit;resize:none;max-height:200px;min-height:48px}
#inp:focus{outline:none;border-color:var(--acc)}
#send{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#0b0d12;border:none;border-radius:12px;padding:0 20px;height:48px;font-weight:700;cursor:pointer;transition:.15s}
#send:hover{filter:brightness(1.1)}
#send.stop{background:#2a2f40;color:var(--txt)}
.stat{max-width:840px;margin:6px auto 0;padding:0 20px;font-size:11px;color:var(--dim);height:14px}
.empty{text-align:center;color:var(--dim);margin-top:14vh}
.empty h2{font-weight:600;margin-bottom:8px;color:var(--txt)}
.empty code{background:var(--code);border:1px solid var(--line);padding:1px 6px;border-radius:6px;font-size:12px}
</style></head>
<body>
<header>
  <span class="logo">DeepSeek-V4-Flash-0731-REAP</span><span class="badge">pure-CUDA + DSpark speculative decode</span>
  <span class="spacer"></span>
  <button class="hbtn" onclick="document.getElementById('settings').classList.toggle('open')">Settings</button>
  <button class="hbtn" onclick="newChat()">New chat</button>
</header>
<div id="settings">
  <label>System prompt<textarea id="sys" placeholder="(optional)"></textarea></label>
  <label>Tools (OpenAI JSON array, optional)<textarea id="tools" placeholder='[{"type":"function","function":{"name":"get_weather","parameters":{}}}]'></textarea></label>
  <label>Thinking mode
    <select id="tmode"><option value="thinking">thinking</option><option value="chat">chat</option></select></label>
  <label>Reasoning effort
    <select id="effort"><option value="low">low</option><option value="high">high</option><option value="max">max</option></select></label>
  <label>Temperature <span id="tv">1.00</span>
    <input type="range" id="temp" min="0" max="1.5" step="0.05" value="1" oninput="tv.textContent=(+this.value).toFixed(2)"></label>
  <label>top_p <span id="pv">1.00</span>
    <input type="range" id="topp" min="0.05" max="1" step="0.05" value="1" oninput="pv.textContent=(+this.value).toFixed(2)"></label>
  <label>Max tokens<input type="text" id="maxt" value="512" style="width:80px"></label>
</div>
<div id="chat"><div class="wrap" id="msgs"></div></div>
<div class="stat" id="stat"></div>
<footer><div class="ibar">
  <textarea id="inp" placeholder="Message DeepSeek-V4-Flash...  (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
  <button id="send" onclick="onSend()">Send</button>
</div></footer>
<script>
const msgs=[], E=id=>document.getElementById(id);
let streaming=false, ctrl=null;
const inp=E('inp');
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,200)+'px'});
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();onSend()}});
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function md(t){
  const blocks=[]; t=String(t).replace(/```(\w*)\n?([\s\S]*?)```/g,(m,l,c)=>{blocks.push({l,c});return ' '+(blocks.length-1)+' '});
  t=esc(t);
  t=t.replace(/`([^`]+)`/g,'<code class="inl">$1</code>');
  t=t.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<i>$2</i>');
  t=t.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  const lines=t.split('\n'); let out='',inl=null,tbl=null;
  const flush=()=>{if(inl){out+='</'+inl+'>';inl=null}};
  const flushT=()=>{if(tbl){out+='</table>';tbl=null}};
  for(const ln of lines){
    let m;
    if(/^\s*\|(.+)\|\s*$/.test(ln)){
      const cells=ln.trim().slice(1,-1).split('|').map(s=>s.trim());
      if(cells.every(c=>/^:?-{2,}:?$/.test(c)))continue;
      if(!tbl){flush();out+='<table>';tbl=1}
      out+='<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';continue;
    }
    flushT();
    if(m=ln.match(/^(#{1,3})\s+(.*)/)){flush();out+='<h'+m[1].length+'>'+m[2]+'</h'+m[1].length+'>';continue}
    if(m=ln.match(/^\s*>\s?(.*)/)){flush();out+='<blockquote>'+m[1]+'</blockquote>';continue}
    if(m=ln.match(/^\s*[-*]\s+(.*)/)){if(inl!=='ul'){flush();out+='<ul>';inl='ul'}out+='<li>'+m[1]+'</li>';continue}
    if(m=ln.match(/^\s*\d+\.\s+(.*)/)){if(inl!=='ol'){flush();out+='<ol>';inl='ol'}out+='<li>'+m[1]+'</li>';continue}
    flush(); if(ln.trim()==='')continue; out+='<p>'+ln+'</p>';
  }
  flush();flushT();
  out=out.replace(/ (\d+) /g,(m,i)=>{const b=blocks[i];const id='c'+Math.random().toString(36).slice(2,8);
    return '<div class="cb"><div class="cbh"><span>'+(b.l||'code')+'</span><button class="cpy" onclick="cp(this,\''+id+'\')">copy</button></div><pre><code id="'+id+'">'+esc(b.c.replace(/\n$/,''))+'</code></pre></div>'});
  return out;
}
function cp(btn,id){navigator.clipboard.writeText(E(id).textContent);btn.textContent='copied';setTimeout(()=>btn.textContent='copy',1200)}
function toolHtml(tc){
  let args=tc.function&&tc.function.arguments||'{}';
  try{args=JSON.stringify(JSON.parse(args),null,2)}catch(e){}
  return '<div class="tool"><div class="th">tool call &middot; '+esc(tc.function&&tc.function.name||'?')+'</div><pre>'+esc(args)+'</pre></div>';
}
function render(){
  const c=E('msgs');
  if(!msgs.length){c.innerHTML='<div class="empty"><h2>Ready.</h2><div>Local pure-CUDA DeepSeek-V4-Flash-0731-REAP with embedded DSpark speculative decoding on Jetson AGX Thor.</div><div style="margin-top:8px">OpenAI-compatible at <code>/v1/chat/completions</code></div></div>';return}
  c.innerHTML='';
  for(const m of msgs){
    const d=document.createElement('div');d.className='msg';
    const think=m.reasoning?'<details class="think"'+(m.streaming?' open':'')+'><summary>Thinking</summary><div class="tc">'+esc(m.reasoning)+'</div></details>':'';
    const tools=(m.tool_calls||[]).map(toolHtml).join('');
    d.innerHTML='<div class="av '+(m.role==='user'?'u':'a')+'">'+(m.role==='user'?'You':'DS')+'</div>'+
      '<div class="body"><div class="who">'+(m.role==='user'?'You':'DeepSeek-V4-Flash')+'</div>'+think+
      '<div class="content'+(m.streaming?' cur':'')+'">'+(m.role==='user'?esc(m.content):md(m.content))+'</div>'+tools+'</div>';
    c.appendChild(d);
  }
  E('chat').scrollTop=E('chat').scrollHeight;
}
function newChat(){msgs.length=0;E('stat').textContent='';render()}
async function onSend(){
  if(streaming){ctrl&&ctrl.abort();return}
  const txt=inp.value.trim(); if(!txt)return;
  msgs.push({role:'user',content:txt}); inp.value='';inp.style.height='auto';
  const a={role:'assistant',content:'',reasoning:'',tool_calls:[],streaming:true}; msgs.push(a); render();
  const body={messages:[],stream:true,
    temperature:parseFloat(E('temp').value),top_p:parseFloat(E('topp').value),
    max_tokens:parseInt(E('maxt').value)||512,
    thinking_mode:E('tmode').value,reasoning_effort:E('effort').value};
  const sys=E('sys').value.trim(); if(sys)body.messages.push({role:'system',content:sys});
  const tl=E('tools').value.trim();
  if(tl){try{body.tools=JSON.parse(tl)}catch(e){alert('Tools JSON is invalid: '+e.message);return}}
  for(const m of msgs){
    if(m===a)continue;
    if(m.role==='user')body.messages.push({role:'user',content:m.content});
    else body.messages.push({role:'assistant',content:m.content,reasoning_content:m.reasoning||''});
  }
  streaming=true;const sb=E('send');sb.textContent='Stop';sb.classList.add('stop');
  const t0=performance.now();let n=0;ctrl=new AbortController();
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
    if(!r.ok){throw new Error('HTTP '+r.status+': '+(await r.text()).slice(0,300))}
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);
        if(!line.startsWith('data: '))continue;const p=line.slice(6);if(p==='[DONE]')continue;
        let j;try{j=JSON.parse(p)}catch(e){continue}
        if(j.error){a.content+='\n\n*[error: '+(j.error.message||'unknown')+']*';continue}
        const ch=j.choices&&j.choices[0];
        if(ch&&ch.delta){
          if(ch.delta.reasoning_content)a.reasoning+=ch.delta.reasoning_content;
          if(ch.delta.content){a.content+=ch.delta.content;n++}
          if(ch.delta.tool_calls)a.tool_calls=ch.delta.tool_calls;
        }
        if(j.usage){
          const cd=(j.usage.prompt_tokens_details||{}).cached_tokens||0;
          const tm=j.timings||{};
          E('stat').textContent=j.usage.completion_tokens+' tokens, '+(tm.tokens_per_second||0).toFixed(1)+
            ' tok/s, '+(tm.tokens_per_verify||0).toFixed(2)+' tok/verify, prefill '+(tm.prefill_ms||0).toFixed(0)+
            ' ms ('+cd+'/'+j.usage.prompt_tokens+' prompt tokens cached)';
        }
        render();
        if(!E('stat').textContent){const dt=(performance.now()-t0)/1000;E('stat').textContent=n+' tokens, '+(n/dt).toFixed(1)+' tok/s'}
      }
    }
  }catch(e){if(e.name!=='AbortError')a.content+='\n\n*[error: '+e.message+']*'}
  a.streaming=false;streaming=false;sb.textContent='Send';sb.classList.remove('stop');render();
}
render();
</script></body></html>)WEBUI";

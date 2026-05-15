import { useState, useEffect, useCallback } from "react";

const API = import.meta.env.VITE_API_URL ?? "";

// ── Citation full names ────────────────────────────────────────────────────────
const CITATION_NAMES = {
  "POH §7.1":           "Cessna 182Q POH §7.1 — Electrical Systems Emergency",
  "POH §7.3":           "Cessna 182Q POH §7.3 — Vacuum System Failure",
  "POH §4.2":           "Cessna 182Q POH §4.2 — Fuel System (Normal Procedures)",
  "POH §5.1":           "Cessna 182Q POH §5.1 — Performance & Service Ceiling",
  "FAA AC 43.13-1B":    "FAA AC 43.13-1B — Acceptable Maintenance Methods §11 (Electrical)",
  "FAA AC 23.1309-1A":  "FAA AC 23.1309-1A — System Safety Analysis (Flaps/Trim Thresholds)",
  "FAA AC 61-23C":      "FAA AC 61-23C — Pilot's Handbook of Aeronautical Knowledge (Carburetor)",
  "FAA AC 61-27C":      "FAA AC 61-27C — Instrument Flying Handbook (Spatial Disorientation)",
  "FAA-H-8083-25C":     "FAA-H-8083-25C — Pilot's Handbook of Aeronautical Knowledge",
  "FAA-H-8083-3C":      "FAA-H-8083-3C — Airplane Flying Handbook",
  "Continental O-470 Manual": "Continental O-470 Overhaul Manual — Fuel Pump Specifications",
  "Service Manual D2006R4-13": "Cessna Service Manual D2006R4-13 — Battery Specifications (33Ah)",
  "MIL-SPEC-21030":     "MIL-SPEC-21030 — Sealed Lead-Acid Aviation Battery (Internal Resistance)",
};

const getCitationName = (raw) => CITATION_NAMES[raw] || raw;

// ── Component zones ────────────────────────────────────────────────────────────
const ZONES = {
  engine_left:        { label: "Engine",           system: "engine",     color: "#f59e0b" },
  carburetor:         { label: "Carburetor",        system: "engine",     color: "#f59e0b" },
  magnetos:           { label: "Magnetos",          system: "engine",     color: "#f59e0b" },
  alternator:         { label: "Alternator",        system: "electrical", color: "#3b82f6" },
  battery:            { label: "Battery",           system: "electrical", color: "#3b82f6" },
  voltage_regulator:  { label: "Volt. Reg.",        system: "electrical", color: "#3b82f6" },
  vacuum_pump:        { label: "Vacuum Pump",       system: "vacuum",     color: "#8b5cf6" },
  attitude_indicator: { label: "Attitude Ind.",     system: "vacuum",     color: "#8b5cf6" },
  fuel_pump:          { label: "Fuel Pump",         system: "fuel",       color: "#10b981" },
  fuel_selector:      { label: "Fuel Selector",     system: "fuel",       color: "#10b981" },
  fuel_tank_left:     { label: "Left Tank",         system: "fuel",       color: "#10b981" },
  fuel_tank_right:    { label: "Right Tank",        system: "fuel",       color: "#10b981" },
  flaps:              { label: "Flaps",             system: "controls",   color: "#ec4899" },
  trim_system:        { label: "Trim System",       system: "controls",   color: "#ec4899" },
};

const SYSTEM_LABELS = {
  engine: "POWERPLANT", electrical: "ELECTRICAL",
  vacuum: "VACUUM / GYRO", fuel: "FUEL SYSTEM", controls: "FLIGHT CONTROLS",
};

// ── Variation presets for multi-simulation ─────────────────────────────────────
const VARIATION_GROUPS = [
  {
    label: "PILOT FATIGUE",
    description: "Same failure, fatigue ∈ {fresh, moderate, exhausted}",
    variants: [
      { label: "FRESH",      color: "#10b981", mod: ", rested fresh pilot" },
      { label: "MODERATE",   color: "#f59e0b", mod: ", moderately fatigued pilot" },
      { label: "EXHAUSTED",  color: "#ef4444", mod: ", exhausted pilot" },
    ],
  },
  {
    label: "ALTITUDE",
    description: "Same failure, altitude ∈ {1500ft, 5000ft, 12000ft}. Changes recovery time window.",
    variants: [
      { label: "LOW 1500ft",   color: "#ef4444", mod: " at 1500 feet altitude" },
      { label: "MID 5000ft",   color: "#f59e0b", mod: " at 5000 feet altitude" },
      { label: "HIGH 12000ft", color: "#10b981", mod: " at 12000 feet altitude" },
    ],
  },
  {
    label: "WEATHER",
    description: "Same failure, weather ∈ {VMC, IMC, night}. IMC activates R014 spatial disorientation.",
    variants: [
      { label: "VMC",   color: "#10b981", mod: ", visual meteorological conditions" },
      { label: "IMC",   color: "#ef4444", mod: ", instrument meteorological conditions" },
      { label: "NIGHT", color: "#f59e0b", mod: ", night flight conditions" },
    ],
  },
];

// ── Global CSS ─────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Barlow+Condensed:wght@300;400;500;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:#060b14; --bg2:#0c1524; --bg3:#111d2e; --border:#1e3048;
    --amber:#f59e0b; --amber-dim:#92610a; --red:#ef4444; --green:#10b981;
    --blue:#3b82f6; --text:#c8d8e8; --text-dim:#4a6580;
    --mono:'IBM Plex Mono',monospace; --ui:'Barlow Condensed',sans-serif;
  }
  html,body,#root { height:100%; background:var(--bg); color:var(--text); font-family:var(--mono); }
  ::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
  @keyframes pulse-red { 0%,100%{opacity:1;box-shadow:0 0 8px #ef444488} 50%{opacity:.6;box-shadow:0 0 20px #ef444488} }
  @keyframes pulse-amber { 0%,100%{opacity:1} 50%{opacity:.3} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  .fade-in { animation:fadeIn .3s ease forwards; }
  .failed-zone { animation:pulse-red 1.5s ease-in-out infinite; }
`;

const fmt = (v, d = 1) => (v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(d) : v);

// ── Small components ───────────────────────────────────────────────────────────
function Label({ children }) {
  return <div style={{ fontSize:"0.58rem", fontFamily:"var(--ui)", letterSpacing:"0.15em", color:"var(--text-dim)", marginBottom:6, textTransform:"uppercase" }}>{children}</div>;
}

function CitationTag({ raw }) {
  const [show, setShow] = useState(false);
  const full = getCitationName(raw);
  return (
    <span style={{ position:"relative", display:"inline-block" }}>
      <span
        onMouseEnter={()=>setShow(true)} onMouseLeave={()=>setShow(false)}
        style={{ padding:"3px 8px", border:"1px solid var(--amber-dim)", borderRadius:2,
          fontFamily:"var(--mono)", fontSize:"0.7rem", color:"var(--amber)",
          background:"rgba(245,158,11,0.07)", cursor:"help", display:"inline-block" }}
      >{raw}</span>
      {show && (
        <div style={{ position:"absolute", bottom:"130%", left:0, zIndex:200,
          background:"var(--bg3)", border:"1px solid var(--amber-dim)", color:"var(--text)",
          fontSize:"0.68rem", padding:"6px 10px", whiteSpace:"nowrap", borderRadius:2,
          fontFamily:"var(--mono)", maxWidth:420, wordBreak:"break-word", lineHeight:1.5 }}>
          {full}
        </div>
      )}
    </span>
  );
}

function GaugeBar({ label, value, max, unit, crit, warn }) {
  const pct = Math.min(100, (value / max) * 100);
  const color = value <= (crit||0) ? "var(--red)" : value <= (warn||0) ? "var(--amber)" : "var(--green)";
  return (
    <div style={{ marginBottom:10 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
        <span style={{ fontSize:"0.6rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.1em" }}>{label}</span>
        <span style={{ fontSize:"0.7rem", color, fontFamily:"var(--mono)" }}>
          {fmt(value,1)}<span style={{ color:"var(--text-dim)", fontSize:"0.6em", marginLeft:2 }}>{unit}</span>
        </span>
      </div>
      <div style={{ height:3, background:"var(--border)", borderRadius:2 }}>
        <div style={{ height:"100%", width:`${pct}%`, background:color, borderRadius:2, transition:"width .5s,background .3s" }}/>
      </div>
    </div>
  );
}

// ── 4-point state snapshot timeline ───────────────────────────────────────────
function SnapshotTimeline({ snapshots, duration }) {
  const points = [
    { key:"start",   label:"INJECTION",  pct:0 },
    { key:"quarter", label:"25%",        pct:25 },
    { key:"half",    label:"50%",        pct:50 },
    { key:"end",     label:"TERMINAL",   pct:100 },
  ];
  const vars = [
    { k:"v",        l:"Voltage (V)",   max:16,   crit:10.5, warn:11.5 },
    { k:"rpm",      l:"RPM",           max:2400, crit:200,  warn:800 },
    { k:"pump_eff", l:"Pump Eff",      max:1,    crit:0.1,  warn:0.5 },
    { k:"alt",      l:"Altitude (ft)", max:16500,crit:500,  warn:2000 },
  ];
  return (
    <div>
      <Label>STATE SNAPSHOTS — 4-POINT CASCADE TIMELINE</Label>
      <div style={{ overflowX:"auto" }}>
        <table style={{ width:"100%", borderCollapse:"collapse", fontSize:"0.68rem", fontFamily:"var(--mono)" }}>
          <thead>
            <tr>
              <th style={{ textAlign:"left", color:"var(--text-dim)", fontWeight:400, padding:"4px 8px", borderBottom:"1px solid var(--border)" }}>VARIABLE</th>
              {points.map(p => (
                <th key={p.key} style={{ textAlign:"right", color:"var(--text-dim)", fontWeight:400, padding:"4px 8px", borderBottom:"1px solid var(--border)", fontSize:"0.58rem", letterSpacing:"0.08em" }}>
                  {p.label}<br/>
                  <span style={{ color:"var(--text-dim)", fontSize:"0.55em" }}>t={p.pct === 0 ? 0 : p.pct === 100 ? fmt(duration,0) : fmt((duration||0)*p.pct/100,0)}s</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {vars.map(v => (
              <tr key={v.k}>
                <td style={{ padding:"5px 8px", color:"var(--text-dim)", fontSize:"0.62rem" }}>{v.l}</td>
                {points.map(p => {
                  const s = snapshots?.[p.key] || {};
                  const val = parseFloat(s[v.k] ?? 0);
                  const color = val <= v.crit ? "var(--red)" : val <= v.warn ? "var(--amber)" : "var(--text)";
                  return (
                    <td key={p.key} style={{ textAlign:"right", padding:"5px 8px", color, fontWeight: p.key==="end" ? 500 : 400 }}>
                      {fmt(val, v.k==="pump_eff" ? 3 : v.k==="alt" ? 0 : 1)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Kill zone display (properly parsed) ───────────────────────────────────────
function KillZoneDisplay({ killZone, crashProb }) {
  if (!killZone || !Object.keys(killZone).length) return null;

  const renderValue = (k, v) => {
    if (k === "initial_conditions" && typeof v === "object" && !Array.isArray(v)) {
      return (
        <div style={{ marginLeft:16, marginTop:4 }}>
          {Object.entries(v).map(([feat, detail]) => (
            <div key={feat} style={{ marginBottom:4 }}>
              <span style={{ color:"var(--text-dim)", fontSize:"0.6rem" }}>{feat}: </span>
              <span style={{ color:"var(--amber)", fontSize:"0.65rem" }}>
                {typeof detail === "object" ? (
                  `${detail.cat_range?.[0]?.toFixed(2)} – ${detail.cat_range?.[1]?.toFixed(2)}`
                  + ` (${detail.interpretation?.replace(/_/g," ") || ""})`
                ) : String(detail)}
              </span>
            </div>
          ))}
        </div>
      );
    }
    if (Array.isArray(v)) {
      return <span style={{ color:"var(--amber)", fontSize:"0.7rem" }}>[{v[0]?.toFixed?.(2) ?? v[0]}, {v[1]?.toFixed?.(2) ?? v[1]}]</span>;
    }
    if (typeof v === "object") {
      return <span style={{ color:"var(--amber)", fontSize:"0.65rem" }}>{JSON.stringify(v).slice(0,80)}</span>;
    }
    return <span style={{ color:"var(--amber)", fontSize:"0.7rem" }}>{String(v)}</span>;
  };

  return (
    <div style={{ border:"1px solid rgba(239,68,68,0.3)", borderRadius:2, padding:16, background:"rgba(239,68,68,0.04)" }}>
      <Label>KILL ZONE · UNRECOVERABLE STATE SPACE</Label>
      {Object.entries(killZone).map(([k, v]) => (
        <div key={k} style={{ marginBottom:8 }}>
          <div style={{ fontSize:"0.6rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.1em", marginBottom:1 }}>{k.replace(/_/g," ").toUpperCase()}</div>
          {renderValue(k, v)}
        </div>
      ))}
      {crashProb != null && (
        <div style={{ marginTop:12, paddingTop:10, borderTop:"1px solid rgba(239,68,68,0.2)", display:"flex", justifyContent:"space-between" }}>
          <span style={{ fontSize:"0.62rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.1em" }}>CRASH PROBABILITY IN THIS ZONE</span>
          <span style={{ fontFamily:"var(--mono)", fontSize:"0.9rem", color:"var(--red)" }}>{(crashProb*100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
}

// ── C182 SVG Schematic ─────────────────────────────────────────────────────────
function C182Schematic({ failedComponents=[], activeComponent, onSelect }) {
  const getColor = (comp) => {
    if (failedComponents.includes(comp)) return "#ef4444";
    if (comp === activeComponent) return "#ffffff";
    return ZONES[comp]?.color || "#fff";
  };
  const ZoneCircle = ({ comp, cx, cy, r=10 }) => (
    <g className={failedComponents.includes(comp) ? "failed-zone" : ""} onClick={()=>onSelect(comp)} style={{cursor:"pointer"}}>
      <circle cx={cx} cy={cy} r={r} fill={getColor(comp)}
        fillOpacity={failedComponents.includes(comp) ? 0.9 : comp===activeComponent ? 1 : 0.5}
        stroke={comp===activeComponent ? "#fff" : getColor(comp)} strokeWidth={comp===activeComponent ? 2 : 1} strokeOpacity={0.8}/>
      <circle cx={cx} cy={cy} r={r+8} fill="transparent"/>
    </g>
  );
  return (
    <svg viewBox="0 0 300 500" style={{width:"100%",maxWidth:260,display:"block"}}>
      {[0,1,2,3,4,5].map(i=><line key={`h${i}`} x1={0} y1={i*100} x2={300} y2={i*100} stroke="#1e3048" strokeWidth={0.5} strokeDasharray="2,4"/>)}
      {[0,1,2].map(i=><line key={`v${i}`} x1={i*150} y1={0} x2={i*150} y2={500} stroke="#1e3048" strokeWidth={0.5} strokeDasharray="2,4"/>)}
      <ellipse cx={150} cy={250} rx={28} ry={195} fill="#0c1524" stroke="#1e3048" strokeWidth={1.5}/>
      <ellipse cx={150} cy={72} rx={22} ry={30} fill="#111d2e" stroke="#2a4060" strokeWidth={1.5}/>
      <ellipse cx={150} cy={60} rx={14} ry={16} fill="#0c1524" stroke="#2a4060" strokeWidth={1}/>
      <ellipse cx={150} cy={42} rx={40} ry={5} fill="#1e3048" stroke="#2a4060" strokeWidth={1}/>
      <ellipse cx={150} cy={46} rx={5} ry={40} fill="#1e3048" stroke="#2a4060" strokeWidth={1}/>
      <path d="M 130 200 L 20 270 L 15 310 L 125 265 Z" fill="#0c1524" stroke="#1e3048" strokeWidth={1.5}/>
      <path d="M 170 200 L 280 270 L 285 310 L 175 265 Z" fill="#0c1524" stroke="#1e3048" strokeWidth={1.5}/>
      <path d="M 20 270 L 15 310 L 55 295 L 60 258 Z" fill="#1a2a3a" stroke="#1e3048" strokeWidth={1}/>
      <path d="M 280 270 L 285 310 L 245 295 L 240 258 Z" fill="#1a2a3a" stroke="#1e3048" strokeWidth={1}/>
      <path d="M 130 415 L 60 435 L 60 455 L 130 440 Z" fill="#0c1524" stroke="#1e3048" strokeWidth={1.5}/>
      <path d="M 170 415 L 240 435 L 240 455 L 170 440 Z" fill="#0c1524" stroke="#1e3048" strokeWidth={1.5}/>
      <path d="M 145 415 L 148 470 L 152 470 L 155 415 Z" fill="#111d2e" stroke="#1e3048" strokeWidth={1}/>
      <ZoneCircle comp="engine_left"        cx={150} cy={80}  r={12}/>
      <ZoneCircle comp="carburetor"         cx={135} cy={98}  r={8}/>
      <ZoneCircle comp="magnetos"           cx={165} cy={98}  r={8}/>
      <ZoneCircle comp="alternator"         cx={172} cy={175} r={9}/>
      <ZoneCircle comp="battery"            cx={172} cy={198} r={9}/>
      <ZoneCircle comp="voltage_regulator"  cx={172} cy={220} r={8}/>
      <ZoneCircle comp="vacuum_pump"        cx={128} cy={180} r={9}/>
      <ZoneCircle comp="attitude_indicator" cx={128} cy={202} r={8}/>
      <ZoneCircle comp="fuel_pump"          cx={150} cy={210} r={8}/>
      <ZoneCircle comp="fuel_selector"      cx={150} cy={230} r={8}/>
      <ZoneCircle comp="fuel_tank_left"     cx={60}  cy={275} r={11}/>
      <ZoneCircle comp="fuel_tank_right"    cx={240} cy={275} r={11}/>
      <ZoneCircle comp="flaps"             cx={40}  cy={285} r={8}/>
      <ZoneCircle comp="trim_system"       cx={150} cy={395} r={9}/>
      <line x1={150} y1={50} x2={150} y2={465} stroke="#1e3048" strokeWidth={0.5} strokeDasharray="1,6"/>
      <text x={150} y={490} textAnchor="middle" fontSize="7" fill="#2a4060" fontFamily="var(--mono)">↑ FWD</text>
    </svg>
  );
}

// ── Cascade timeline ───────────────────────────────────────────────────────────
function CascadeTimeline({ failureChain, duration, onSelect, active }) {
  if (!failureChain?.length) return <div style={{color:"var(--text-dim)",fontSize:"0.7rem",fontFamily:"var(--mono)"}}>No cascade chain recorded.</div>;
  const maxT = duration || 60;
  const allAtZero = failureChain.every(e => (e.time_s || 0) < 1);
  return (
    <div>
      <Label>CASCADE TIMELINE</Label>
      {allAtZero && (
        <div style={{ fontSize:"0.62rem", color:"var(--amber)", fontFamily:"var(--mono)", marginBottom:8, padding:"4px 8px", background:"rgba(245,158,11,0.08)", borderRadius:2, border:"1px solid var(--amber-dim)" }}>
          ⚠ All events at t=0 — failure injected at simulation start. Downstream cascade propagation
          events (pump degradation, engine stop) are visible in state snapshots above.
        </div>
      )}
      <div style={{ position:"relative", height:40, marginBottom:8 }}>
        <div style={{ position:"absolute", top:18, left:0, right:0, height:1, background:"var(--border)" }}/>
        {[0,0.25,0.5,0.75,1].map(f=>(
          <div key={f} style={{ position:"absolute", left:`${f*100}%`, top:22, transform:"translateX(-50%)", fontSize:"0.55rem", color:"var(--text-dim)", fontFamily:"var(--mono)" }}>
            {Math.round(f*maxT)}s
          </div>
        ))}
        {failureChain.map((ev,i) => {
          const pct = (ev.time_s / maxT) * 100;
          const color = ev.severity === "catastrophic" ? "var(--red)" : "var(--amber)";
          return (
            <div key={i} onClick={()=>onSelect(ev.component)}
              style={{ position:"absolute", left:`${pct}%`, top:8, transform:"translateX(-50%)", cursor:"pointer" }}>
              <div style={{ width:12, height:12, borderRadius:"50%",
                background:ev.component===active?"#fff":color,
                border:`2px solid ${color}`,
                boxShadow:ev.component===active?"0 0 8px #fff":`0 0 6px ${color}`,
                transition:"all .2s" }}/>
            </div>
          );
        })}
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:3 }}>
        {failureChain.map((ev,i) => (
          <div key={i} onClick={()=>onSelect(ev.component)}
            style={{ display:"flex", alignItems:"center", gap:10, cursor:"pointer",
              padding:"4px 8px", borderRadius:2,
              background:ev.component===active?"rgba(245,158,11,0.08)":"transparent",
              border:ev.component===active?"1px solid var(--amber-dim)":"1px solid transparent",
              transition:"all .15s" }}>
            <span style={{ color:"var(--text-dim)", fontFamily:"var(--mono)", fontSize:"0.63rem", width:52, flexShrink:0 }}>
              t={fmt(ev.time_s,0)}s
            </span>
            <span style={{ width:6, height:6, borderRadius:"50%", background:ev.severity==="catastrophic"?"var(--red)":"var(--amber)", flexShrink:0 }}/>
            <span style={{ fontFamily:"var(--mono)", fontSize:"0.72rem", color:ev.component===active?"var(--amber)":"var(--text)" }}>{ev.component}</span>
            <span style={{ fontFamily:"var(--mono)", fontSize:"0.58rem", color:"var(--text-dim)", marginLeft:"auto" }}>{ev.severity}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Multi-simulation comparison ────────────────────────────────────────────────
function MultiSimPanel({ baseText, baseResult }) {
  const [activeGroup, setActiveGroup] = useState(0);
  const [runs, setRuns] = useState({});
  const [loading, setLoading] = useState(false);

  const runVariations = async (groupIdx) => {
    setLoading(true);
    const group = VARIATION_GROUPS[groupIdx];
    const cleaned = baseText
      .replace(/\b(fresh|rested|tired|fatigued|exhausted|moderate)\s*(pilot)?\b/gi, "")
      .replace(/\bat\s+\d{3,5}\s*f(ee)?t?\b/gi, "")
      .replace(/\b(visual|instrument|imc|vmc|night)\s*(conditions|flight|meteorological)?\b/gi, "")
      .replace(/,\s*,/g, ",").replace(/\s+/g, " ").trim();

    const results = await Promise.allSettled(
      group.variants.map(v =>
        fetch(`${API}/api/run_full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: cleaned + v.mod }),
        }).then(r => r.json())
      )
    );
    setRuns(prev => ({
      ...prev,
      [groupIdx]: results.map((r, i) => ({
        variant: group.variants[i],
        data: r.status === "fulfilled" ? r.value : null,
        error: r.status === "rejected" ? r.reason?.message : null,
      })),
    }));
    setLoading(false);
  };

  const currentRuns = runs[activeGroup] || [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", gap: 8 }}>
        {VARIATION_GROUPS.map((g, i) => (
          <button key={i} onClick={() => setActiveGroup(i)}
            style={{
              background: activeGroup === i ? "rgba(245,158,11,0.15)" : "var(--bg2)",
              border: `1px solid ${activeGroup === i ? "var(--amber)" : "var(--border)"}`,
              color: activeGroup === i ? "var(--amber)" : "var(--text-dim)",
              padding: "6px 14px", borderRadius: 2, cursor: "pointer",
              fontFamily: "var(--ui)", fontWeight: 600, fontSize: "0.68rem",
              letterSpacing: "0.1em", transition: "all .15s",
            }}>
            {g.label}
          </button>
        ))}
      </div>
      <div style={{ border: "1px solid var(--border)", borderRadius: 2, padding: 16, background: "var(--bg2)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
          <div>
            <Label>{VARIATION_GROUPS[activeGroup].label} SWEEP</Label>
            <div style={{ fontSize: "0.62rem", color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
              {VARIATION_GROUPS[activeGroup].description}
            </div>
          </div>
          <button onClick={() => runVariations(activeGroup)} disabled={loading}
            style={{
              background: loading ? "var(--border)" : "var(--amber)",
              color: loading ? "var(--text-dim)" : "var(--bg)",
              border: "none", borderRadius: 2, padding: "6px 16px",
              fontFamily: "var(--ui)", fontWeight: 600, fontSize: "0.7rem",
              cursor: loading ? "default" : "pointer", letterSpacing: "0.1em", flexShrink: 0,
            }}>
            {loading ? "RUNNING..." : "RUN →"}
          </button>
        </div>
        {currentRuns.length > 0 ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
            {currentRuns.map((run, i) => {
              const r = run.data?.result;
              const isCAT = r?.outcome === "catastrophic";
              const crashProb = r?.l3_analysis?.snapshot?.crash_probability;
              return (
                <div key={i} style={{
                  border: `1px solid ${isCAT ? "rgba(239,68,68,0.4)" : "rgba(16,185,129,0.4)"}`,
                  borderRadius: 2, padding: 12,
                  background: isCAT ? "rgba(239,68,68,0.04)" : "rgba(16,185,129,0.04)",
                }}>
                  <div style={{ fontFamily: "var(--ui)", fontWeight: 700, fontSize: "0.65rem",
                    letterSpacing: "0.15em", color: run.variant.color, marginBottom: 6 }}>
                    {run.variant.label}
                  </div>
                  {run.error ? (
                    <div style={{ color: "var(--text-dim)", fontSize: "0.65rem" }}>{run.error}</div>
                  ) : r ? (
                    <>
                      <div style={{ fontFamily: "var(--ui)", fontWeight: 700, fontSize: "1.6rem",
                        color: isCAT ? "var(--red)" : "var(--green)", marginBottom: 4 }}>
                        {isCAT ? "CATAS." : "RECOV."}
                      </div>
                      {r.time_to_criticality_s ? (
                        <div style={{ fontFamily: "var(--mono)", fontSize: "0.72rem", color: "var(--text-dim)" }}>
                          TTC: <span style={{ color: "var(--red)" }}>{fmt(r.time_to_criticality_s, 0)}s</span>
                        </div>
                      ) : (
                        <div style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--green)" }}>
                          pilot managed
                        </div>
                      )}
                      {r.unrecoverable_reason && (
                        <div style={{ fontFamily: "var(--mono)", fontSize: "0.6rem", color: "var(--red)", marginTop: 3 }}>
                          {r.unrecoverable_reason.replace(/_/g, " ")}
                        </div>
                      )}
                      {crashProb != null && (
                        <div style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--text-dim)", marginTop: 4 }}>
                          ML: {(crashProb * 100).toFixed(0)}%
                        </div>
                      )}
                    </>
                  ) : <div style={{ color: "var(--text-dim)", fontSize: "0.65rem" }}>no data</div>}
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "24px 0",
            fontSize: "0.65rem", color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
            Click RUN → to simulate 3 variations
          </div>
        )}
      </div>
    </div>
  );
}

// ── Input screen ───────────────────────────────────────────────────────────────
const EXAMPLES = [
  "Cessna 182 at 8000 feet, IMC, student pilot, alternator fails",
  "C182 low altitude approach, tired commercial pilot, carburetor icing",
  "Cessna 182 cruise, vacuum pump and alternator fail simultaneously",
  "Night flight at 5000ft, fatigued pilot, fuel selector on empty tank",
];

function InputScreen({ onSubmit }) {
  const [text, setText] = useState("");
  return (
    <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center", padding:"40px 24px", position:"relative" }}>
      <div style={{ position:"fixed", inset:0, pointerEvents:"none",
        background:"repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(6,11,20,.15) 2px,rgba(6,11,20,.15) 4px)" }}/>
      <div style={{ position:"relative", zIndex:1, width:"100%", maxWidth:680 }}>
        <div style={{ marginBottom:48, textAlign:"center" }}>
          <div style={{ fontFamily:"var(--ui)", fontWeight:700, fontSize:"0.62rem",
            letterSpacing:"0.35em", color:"var(--amber)", marginBottom:14 }}>
            CESSNA 182 · FAILURE CASCADE ANALYSIS SYSTEM
          </div>
          <div style={{ fontFamily:"var(--ui)", fontWeight:700, fontSize:"3.4rem",
            letterSpacing:"-0.02em", lineHeight:1, color:"#e8f0f8" }}>
            FAILURE<span style={{color:"var(--amber)"}}>SCOPE</span>
          </div>
          <div style={{ marginTop:10, fontSize:"0.68rem", color:"var(--text-dim)",
            fontFamily:"var(--mono)", letterSpacing:"0.04em", lineHeight:1.6 }}>
            ML-driven cascade discovery · POH/FAA sourced physics · NTSB validated
          </div>
        </div>
        <div style={{ border:"1px solid var(--border)", background:"var(--bg2)", borderRadius:4, overflow:"hidden" }}>
          <div style={{ padding:"8px 16px", borderBottom:"1px solid var(--border)",
            display:"flex", alignItems:"center", gap:8,
            fontSize:"0.58rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.15em" }}>
            <span style={{color:"var(--amber)"}}>●</span> SCENARIO INPUT · NATURAL LANGUAGE
          </div>
          <textarea value={text} onChange={e=>setText(e.target.value)}
            placeholder="Describe the scenario in plain English: aircraft type, altitude, pilot state, failure..."
            onKeyDown={e=>{if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)&&text.trim())onSubmit(text);}}
            style={{ width:"100%", minHeight:110, background:"transparent", border:"none", outline:"none",
              color:"var(--text)", fontFamily:"var(--mono)", fontSize:"0.875rem",
              padding:"14px 16px", resize:"vertical", lineHeight:1.6 }}/>
          <div style={{ padding:"8px 16px", borderTop:"1px solid var(--border)",
            display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={{ fontSize:"0.58rem", color:"var(--text-dim)", fontFamily:"var(--mono)" }}>⌘↵ to run</span>
            <button onClick={()=>text.trim()&&onSubmit(text)} disabled={!text.trim()}
              style={{ background:text.trim()?"var(--amber)":"var(--border)",
                color:text.trim()?"var(--bg)":"var(--text-dim)", border:"none", borderRadius:2,
                padding:"8px 24px", fontFamily:"var(--ui)", fontWeight:600, fontSize:"0.72rem",
                letterSpacing:"0.1em", cursor:text.trim()?"pointer":"default", transition:"all .2s" }}>
              RUN ANALYSIS →
            </button>
          </div>
        </div>
        <div style={{ marginTop:20 }}>
          <div style={{ fontSize:"0.58rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.15em", marginBottom:8 }}>EXAMPLE SCENARIOS</div>
          <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
            {EXAMPLES.map((ex,i)=>(
              <button key={i} onClick={()=>setText(ex)}
                style={{ background:"transparent", border:"1px solid var(--border)", color:"var(--text-dim)",
                  textAlign:"left", padding:"7px 12px", borderRadius:2, cursor:"pointer",
                  fontFamily:"var(--mono)", fontSize:"0.7rem", transition:"all .15s" }}
                onMouseEnter={e=>{e.currentTarget.style.borderColor="var(--amber)";e.currentTarget.style.color="var(--text)";}}
                onMouseLeave={e=>{e.currentTarget.style.borderColor="var(--border)";e.currentTarget.style.color="var(--text-dim)";}}>
                {ex}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Loading screen ─────────────────────────────────────────────────────────────
const STAGES = [
  { id:"parse",    label:"Parsing scenario via NLP",     sub:"Ollama mistral:7b · Layer 0" },
  { id:"simulate", label:"Running physics simulation",   sub:"C182 POH + FAA physics · Layer 1-2" },
  { id:"analyze",  label:"Analyzing failure cascade",    sub:"XGBoost + Transformer · Layer 3" },
  { id:"validate", label:"Validating against NTSB data", sub:"21,761 Cessna records · Layer 4" },
];
function LoadingScreen({ stage }) {
  const idx = STAGES.findIndex(s=>s.id===stage);
  return (
    <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center", padding:40 }}>
      <div style={{ fontFamily:"var(--ui)", fontWeight:700, fontSize:"0.6rem",
        letterSpacing:"0.3em", color:"var(--amber)", marginBottom:48 }}>
        FAILURESCOPE · ANALYSIS IN PROGRESS
      </div>
      <div style={{ width:"100%", maxWidth:460 }}>
        {STAGES.map((s,i) => {
          const done=i<idx, active=i===idx, pending=i>idx;
          return (
            <div key={s.id} style={{ display:"flex", gap:14, marginBottom:22, opacity:pending?.3:1, transition:"opacity .3s" }}>
              <div style={{ width:26, height:26, border:`1px solid ${done?"var(--green)":active?"var(--amber)":"var(--border)"}`,
                borderRadius:2, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0,
                background:done?"var(--green)":"transparent",
                animation:active?"pulse-amber 1.5s infinite":"none" }}>
                {done
                  ? <span style={{color:"var(--bg)",fontSize:"0.7rem"}}>✓</span>
                  : <span style={{color:active?"var(--amber)":"var(--border)",fontSize:"0.5rem"}}>◆</span>}
              </div>
              <div>
                <div style={{ fontFamily:"var(--ui)", fontWeight:600, fontSize:"0.88rem",
                  color:done?"var(--green)":active?"var(--amber)":"var(--text-dim)" }}>{s.label}</div>
                <div style={{ fontSize:"0.62rem", color:"var(--text-dim)", fontFamily:"var(--mono)", marginTop:2 }}>{s.sub}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Results screen ─────────────────────────────────────────────────────────────
function ResultsScreen({ data, input, onReset }) {
  const [activeComp, setActiveComp] = useState(null);
  const [tab, setTab] = useState("overview");

  const r = data.result;
  const isCAT = r.outcome === "catastrophic";
  const failedComps = r.failure_chain?.map(f=>f.component) || [];
  const l4 = r.l4_findings || {};
  const l3 = r.l3_analysis || {};
  const snap = r.state_snapshots || {};
  const endState = snap.end || {};
  const outcomeColor = isCAT ? "var(--red)" : "var(--green)";

  // L3/L4 reliability note for inference resolution mismatch
  const l3Reliable = r.duration_s != null && r.duration_s < 300;

  return (
    <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column" }}>
      {/* Header */}
      <div style={{ borderBottom:"1px solid var(--border)", padding:"9px 20px",
        display:"flex", alignItems:"center", justifyContent:"space-between", background:"var(--bg2)" }}>
        <div style={{ fontFamily:"var(--ui)", fontWeight:700, fontSize:"1rem" }}>
          FAILURE<span style={{color:"var(--amber)"}}>SCOPE</span>
        </div>
        <div style={{ fontFamily:"var(--mono)", fontSize:"0.62rem", color:"var(--text-dim)",
          maxWidth:400, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
          {input}
        </div>
        <button onClick={onReset}
          style={{ background:"transparent", border:"1px solid var(--border)", color:"var(--text-dim)",
            padding:"5px 14px", borderRadius:2, fontFamily:"var(--ui)", fontSize:"0.68rem",
            cursor:"pointer", letterSpacing:"0.1em" }}
          onMouseEnter={e=>e.currentTarget.style.borderColor="var(--amber)"}
          onMouseLeave={e=>e.currentTarget.style.borderColor="var(--border)"}>
          NEW SCENARIO
        </button>
      </div>

      {/* Outcome banner */}
      <div style={{ padding:"14px 20px", borderBottom:`1px solid ${outcomeColor}44`,
        background:`${outcomeColor}08`, display:"flex", alignItems:"center", gap:24, flexWrap:"wrap" }}>
        <div>
          <div style={{ fontSize:"0.52rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.2em", marginBottom:2 }}>SIMULATION OUTCOME · L2 PHYSICS (GROUND TRUTH)</div>
          <div style={{ fontFamily:"var(--ui)", fontWeight:700, fontSize:"2.2rem", letterSpacing:"0.05em", color:outcomeColor,
            animation:isCAT?"pulse-red 2s infinite":"none" }}>
            {isCAT ? "CATASTROPHIC" : "RECOVERABLE"}
          </div>
        </div>
        {r.time_to_criticality_s && (
          <div>
            <div style={{ fontSize:"0.52rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.2em", marginBottom:2 }}>TIME TO CRITICALITY</div>
            <div style={{ fontFamily:"var(--mono)", fontSize:"1.8rem", color:"var(--red)" }}>
              {fmt(r.time_to_criticality_s,0)}<span style={{fontSize:"0.8rem",marginLeft:4,color:"var(--text-dim)"}}>sec</span>
            </div>
          </div>
        )}
        <div>
          <div style={{ fontSize:"0.52rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.2em", marginBottom:2 }}>SIM DURATION</div>
          <div style={{ fontFamily:"var(--mono)", fontSize:"1.8rem", color:"var(--text)" }}>
            {fmt(r.duration_s,0)}<span style={{fontSize:"0.8rem",marginLeft:4,color:"var(--text-dim)"}}>sec</span>
          </div>
        </div>
        {r.unrecoverable_reason && (
          <div>
            <div style={{ fontSize:"0.52rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.2em", marginBottom:4 }}>FAILURE MODE</div>
            <div style={{
              padding:"5px 12px", borderRadius:2,
              background:"rgba(239,68,68,0.12)", border:"1px solid rgba(239,68,68,0.4)",
              fontFamily:"var(--mono)", fontSize:"0.78rem", color:"var(--red)", letterSpacing:"0.04em",
            }}>
              {r.unrecoverable_reason.replace(/_/g," ").toUpperCase()}
            </div>
          </div>
        )}
        <div style={{ marginLeft:"auto" }}>
          <div style={{ fontSize:"0.52rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.2em", marginBottom:2 }}>
            L4 VALIDATION {!l4.found && "(COMPONENT NOT IN FINDINGS)"}
          </div>
          <div style={{ fontFamily:"var(--mono)", fontSize:"0.9rem",
            color:l4.found ? (l4.validation_status==="VALIDATED"?"var(--green)":"var(--amber)") : "var(--text-dim)" }}>
            {l4.found ? `${l4.validation_status} ${l4.combined_confidence?.toFixed(1)}%` : "L3 ONLY (NO L4 ENTRY)"}
          </div>
        </div>
      </div>

      {/* Main grid */}
      <div style={{ flex:1, display:"grid", gridTemplateColumns:"250px 1fr", overflow:"hidden" }}>
        {/* Schematic sidebar */}
        <div style={{ borderRight:"1px solid var(--border)", padding:14, overflowY:"auto", display:"flex", flexDirection:"column", gap:12 }}>
          <div style={{ fontSize:"0.58rem", fontFamily:"var(--ui)", letterSpacing:"0.15em", color:"var(--text-dim)" }}>
            C182 SCHEMATIC · CLICK COMPONENT
          </div>
          <C182Schematic failedComponents={failedComps} activeComponent={activeComp}
            onSelect={c=>setActiveComp(activeComp===c?null:c)}/>
          <div style={{ borderTop:"1px solid var(--border)", paddingTop:10 }}>
            {[{c:"var(--red)",l:"FAILED / CASCADE SOURCE"},{c:"var(--text-dim)",l:"NOMINAL"}].map(v=>(
              <div key={v.l} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
                <div style={{ width:7, height:7, borderRadius:"50%", background:v.c }}/>
                <span style={{ fontSize:"0.58rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.1em" }}>{v.l}</span>
              </div>
            ))}
          </div>
          {activeComp && ZONES[activeComp] && (
            <div className="fade-in" style={{ borderTop:"1px solid var(--border)", paddingTop:10 }}>
              <div style={{ fontFamily:"var(--ui)", fontWeight:600, fontSize:"0.9rem",
                color:failedComps.includes(activeComp)?"var(--red)":"var(--text)", marginBottom:3 }}>
                {ZONES[activeComp].label}
              </div>
              <div style={{ fontSize:"0.58rem", fontFamily:"var(--ui)", color:"var(--text-dim)", letterSpacing:"0.1em", marginBottom:6 }}>
                {SYSTEM_LABELS[ZONES[activeComp].system]}
              </div>
              <div style={{ padding:"5px 8px", borderRadius:2, fontSize:"0.63rem", fontFamily:"var(--mono)",
                background:failedComps.includes(activeComp)?"rgba(239,68,68,.1)":"var(--bg3)",
                color:failedComps.includes(activeComp)?"var(--red)":"var(--text-dim)",
                border:`1px solid ${failedComps.includes(activeComp)?"rgba(239,68,68,.3)":"var(--border)"}` }}>
                {failedComps.includes(activeComp) ? "FAILED — CASCADE SOURCE" : "NOMINAL"}
              </div>
              {l4.found && failedComps.includes(activeComp) && l4.cascade_variables?.length > 0 && (
                <div style={{ marginTop:8, fontSize:"0.6rem", fontFamily:"var(--mono)", color:"var(--text-dim)", lineHeight:1.5 }}>
                  <span style={{color:"var(--amber)"}}>cascade: </span>
                  {l4.cascade_variables.join(" × ")}
                </div>
              )}
              {!l4.found && failedComps.includes(activeComp) && (
                <div style={{ marginTop:6, fontSize:"0.58rem", color:"var(--amber)", fontFamily:"var(--mono)", lineHeight:1.5 }}>
                  ⚠ No L4 entry — component not in scenario generator training set. L3 ML analysis still available.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Tabs + content */}
        <div style={{ overflowY:"auto", display:"flex", flexDirection:"column" }}>
          <div style={{ display:"flex", borderBottom:"1px solid var(--border)", padding:"0 20px" }}>
            {[{id:"overview",l:"OVERVIEW"},{id:"cascade",l:"CASCADE ANALYSIS"},{id:"evidence",l:"VALIDATION & EVIDENCE"},{id:"variations",l:"PARAMETER SWEEP"}].map(t=>(
              <button key={t.id} onClick={()=>setTab(t.id)}
                style={{ background:"transparent", border:"none",
                  borderBottom:tab===t.id?"2px solid var(--amber)":"2px solid transparent",
                  color:tab===t.id?"var(--amber)":"var(--text-dim)",
                  padding:"10px 14px", cursor:"pointer", fontFamily:"var(--ui)",
                  fontWeight:600, fontSize:"0.68rem", letterSpacing:"0.1em", transition:"color .15s" }}>
                {t.l}
              </button>
            ))}
          </div>

          <div style={{ padding:20, flex:1 }}>

            {/* ── OVERVIEW ── */}
            {tab === "overview" && (
              <div className="fade-in" style={{ display:"flex", flexDirection:"column", gap:20 }}>
                {/* State snapshots 4-point */}
                {snap && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:16, background:"var(--bg2)" }}>
                    <SnapshotTimeline snapshots={snap} duration={r.duration_s}/>
                  </div>
                )}
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                  {/* Initial conditions */}
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>INITIAL CONDITIONS</Label>
                    {[
                      {l:"Altitude",    v:`${fmt(r.initial?.altitude_ft,0)} ft`},
                      {l:"Airspeed",    v:`${fmt(r.initial?.airspeed_kts,0)} kts`},
                      {l:"Pilot skill", v:r.initial?.skill||"—"},
                      {l:"Fatigue",     v:fmt(r.initial?.fatigue,2)},
                    ].map(d=>(
                      <div key={d.l} style={{ marginBottom:7 }}>
                        <div style={{ fontSize:"0.58rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.1em", marginBottom:1 }}>{d.l}</div>
                        <div style={{ fontFamily:"var(--mono)", fontSize:"0.82rem", color:"var(--text)" }}>{d.v}</div>
                      </div>
                    ))}
                  </div>
                  {/* Terminal gauges */}
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>TERMINAL STATE GAUGES</Label>
                    <GaugeBar label="VOLTAGE" value={parseFloat(endState.v||0)} max={16} unit="V" crit={10.5} warn={11.5}/>
                    <GaugeBar label="ENGINE RPM" value={parseFloat(endState.rpm||0)} max={2400} unit="RPM" crit={200} warn={800}/>
                    <GaugeBar label="PUMP EFF" value={parseFloat(endState.pump_eff||0)} max={1} unit="" crit={0.1} warn={0.5}/>
                    <GaugeBar label="ALTITUDE" value={parseFloat(endState.alt_agl||endState.alt||0)} max={16500} unit="ft" crit={500} warn={2000}/>
                  </div>
                </div>

                {/* L3 snapshot */}
                {l3.available && l3.snapshot && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>SWISS CHEESE ANALYSIS · XGBoost · Pre-existing conditions at failure injection</Label>
                    {!l3Reliable && (
                      <div style={{ fontSize:"0.62rem", color:"var(--amber)", fontFamily:"var(--mono)", marginBottom:10,
                        padding:"4px 8px", background:"rgba(245,158,11,0.06)", borderRadius:2, border:"1px solid var(--amber-dim)" }}>
                        ⚠ L3 cascade prediction has reduced accuracy for slow cascades (&gt;300s) due to inference resolution.
                        L2 physics outcome is ground truth. Snapshot (XGBoost) analysis is reliable for all cases.
                      </div>
                    )}
                    <div style={{ display:"flex", alignItems:"center", gap:32, marginBottom:14 }}>
                      <div>
                        <div style={{ fontSize:"0.52rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.15em", marginBottom:2 }}>ML CRASH PROBABILITY</div>
                        <div style={{ fontFamily:"var(--mono)", fontSize:"2rem",
                          color:(l3.snapshot.crash_probability||0)>.5?"var(--red)":"var(--green)" }}>
                          {((l3.snapshot.crash_probability||0)*100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    {l3.snapshot.top_factors && (
                      <>
                        <Label>DRIVING FACTORS (SHAP)</Label>
                        {l3.snapshot.top_factors.slice(0,5).map((f,i)=>{
                          const maxImp = l3.snapshot.top_factors[0]?.importance||1;
                          return (
                            <div key={i} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:5 }}>
                              <span style={{ fontFamily:"var(--mono)", fontSize:"0.68rem", color:"var(--text)", width:185, flexShrink:0 }}>{f.feature}</span>
                              <div style={{ flex:1, height:3, background:"var(--border)", borderRadius:2 }}>
                                <div style={{ height:"100%", width:`${(f.importance/maxImp)*100}%`,
                                  background:i===0?"var(--amber)":"var(--blue)", borderRadius:2 }}/>
                              </div>
                              <span style={{ fontFamily:"var(--mono)", fontSize:"0.62rem", color:"var(--text-dim)", width:50, textAlign:"right" }}>{fmt(f.importance,3)}</span>
                            </div>
                          );
                        })}
                      </>
                    )}
                  </div>
                )}

                {/* L4 interpretation */}
                {l4.found && l4.interpretation && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>LAYER 4 INTERPRETATION · VALIDATED FINDING</Label>
                    <div style={{ fontSize:"0.75rem", fontFamily:"var(--mono)", color:"var(--text)", lineHeight:1.75 }}>
                      {l4.interpretation}
                    </div>
                  </div>
                )}
                {!l4.found && l3.available && (
                  <div style={{ border:"1px solid rgba(245,158,11,0.3)", borderRadius:2, padding:14, background:"rgba(245,158,11,0.04)" }}>
                    <Label>L4 VALIDATION NOT AVAILABLE FOR THIS COMPONENT</Label>
                    <div style={{ fontSize:"0.7rem", fontFamily:"var(--mono)", color:"var(--text-dim)", lineHeight:1.7, marginBottom:8 }}>
                      This component was not included in the training scenario generator. L3 ML analysis (XGBoost + Transformer) is still valid.
                      L4 physics and NTSB validation require the component to have appeared in the 500K training trajectories.
                    </div>
                    <div style={{ fontSize:"0.65rem", color:"var(--amber)", fontFamily:"var(--mono)" }}>
                      L3 crash probability: {((l3.snapshot?.crash_probability||0)*100).toFixed(1)}%
                      {l3.cascade?.cascade_prediction && ` · Cascade: ${l3.cascade.cascade_prediction}`}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── CASCADE ANALYSIS ── */}
            {tab === "cascade" && (
              <div className="fade-in" style={{ display:"flex", flexDirection:"column", gap:18 }}>
                <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:16, background:"var(--bg2)" }}>
                  <CascadeTimeline failureChain={r.failure_chain} duration={r.duration_s}
                    onSelect={c=>setActiveComp(activeComp===c?null:c)} active={activeComp}/>
                </div>

                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                  {/* Cascade transformer */}
                  {l3.available && l3.cascade && (
                    <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                      <Label>CASCADE TRANSFORMER · SEQUENCE PREDICTION</Label>
                      {!l3Reliable && (
                        <div style={{ fontSize:"0.58rem", color:"var(--amber)", fontFamily:"var(--mono)", marginBottom:8,
                          padding:"3px 6px", background:"rgba(245,158,11,0.06)", borderRadius:2 }}>
                          ⚠ Slow cascade — inference resolution mismatch reduces accuracy
                        </div>
                      )}
                      <div style={{ fontFamily:"var(--mono)", fontSize:"0.9rem",
                        color:l3.cascade.cascade_prediction==="CATASTROPHIC"?"var(--red)":"var(--green)", marginBottom:8 }}>
                        {l3.cascade.cascade_prediction || "—"}
                      </div>
                      {l3.cascade.cascade_crash_probability != null && (
                        <div style={{ marginBottom:8 }}>
                          <div style={{ fontSize:"0.58rem", color:"var(--text-dim)", fontFamily:"var(--ui)", letterSpacing:"0.1em", marginBottom:1 }}>CASCADE CRASH PROB</div>
                          <div style={{ fontFamily:"var(--mono)", fontSize:"0.82rem", color:"var(--text)" }}>
                            {((l3.cascade.cascade_crash_probability)*100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                      {l3.cascade.key_cascade_variables?.length > 0 && (
                        <>
                          <Label>KEY CASCADE VARIABLES (ATTENTION)</Label>
                          {l3.cascade.key_cascade_variables.map((v,i)=>(
                            <div key={i} style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                              <span style={{ fontFamily:"var(--mono)", fontSize:"0.7rem", color:"var(--text)" }}>{v.variable}</span>
                              <span style={{ fontFamily:"var(--mono)", fontSize:"0.65rem", color:"var(--amber)" }}>{fmt(v.attention_weight,4)}</span>
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  )}

                  {/* Kill zone */}
                  <KillZoneDisplay killZone={l4.kill_zone} crashProb={l4.crash_probability}/>
                </div>

                {/* Altitude sub-rules */}
                {l4.sub_rules_by_altitude?.length > 0 && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>ALTITUDE-STRATIFIED CRASH PROBABILITY</Label>
                    {l4.sub_rules_by_altitude.map((sr,i)=>(
                      <div key={i} style={{ display:"flex", alignItems:"center", gap:14, marginBottom:8,
                        padding:"6px 10px", background:"var(--bg3)", borderRadius:2, border:"1px solid var(--border)" }}>
                        <span style={{ fontFamily:"var(--mono)", fontSize:"0.65rem", color:"var(--amber)", width:160, flexShrink:0 }}>{sr.altitude_band}</span>
                        <div style={{ flex:1, height:4, background:"var(--border)", borderRadius:2 }}>
                          <div style={{ height:"100%", width:`${(sr.crash_probability||0)*100}%`,
                            background:(sr.crash_probability||0)>.5?"var(--red)":"var(--amber)", borderRadius:2 }}/>
                        </div>
                        <span style={{ fontFamily:"var(--mono)", fontSize:"0.72rem",
                          color:(sr.crash_probability||0)>.5?"var(--red)":"var(--amber)", width:42, textAlign:"right" }}>
                          {((sr.crash_probability||0)*100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── VALIDATION & EVIDENCE ── */}
            {tab === "evidence" && (
              <div className="fade-in" style={{ display:"flex", flexDirection:"column", gap:18 }}>
                {/* Scores */}
                {l4.found ? (
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:14 }}>
                    {[
                      {l:"PHYSICS SCORE",  v:l4.physics_score,              s:"L2 POH+FAA physics simulation"},
                      {l:"NTSB SCORE",     v:l4.ntsb_score,                 s:`n=${l4.n_accidents_searched} Cessna NTSB records`},
                      {l:"COMBINED",       v:l4.combined_confidence?.toFixed(1), s:"Weighted validation score"},
                    ].map(s=>(
                      <div key={s.l} style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)", textAlign:"center" }}>
                        <div style={{ fontSize:"0.52rem", fontFamily:"var(--ui)", letterSpacing:"0.15em", color:"var(--text-dim)", marginBottom:6 }}>{s.l}</div>
                        <div style={{ fontFamily:"var(--mono)", fontSize:"1.8rem",
                          color:s.v>=80?"var(--green)":s.v>=60?"var(--amber)":"var(--red)" }}>{s.v}</div>
                        <div style={{ fontSize:"0.52rem", fontFamily:"var(--mono)", color:"var(--text-dim)", marginTop:4 }}>{s.s}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ border:"1px solid rgba(245,158,11,0.3)", borderRadius:2, padding:16, background:"rgba(245,158,11,0.04)" }}>
                    <Label>L4 VALIDATION NOT AVAILABLE</Label>
                    <div style={{ fontSize:"0.72rem", color:"var(--text-dim)", fontFamily:"var(--mono)", lineHeight:1.7 }}>
                      Component <span style={{color:"var(--amber)"}}>{r.failure_chain?.[0]?.component || "unknown"}</span> has no entry in
                      validated_findings.json. This occurs for components outside the 14-component scenario generator scope
                      (engine_right, hydraulics, landing_gear, etc.). The L3 ML models were trained on the 14 primary failure modes.
                    </div>
                    <div style={{ marginTop:10, fontSize:"0.65rem", color:"var(--amber)", fontFamily:"var(--mono)" }}>
                      L3 crash probability: {((l3.snapshot?.crash_probability||0)*100).toFixed(1)}%
                    </div>
                  </div>
                )}

                {/* Citations with full names */}
                {l4.citations?.length > 0 && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>SOURCE CITATIONS · HOVER FOR FULL DOCUMENT NAME</Label>
                    <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
                      {l4.citations.map((c,i)=><CitationTag key={i} raw={c}/>)}
                    </div>
                  </div>
                )}

                {/* NTSB accidents */}
                {l4.ntsb_matching_accidents?.length > 0 && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>NTSB MATCHING ACCIDENTS · {l4.ntsb_matching_accidents.length} Cessna records</Label>
                    <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginBottom:10 }}>
                      {l4.ntsb_matching_accidents.slice(0,24).map((id,i)=>(
                        <span key={i} style={{ padding:"3px 7px", border:"1px solid var(--border)", borderRadius:2,
                          fontFamily:"var(--mono)", fontSize:"0.62rem", color:"var(--text-dim)", background:"var(--bg3)" }}>
                          {id}
                        </span>
                      ))}
                      {l4.ntsb_matching_accidents.length > 24 && (
                        <span style={{ fontFamily:"var(--mono)", fontSize:"0.62rem", color:"var(--text-dim)", padding:"3px 7px" }}>
                          +{l4.ntsb_matching_accidents.length - 24} more
                        </span>
                      )}
                    </div>
                    {l4.match_quality && (
                      <div style={{ fontSize:"0.62rem", fontFamily:"var(--mono)", color:"var(--text-dim)" }}>
                        match quality: {Object.entries(l4.match_quality).map(([k,v])=>`${k}=${v}`).join(" · ")}
                      </div>
                    )}
                  </div>
                )}

                {/* Induced relationship */}
                {l4.induced_relationship && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>INDUCED RELATIONSHIP PATTERN · ML-DISCOVERED</Label>
                    <div style={{ fontFamily:"var(--mono)", fontSize:"0.78rem", color:"var(--amber)", lineHeight:1.6, marginBottom:10 }}>
                      {l4.induced_relationship}
                    </div>
                    <div style={{ fontSize:"0.6rem", fontFamily:"var(--ui)", letterSpacing:"0.1em" }}>
                      <span style={{ color:"var(--text-dim)" }}>CLASS: </span>
                      <span style={{ color:l4.classification==="INDUCED"?"var(--amber)":"var(--blue)" }}>{l4.classification}</span>
                      <span style={{ color:"var(--text-dim)", margin:"0 10px" }}>·</span>
                      <span style={{ color:l4.validation_status==="VALIDATED"?"var(--green)":"var(--amber)" }}>{l4.validation_status}</span>
                    </div>
                  </div>
                )}

                {/* L1 related rules */}
                {l4.l1_related_rules?.length > 0 && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>RELATED L1 CONSTRAINT RULES (POH GROUND TRUTH)</Label>
                    <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
                      {l4.l1_related_rules.map((r,i)=>(
                        <span key={i} style={{ padding:"3px 8px", border:"1px solid var(--blue)", borderRadius:2,
                          fontFamily:"var(--mono)", fontSize:"0.68rem", color:"var(--blue)", background:"rgba(59,130,246,0.06)" }}>
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Parsed scenario */}
                {data.understood && (
                  <div style={{ border:"1px solid var(--border)", borderRadius:2, padding:14, background:"var(--bg2)" }}>
                    <Label>PARSED SCENARIO · L0 NLP OUTPUT</Label>
                    <pre style={{ fontFamily:"var(--mono)", fontSize:"0.7rem", color:"var(--text)", lineHeight:1.7, whiteSpace:"pre-wrap" }}>
                      {data.understood}
                    </pre>
                    {data.assumptions?.length > 0 && (
                      <div style={{ marginTop:8 }}>
                        {data.assumptions.map((a,i)=>(
                          <div key={i} style={{ fontSize:"0.62rem", fontFamily:"var(--mono)", color:"var(--text-dim)", marginTop:3 }}>⚠ {a}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ── PARAMETER SWEEP ── */}
            {tab === "variations" && (
              <div className="fade-in">
                <MultiSimPanel baseText={input} baseResult={data}/>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── App root ───────────────────────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase] = useState("input");
  const [stage, setStage] = useState("parse");
  const [result, setResult] = useState(null);
  const [input, setInput]   = useState("");
  const [error, setError]   = useState(null);

  async function handleSubmit(text) {
    setInput(text); setPhase("loading"); setStage("parse"); setError(null);
    const seq = ["parse","simulate","analyze","validate"];
    [1,2,3].forEach(i => setTimeout(()=>setStage(seq[i]), i*2300));
    try {
      const res = await fetch(`${API}/api/run_full`, {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text}),
      });
      if (!res.ok) { const e=await res.json().catch(()=>({})); throw new Error(e.detail||`HTTP ${res.status}`); }
      setResult(await res.json()); setPhase("results");
    } catch(e) { setError(e.message); setPhase("error"); }
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{__html:CSS}}/>
      {phase==="input"   && <InputScreen onSubmit={handleSubmit}/>}
      {phase==="loading" && <LoadingScreen stage={stage}/>}
      {phase==="results" && result && <ResultsScreen data={result} input={input} onReset={()=>setPhase("input")}/>}
      {phase==="error"   && (
        <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:40 }}>
          <div style={{ fontFamily:"var(--ui)", fontWeight:700, color:"var(--red)", fontSize:"1.6rem", marginBottom:16 }}>ANALYSIS FAILED</div>
          <div style={{ fontFamily:"var(--mono)", fontSize:"0.78rem", color:"var(--text-dim)", marginBottom:24, maxWidth:500, textAlign:"center" }}>{error}</div>
          <div style={{ fontFamily:"var(--mono)", fontSize:"0.62rem", color:"var(--text-dim)", marginBottom:20 }}>
            Backend: uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level warning
          </div>
          <button onClick={()=>setPhase("input")} style={{ background:"var(--amber)", color:"var(--bg)", border:"none", padding:"10px 28px", borderRadius:2, fontFamily:"var(--ui)", fontWeight:600, fontSize:"0.8rem", cursor:"pointer" }}>
            RETRY
          </button>
        </div>
      )}
    </>
  );
}
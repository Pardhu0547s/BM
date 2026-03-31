import React, { useState } from 'react';
import axios from 'axios';
import { ShieldAlert, Cpu } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Predict() {
  const [formData, setFormData] = useState({
    Age: 55, Height: 165.0, Weight: 65.0, BMI: 23.9,
    ALT: 25.0, AST: 22.0, BUN: 5.5, CREA: 70.0, URIC: 320.0, FBG: 5.2,
    "HDL-C": 1.3, "LDL-C": 2.8, Ca: 2.2, P: 1.1, Mg: 0.85,
    Calsium: 0, Calcitriol: 0, Bisphosphonate: 0, Calcitonin: 0,
    HTN: 0, COPD: 0, DM: 0, Hyperlipidaemia: 0, Hyperuricemia: 0,
    AS: 0, VT: 0, VD: 0, CAD: 0, CKD: 0, Smoking: 0, Drinking: 0
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInput = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.post('http://localhost:8000/api/predict', formData);
      setResult(data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        // FastAPI validation errors are arrays of objects
        setError(detail.map(d => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join(' | '));
      } else {
        setError(detail || err.message);
      }
    }
    setLoading(false);
  };

  return (
    <div className="animate-fade-in">
      <h1 style={{marginBottom: '4px'}}>OsteoScan Prediction</h1>
      <p className="subtext" style={{marginBottom: '32px'}}>Female-Specific Clinical BMD Diagnostic Engine</p>

      {error && (
        <div className="error-banner animate-fade-in">
          <ShieldAlert /> {error}
        </div>
      )}

      <div style={{display: 'flex', gap: '32px', alignItems: 'flex-start'}}>
        
        {/* LEFT COLUMN: FORMS */}
        <div style={{flex: '1'}}>
          <div className="glass-card" style={{marginBottom: '24px'}}>
            <div className="form-section">
              <div className="sec-label">01 • Demographics & Anthropometrics</div>
              <div className="form-grid">
                <div className="form-group"><label>Age (yrs)</label><input type="number" min="0" name="Age" className="form-input" value={formData.Age} onChange={handleInput} /></div>
                <div className="form-group"><label>Height (cm)</label><input type="number" min="0" step="0.1" name="Height" className="form-input" value={formData.Height} onChange={handleInput} /></div>
                <div className="form-group"><label>Weight (kg)</label><input type="number" min="0" step="0.1" name="Weight" className="form-input" value={formData.Weight} onChange={handleInput} /></div>
                <div className="form-group"><label>BMI</label><input type="number" min="0" step="0.1" name="BMI" className="form-input" value={formData.BMI} onChange={handleInput} /></div>
              </div>
            </div>

            <div className="form-section">
              <div className="sec-label">02 • Biochemical Markers</div>
              <div className="form-grid">
                {['ALT','AST','BUN','CREA','URIC','FBG','HDL-C','LDL-C','Ca','P','Mg'].map((k) => (
                  <div className="form-group" key={k}>
                    <label>{k}</label>
                    <input type="number" min="0" step="0.01" name={k} className="form-input" value={formData[k]} onChange={handleInput} />
                  </div>
                ))}
              </div>
            </div>

            <div className="form-section">
              <div className="sec-label">03 • Medications & Supplements</div>
              <div className="form-grid">
                {['Calsium','Calcitriol','Bisphosphonate','Calcitonin'].map((k) => (
                  <div className="form-group" key={k}>
                    <label>{k}</label>
                    <select name={k} className="form-select" value={formData[k]} onChange={handleInput}>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>

            <div className="form-section">
              <div className="sec-label">04 • Comorbidities & Lifestyle</div>
              <div className="form-grid">
                {['HTN','COPD','DM','Hyperlipidaemia','Hyperuricemia','AS','VT','VD','CAD','CKD','Smoking','Drinking'].map((k) => (
                  <div className="form-group" key={k}>
                    <label>{k}</label>
                    <select name={k} className="form-select" value={formData[k]} onChange={handleInput}>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>

            <button className="btn-primary" onClick={handleRun} disabled={loading}>
              <Cpu size={20} /> {loading ? "Computing Pipeline..." : "Run Clinical Prediction"}
            </button>
          </div>
        </div>

        {/* RIGHT COLUMN: RESULTS */}
        <div style={{width: '400px'}}>
          {result ? (
            <div className="glass-card animate-fade-in" style={{position: 'sticky', top: '32px'}}>
              <h2 style={{fontSize: '1.2rem', paddingBottom: '16px', borderBottom: '1px solid var(--border-color)', marginBottom: '24px'}}>Diagnostic Result</h2>
              
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px'}}>
                <div>
                  <div style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Status</div>
                  <div className={`chip ${result.label.toLowerCase()}`} style={{marginTop: '4px', fontSize: '0.9rem', padding: '6px 14px'}}>{result.label}</div>
                </div>
                <div style={{textAlign: 'right'}}>
                  <div style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>L1-4 BMD (g/cm²)</div>
                  <div className="kpi-value" style={{color: result.color, fontSize: '2rem'}}>{result.prediction_bmd.toFixed(3)}</div>
                  <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Risk Gauge: {result.risk_score_pct.toFixed(0)}%</div>
                </div>
              </div>

              {/* Progress Bar Guage */}
              <div style={{width: '100%', height: '12px', background: 'rgba(0,0,0,0.05)', borderRadius: '10px', overflow: 'hidden', marginBottom: '32px'}}>
                <div style={{
                  width: `${result.risk_score_pct}%`,
                  height: '100%',
                  background: `linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%)`,
                  borderRadius: '10px',
                  transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)'
                }}></div>
              </div>

              <h2 style={{fontSize: '1rem', marginBottom: '16px'}}>Feature Attribution (SHAP)</h2>
              <div style={{height: '300px', width: '100%'}}>
                <ResponsiveContainer>
                  <BarChart data={result.shap_waterfall} layout="vertical" margin={{top: 0, right: 30, left: 10, bottom: 0}}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="feature" type="category" width={80} tick={{fontSize: 11, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{fill: 'rgba(0,0,0,0.03)'}} formatter={(v, n, p) => [`${v.toFixed(4)} (input: ${p.payload.value})`, "SHAP Value"]} />
                    <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                      {result.shap_waterfall.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? 'var(--color-green)' : 'var(--color-red)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '12px'}}>
                Base Value: {result.shap_expected_value.toFixed(4)} • Top 10 Drivers
              </div>

            </div>
          ) : (
            <div className="glass-card" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', textAlign: 'center', color: 'var(--text-secondary)'}}>
              <ShieldAlert size={48} style={{opacity: 0.2, marginBottom: '16px'}} />
              <p>Configure clinical inputs<br/>and run prediction to view results.</p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

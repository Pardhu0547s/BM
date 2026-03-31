import React from 'react';
import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { Stethoscope, Activity, FileSpreadsheet } from 'lucide-react';
import Predict from './pages/Predict';
import Evaluation from './pages/Evaluation';

const Sidebar = () => {
  return (
    <nav className="sidebar">
      <div className="brand">
        <div className="brand-icon">🦴</div>
        <div>
          <div className="brand-name">OsteoScan</div>
          <div className="brand-tag">Clinical BMD</div>
        </div>
      </div>

      <div className="nav-section">
        <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '12px', paddingLeft: '12px' }}>Dashboard</div>
        <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          <Stethoscope size={20} />
          <span>Predict</span>
        </NavLink>
        <NavLink to="/evaluation" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          <Activity size={20} />
          <span>Evaluation</span>
        </NavLink>
      </div>

      <div className="model-card">
        <div style={{ fontWeight: 700, marginBottom: '16px', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}><FileSpreadsheet size={16} /> Model Details</div>
        <div className="trow"><span className="trow-label">Algorithm</span><span className="trow-value">GradientBoost</span></div>
        <div className="trow"><span className="trow-label">Dataset</span><span className="trow-value">593 pts (F)</span></div>
        <div className="trow"><span className="trow-label">Features</span><span className="trow-value">31 markers</span></div>
        <div className="trow"><span className="trow-label">Validation</span><span className="trow-value">5-Fold CV</span></div>
        <div className="trow"><span className="trow-label">Target</span><span className="trow-value">L1–4 BMD</span></div>
      </div>
    </nav>
  );
};

export default function App() {
  const loc = useLocation();
  // We use key on Routes so animations work properly on route changes
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Routes key={loc.pathname}>
          <Route path="/" element={<Predict />} />
          <Route path="/evaluation" element={<Evaluation />} />
        </Routes>
      </main>
    </div>
  );
}

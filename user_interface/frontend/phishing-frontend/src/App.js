import { useState } from 'react'; 
import './vendor/Bootstrap/bootstrap.min.css'; 
import { 
  ShieldAlert, Globe, FileText, Cpu, Upload, 
  AlertTriangle, CheckCircle, List, Download, Search, Activity, X 
} from 'lucide-react';
import { detectUrl, uploadCsv, generatePhishing, exportCsv } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [csvData, setCsvData] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [modal, setModal] = useState({ show: false, type: '', title: '', message: '' });

  const notify = (type, title, message) => {
    setModal({ show: true, type, title, message });
  };

  const closeNotify = () => setModal({ ...modal, show: false });

  
  const stats = (() => {
    if (!csvData || csvData.length === 0) return { total: 0, phishing: 0, legitimate: 0 };
    const phishing = csvData.filter(r => r.prediction === 'phishing').length;
    return { total: csvData.length, phishing, legitimate: csvData.length - phishing };
  })();

  const filteredData = csvData?.filter(item => 
    item.original_url?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleDetect = async () => {
    if(!url) {
      notify('warning', 'Input Required', 'Please provide a URL to start the security scan.');
      return;
    }
    setLoading(true);
    try {
      const res = await detectUrl(url);
      setResult(res.data);
      if(res.data.prediction === 'phishing') {
        notify('error', 'Threat Detected', 'Our AI has flagged this URL as highly suspicious!');
      } else {
        notify('success', 'Secure URL', 'The analysis confirms this URL is safe to visit.');
      }
    } catch (err) { 
      notify('error', 'System Offline', 'Backend server is not responding.');
    }
    setLoading(false);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    try {
      const res = await uploadCsv(formData);
    
      const dataReceived = res.data.results || res.data; 
      setCsvData(dataReceived);
      notify('success', 'Analysis Complete', `Processed ${dataReceived.length} URLs successfully.`);
    } catch (err) { 
      notify('error', 'Processing Error', 'Check your CSV format or Backend connection.');
    }
    setLoading(false);
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await generatePhishing();
      setCsvData(res.data);
      setActiveTab('bulk');
      notify('success', 'Samples Ready', 'Generated AI adversarial phishing samples.');
    } catch (err) { 
      notify('error', 'Generator Error', 'Could not start the simulation engine.');
    }
    setLoading(false);
  };

  const handleExport = async () => {
    if(!csvData) return;
    try {
      const res = await exportCsv(csvData);
      const blob = new Blob([res.data], { type: 'text/csv' });
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', 'security_report.csv');
      document.body.appendChild(link);
      link.click();
    } catch (err) { 
      notify('error', 'Export Failed', 'Unable to generate report.');
    }
  };

  return (
    <div className="min-vh-100" style={{ backgroundColor: '#f8fafc', color: '#1e293b' }}>
      
      {/* Modal */}
      {modal.show && (
        <div className="modal-overlay">
          <div className="modal-card animate-pop shadow-lg border-0 rounded-4 overflow-hidden bg-white">
            <div className={`status-strip ${modal.type}`}></div>
            <div className="p-4 text-center position-relative">
              <button onClick={closeNotify} className="btn-close-custom"><X size={20}/></button>
              <div className={`icon-wrapper mb-3 mx-auto ${modal.type}`}>
                {modal.type === 'success' ? <CheckCircle size={40}/> : modal.type === 'error' ? <AlertTriangle size={40}/> : <ShieldAlert size={40}/>}
              </div>
              <h4 className="fw-bold mb-2">{modal.title}</h4>
              <p className="text-muted small mb-4">{modal.message}</p>
              <button onClick={closeNotify} className="btn btn-dark w-100 rounded-pill fw-bold">CONTINUE</button>
            </div>
          </div>
        </div>
      )}

      {/* Navbar */}
      <nav className="navbar navbar-expand-lg navbar-dark sticky-top shadow-sm" style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}>
        <div className="container">
          <span className="navbar-brand d-flex align-items-center gap-2 fw-bold fs-4">
            <ShieldAlert size={28} className="text-warning" />
            <span style={{ letterSpacing: '1px' }}>URL PhishDetector <span className="text-warning"></span></span>
          </span>
          <div className="ms-auto d-flex gap-2">
            <button onClick={() => {setActiveTab('single'); setCsvData(null);}} className={`nav-btn ${activeTab === 'single' ? 'active' : ''}`}><Globe size={18}/> Single Scan</button>
            <button onClick={() => {setActiveTab('bulk'); setCsvData(null);}} className={`nav-btn ${activeTab === 'bulk' ? 'active' : ''}`}><List size={18}/> Bulk Scan</button>
            <button onClick={() => {setActiveTab('gen'); setCsvData(null);}} className={`nav-btn ${activeTab === 'gen' ? 'active' : ''}`}><Cpu size={18}/> Generator</button>
          </div>
        </div>
      </nav>

      <div className="container py-5">
        
        {/* Tab 1: Single Scan */}
        {activeTab === 'single' && (
          <div className="row justify-content-center">
            <div className="col-lg-8 card border-0 shadow-lg rounded-4 p-5 bg-white">
                <div className="text-center mb-4">
                  <h3 className="fw-bold text-dark">URL Analyzer</h3>
                  <p className="text-muted">Real-time analysis for URL</p>
                </div>
                <div className="input-group input-group-lg shadow-sm border rounded-pill overflow-hidden p-1">
                  <input type="text" className="form-control border-0 px-4" placeholder="Enter URL to check safety" value={url} onChange={(e)=>setUrl(e.target.value)} />
                  <button className="btn btn-primary px-5 rounded-pill fw-bold" onClick={handleDetect} disabled={loading}>{loading ? 'SCANNING...' : 'SCAN'}</button>
                </div>

                {result && (
                  <div className={`mt-5 p-4 rounded-4 border-start border-5 shadow-sm ${result.prediction === 'phishing' ? 'result-danger' : 'result-success'}`}>
                    <div className="d-flex align-items-center gap-4">
                      <div className="status-icon-box">{result.prediction === 'phishing' ? <AlertTriangle className="text-danger" size={40}/> : <CheckCircle className="text-success" size={40}/>}</div>
                      <div className="flex-grow-1">
                        <h3 className="fw-bold text-uppercase mb-1">{result.prediction}</h3>
                        <div className="progress rounded-pill mb-1" style={{ height: '8px' }}>
                          <div className={`progress-bar ${result.prediction === 'phishing' ? 'bg-danger' : 'bg-success'}`} style={{ width: `${result.phishing_probability * 100}%` }}></div>
                        </div>
                        <span className="small fw-bold">Risk Level: {Math.round(result.phishing_probability * 100)}%</span>
                      </div>
                    </div>
                  </div>
                )}
            </div>
          </div>
        )}

  {/* Tab 2: Bulk Scan */}
{activeTab === 'bulk' && (
  <div className="row g-4">
    
    {csvData && (
      <div className="col-12">
        <div className="row g-3 mb-2">
          <div className="col-md-4">
            <div className="stat-card blue shadow-sm border-0 rounded-4 p-3 bg-white border-start border-primary border-5">
              <div className="d-flex justify-content-between align-items-center">
                <div className="text-start"><h6>TOTAL ANALYZED</h6><h2 className="fw-bold">{stats.total}</h2></div>
                <List size={32} className="text-primary opacity-25" />
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="stat-card red shadow-sm border-0 rounded-4 p-3 bg-white border-start border-danger border-5">
              <div className="d-flex justify-content-between align-items-center">
                <div className="text-start"><h6>PHISHING THREATS</h6><h2 className="fw-bold text-danger">{stats.phishing}</h2></div>
                <AlertTriangle size={32} className="text-danger opacity-25" />
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="stat-card green shadow-sm border-0 rounded-4 p-3 bg-white border-start border-success border-5">
              <div className="d-flex justify-content-between align-items-center">
                <div className="text-start"><h6>VERIFIED SAFE</h6><h2 className="fw-bold text-success">{stats.legitimate}</h2></div>
                <CheckCircle size={32} className="text-success opacity-25" />
              </div>
            </div>
          </div>
        </div>
      </div>
    )}

    
    <div className="col-12">
      <div className="card shadow-lg border-0 rounded-4 p-4 bg-white">
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h5 className="fw-bold mb-0 d-flex align-items-center gap-2">
            <Activity size={20} className="text-primary"/> Dataset Analysis
          </h5>
          {csvData && (
            <div className="d-flex gap-2">
              <div className="search-box bg-light border rounded-pill px-3 py-1 d-flex align-items-center gap-2">
                <Search size={16} className="text-muted"/>
                <input type="text" className="border-0 bg-transparent outline-none small" placeholder="Filter URLs..." onChange={(e)=>setSearchTerm(e.target.value)}/>
              </div>
              <button onClick={handleExport} className="btn btn-sm btn-success rounded-pill px-3 shadow-sm">
                <Download size={16} className="me-1"/> Export
              </button>
              <button onClick={() => setCsvData(null)} className="btn btn-sm btn-outline-secondary rounded-pill px-3">Clear</button>
            </div>
          )}
        </div>

        {!csvData ? (
          <div className="upload-area py-5 text-center border-2 border-dashed rounded-4 bg-light">
            <Upload size={50} className="text-primary mb-3 mx-auto d-block" />
            <h5 className="fw-bold">Upload CSV Dataset</h5>
            <label className="btn btn-primary rounded-pill px-4 mt-3 cursor-pointer shadow-sm">
              Browse Files
              <input type="file" hidden accept=".csv" onChange={handleFileUpload} />
            </label>
          </div>
        ) : (
          <div className="table-responsive rounded-3 border overflow-hidden">
            <table className="table table-hover align-middle mb-0">
              
              <thead className="table-light border-bottom">
                <tr className="text-uppercase small fw-bold text-secondary">
                  <th className="px-4 py-3 border-0">Website URL</th>
                  <th className="px-4 py-3 border-0 text-center">status</th>
                  <th className="px-4 py-3 border-0" style={{ width: '200px' }}>Risk</th>
                  <th className="px-4 py-3 border-0 text-end">Probability</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((r, i) => (
                  <tr key={i}>
                  
                    <td className="px-4 py-3 font-monospace small text-primary text-truncate" style={{maxWidth: '350px'}}>
                      {r.original_url || r.url}
                    </td>
                    
                    
                    <td className="px-4 py-3 text-center">
                      <span className={`badge rounded-pill px-3 py-2 shadow-sm ${r.prediction === 'phishing' ? 'bg-danger' : 'bg-success'}`}>
                        {r.prediction?.toUpperCase()}
                      </span>
                    </td>

                    
                    <td className="px-4 py-3">
                      <div className="progress rounded-pill bg-light" style={{ height: '8px' }}>
                        <div 
                          className={`progress-bar progress-bar-striped progress-bar-animated ${r.prediction === 'phishing' ? 'bg-danger' : 'bg-success'}`} 
                          style={{ width: `${(r.phishing_probability || 0) * 100}%` }}
                        ></div>
                      </div>
                    </td>

                    
                    <td className="px-4 py-3 text-end fw-bold text-dark">
                      {Math.round((r.phishing_probability || 0) * 100)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  </div>
)}

        {/* Tab 3: Generator */}
        {activeTab === 'gen' && (
          <div className="row justify-content-center py-5">
            <div className="col-lg-7 card border-0 shadow-lg rounded-4 p-5 text-center bg-white">
                <Cpu size={80} className="text-primary mb-4 mx-auto animate-bounce" />
                <h2 className="fw-bold">Adversarial Threat Engine</h2>
                <p className="text-muted mb-5">Generate synthetic phishing samples for model testing</p>
                <button className="btn btn-primary btn-lg rounded-pill px-5 py-3 fw-bold shadow-blue" onClick={handleGenerate} disabled={loading}>{loading ? 'GENERATING...' : 'START SIMULATION'}</button>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(15, 23, 42, 0.7); display: flex; justify-content: center; align-items: center; z-index: 9999; backdrop-filter: blur(8px); }
        .modal-card { width: 90%; max-width: 400px; position: relative; }
        .status-strip { height: 6px; width: 100%; }
        .status-strip.success { background: #10b981; }
        .status-strip.error { background: #ef4444; }
        .status-strip.warning { background: #f59e0b; }
        .icon-wrapper { width: 80px; height: 80px; border-radius: 50%; display: flex; justify-content: center; align-items: center; }
        .icon-wrapper.success { background: #d1fae5; color: #059669; }
        .icon-wrapper.error { background: #fee2e2; color: #dc2626; }
        .icon-wrapper.warning { background: #fef3c7; color: #d97706; }
        .btn-close-custom { position: absolute; top: 15px; right: 15px; background: none; border: none; color: #94a3b8; }
        .animate-pop { animation: pop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
        @keyframes pop { from { transform: scale(0.8); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        
        .nav-btn { background: none; border: none; color: #94a3b8; padding: 8px 20px; border-radius: 50px; font-weight: 500; transition: 0.3s; display: flex; align-items: center; gap: 8px; }
        .nav-btn.active { background: #fff; color: #0f172a; font-weight: 700; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        
        .stat-card { background: #fff; padding: 20px; border-radius: 1rem; border-bottom: 4px solid #ddd; }
        .stat-card.blue { border-color: #3b82f6; }
        .stat-card.red { border-color: #ef4444; }
        .stat-card.green { border-color: #10b981; }
        .stat-card h6 { font-size: 0.7rem; font-weight: 800; color: #64748b; margin-bottom: 5px; }

        .search-box { background: #f1f5f9; border-radius: 50px; padding: 5px 15px; display: flex; align-items: center; gap: 10px; }
        .search-box input { border: none; background: none; outline: none; font-size: 0.85rem; }
        
        .result-danger { background: #fef2f2; border-color: #ef4444 !important; }
        .result-success { background: #f0fdf4; border-color: #10b981 !important; }
        .status-icon-box { background: #fff; padding: 12px; border-radius: 50%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        
        .rounded-4 { border-radius: 1.25rem !important; }
        .shadow-blue { box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3); }
        .cursor-pointer { cursor: pointer; }
        .animate-bounce { animation: bounce 2s infinite; }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
      `}</style>
    </div>
  );
}

export default App;
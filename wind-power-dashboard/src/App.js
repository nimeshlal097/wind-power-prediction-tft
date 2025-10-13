import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Wind, Upload, TrendingUp, Target, Activity, Zap, Download, RefreshCw, ChevronDown, Info } from 'lucide-react';

const WindPowerDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedHorizon, setSelectedHorizon] = useState('12h');
  const [activeMetric, setActiveMetric] = useState('mae');
  const [showInfo, setShowInfo] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);
  const [chartType, setChartType] = useState('line');

  const styles = {
    dashboard: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #1e293b 75%, #0f172a 100%)',
      backgroundSize: '400% 400%',
      animation: 'gradientShift 15s ease infinite',
      color: 'white',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif'
    },
    header: {
      background: 'rgba(15, 23, 42, 0.8)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(59, 130, 246, 0.3)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    },
    headerContent: {
      maxWidth: '1400px',
      margin: '0 auto',
      padding: '1.5rem 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1.5rem'
    },
    headerLeft: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem'
    },
    headerIconContainer: {
      width: '3.5rem',
      height: '3.5rem',
      background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
      borderRadius: '1rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      animation: 'pulse 2s ease-in-out infinite',
      boxShadow: '0 0 30px rgba(59, 130, 246, 0.5)'
    },
    headerIcon: {
      width: '2rem',
      height: '2rem',
      color: 'white'
    },
    headerTitle: {
      fontSize: '1.75rem',
      fontWeight: 'bold',
      margin: 0,
      background: 'linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text'
    },
    headerSubtitle: {
      color: 'rgba(255, 255, 255, 0.6)',
      margin: 0,
      fontSize: '0.9rem',
      fontWeight: '400'
    },
    headerRight: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      flexWrap: 'wrap'
    },
    horizonSelector: {
      display: 'flex',
      gap: '0.5rem',
      background: 'rgba(30, 41, 59, 0.6)',
      padding: '0.375rem',
      borderRadius: '0.75rem',
      border: '1px solid rgba(59, 130, 246, 0.3)',
      backdropFilter: 'blur(10px)'
    },
    horizonButton: {
      padding: '0.625rem 1.25rem',
      borderRadius: '0.5rem',
      border: 'none',
      cursor: 'pointer',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      fontSize: '0.875rem',
      fontWeight: '600',
      background: 'transparent',
      color: 'rgba(255, 255, 255, 0.6)',
      position: 'relative',
      overflow: 'hidden'
    },
    horizonButtonActive: {
      background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
      color: 'white',
      boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)',
      transform: 'translateY(-2px)'
    },
    uploadButton: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
      padding: '0.625rem 1.25rem',
      borderRadius: '0.75rem',
      cursor: 'pointer',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      border: 'none',
      color: 'white',
      fontWeight: '600',
      fontSize: '0.875rem',
      boxShadow: '0 4px 15px rgba(139, 92, 246, 0.3)'
    },
    iconButton: {
      padding: '0.625rem',
      borderRadius: '0.5rem',
      border: '1px solid rgba(59, 130, 246, 0.3)',
      background: 'rgba(30, 41, 59, 0.6)',
      color: 'rgba(255, 255, 255, 0.8)',
      cursor: 'pointer',
      transition: 'all 0.3s',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    },
    fileInput: {
      display: 'none'
    },
    loadingSpinner: {
      width: '1.5rem',
      height: '1.5rem',
      border: '3px solid rgba(59, 130, 246, 0.3)',
      borderTop: '3px solid #3b82f6',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite'
    },
    mainContent: {
      maxWidth: '1400px',
      margin: '0 auto',
      padding: '2rem'
    },
    metricsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '1.5rem',
      marginBottom: '2rem'
    },
    metricCard: {
      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.75rem',
      border: '1px solid rgba(59, 130, 246, 0.2)',
      transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      cursor: 'pointer',
      position: 'relative',
      overflow: 'hidden'
    },
    metricCardHover: {
      transform: 'translateY(-8px) scale(1.02)',
      borderColor: 'rgba(59, 130, 246, 0.5)',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3), 0 0 40px rgba(59, 130, 246, 0.2)'
    },
    metricGlow: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
      opacity: 0,
      transition: 'opacity 0.4s'
    },
    metricContent: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'relative',
      zIndex: 1
    },
    metricTitle: {
      color: 'rgba(255, 255, 255, 0.7)',
      fontSize: '0.875rem',
      fontWeight: '600',
      marginBottom: '0.75rem',
      textTransform: 'uppercase',
      letterSpacing: '0.05em'
    },
    metricValue: {
      fontSize: '2rem',
      fontWeight: 'bold',
      margin: 0,
      lineHeight: 1.2
    },
    metricChange: {
      fontSize: '0.75rem',
      marginTop: '0.5rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.25rem'
    },
    metricIconContainer: {
      width: '3.5rem',
      height: '3.5rem',
      borderRadius: '0.75rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      transition: 'transform 0.4s'
    },
    metricIcon: {
      width: '2rem',
      height: '2rem'
    },
    controlPanel: {
      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid rgba(59, 130, 246, 0.2)',
      marginBottom: '2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem'
    },
    chartTypeSelector: {
      display: 'flex',
      gap: '0.5rem'
    },
    chartTypeButton: {
      padding: '0.5rem 1rem',
      borderRadius: '0.5rem',
      border: '1px solid rgba(59, 130, 246, 0.3)',
      background: 'rgba(30, 41, 59, 0.6)',
      color: 'rgba(255, 255, 255, 0.7)',
      cursor: 'pointer',
      transition: 'all 0.3s',
      fontSize: '0.875rem',
      fontWeight: '500'
    },
    chartTypeButtonActive: {
      background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
      color: 'white',
      borderColor: 'transparent'
    },
    chartsMainRow: {
      display: 'grid',
      gridTemplateColumns: '1.5fr 1fr',
      gap: '1.5rem',
      marginBottom: '1.5rem'
    },
    chartsBottomRow: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '1.5rem',
      marginBottom: '1.5rem'
    },
    chartContainer: {
      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '2rem',
      border: '1px solid rgba(59, 130, 246, 0.2)',
      transition: 'all 0.3s',
      position: 'relative',
      overflow: 'hidden'
    },
    chartHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '1.5rem'
    },
    chartTitle: {
      fontSize: '1.25rem',
      fontWeight: '600',
      margin: 0,
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem'
    },
    infoButton: {
      padding: '0.375rem',
      borderRadius: '0.375rem',
      border: '1px solid rgba(59, 130, 246, 0.3)',
      background: 'rgba(30, 41, 59, 0.6)',
      color: 'rgba(255, 255, 255, 0.7)',
      cursor: 'pointer',
      transition: 'all 0.3s',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    },
    dataSummary: {
      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '2rem',
      border: '1px solid rgba(59, 130, 246, 0.2)'
    },
    summaryGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1.5rem'
    },
    summaryItem: {
      padding: '1rem',
      borderRadius: '0.75rem',
      background: 'rgba(15, 23, 42, 0.5)',
      border: '1px solid rgba(59, 130, 246, 0.1)',
      transition: 'all 0.3s'
    },
    summaryLabel: {
      color: 'rgba(255, 255, 255, 0.6)',
      marginBottom: '0.5rem',
      fontSize: '0.875rem',
      fontWeight: '500'
    },
    summaryValue: {
      color: 'white',
      fontWeight: '700',
      fontSize: '1.25rem',
      margin: 0
    }
  };

  const colors = {
    red: '#ef4444',
    orange: '#f97316',
    purple: '#a855f7',
    green: '#22c55e',
    blue: '#3b82f6',
    cyan: '#06b6d4'
  };

  useEffect(() => {
    const loadDataForHorizon = async (horizon) => {
      setLoading(true);
      try {
        const filename = `predictions_${horizon}.json`;
        const fileData = await window.fs.readFile(filename, { encoding: 'utf8' });
        const jsonData = JSON.parse(fileData);
        setData(jsonData);
      } catch (error) {
        console.error(`Error loading ${horizon} data:`, error);
        const initialData = {
          "predictions": [
            0.3695147931575775, 0.3569389879703522, 0.34646502137184143, 0.31929484009742737,
            0.33955103158950806, 0.34531086683273315, 0.3115798532962799, 0.2989242970943451,
            0.2260715812444687, 0.17876148223876953, 0.13795757293701172, 0.11656713485717773,
            0.11286164820194244, 0.11419560015201569, 0.1115637868642807, 0.10826325416564941,
            0.09884662926197052, 0.1085803210735321, 0.12813928723335266, 0.14936494827270508,
            0.20025832951068878, 0.196164071559906, 0.1912350356578827, 0.17193345725536346,
            0.1662893295288086, 0.174473375082016, 0.17265647649765015, 0.17762842774391174,
            0.17813923954963684, 0.166751891374588, 0.19212855398654938, 0.23582246899604797,
            0.24037688970565796, 0.2874486744403839, 0.3003947138786316, 0.26760539412498474,
            0.2170484960079193, 0.2023586481809616, 0.2066260278224945, 0.2347101867198944,
            0.26180779933929443, 0.2863529920578003, 0.298239141702652, 0.37402093410491943,
            0.38466310501098633, 0.3293027877807617, 0.4479927122592926, 0.5565035343170166
          ],
          "actuals": [
            0.29574286937713623, 0.37186354398727417, 0.5173385739326477, 0.4810638129711151,
            0.3723334074020386, 0.31115496158599854, 0.301193505525589, 0.27290669083595276,
            0.17573535442352295, 0.1356075555086136, 0.06042665243148804, 0.07687247544527054,
            0.06521943211555481, 0.042289260774850845, 0.11502677947282791, 0.042477209120988846,
            0.10779061913490295, 0.13034488260746002, 0.04304106533527374, 0.034301288425922394,
            0.10440748184919357, 0.11483883112668991, 0.16257870197296143, 0.09623155742883682,
            0.09623155742883682, 0.09623155742883682, 0.09623155742883682, 0.09623155742883682,
            0.09623155742883682, 0.09623155742883682, 0.09623155742883682, 0.09623155742883682,
            0.09623155742883682, 0.09623155742883682, 0.09623155742883682, 0.09623155742883682,
            0.09623155742883682, 0.09623155742883682, 0.09623155742883682, 0.09623155742883682,
            0.09623155742883682, 0.09623155742883682, 0.09623155742883682, 0.09623155742883682,
            0.09623155742883682, 0.09623155742883682, 0.09623155742883682, 0.09623155742883682
          ],
          "feature_importance": {
            "V10": 5.4041900634765625,
            "V100": 18.21274757385254,
            "U100": 27.504653930664062,
            "U10": 48.8784065246582
          },
          "metrics": {
            "mae": 0.11429966241121292,
            "rmse": 0.14686461049008653,
            "mse": 0.021569213814404836,
            "r2": -0.8083286285400391
          }
        };
        setData(initialData);
      }
      setLoading(false);
    };

    loadDataForHorizon(selectedHorizon);
  }, [selectedHorizon]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    try {
      const text = await file.text();
      const jsonData = JSON.parse(text);
      setData(jsonData);
    } catch (error) {
      alert('Error parsing JSON file: ' + error.message);
    }
    setLoading(false);
  };

  const exportData = () => {
    if (!data) return;
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `wind_power_data_${selectedHorizon}.json`;
    link.click();
  };

  if (!data) {
    return (
      <div style={{...styles.dashboard, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem'}}>
        <div style={styles.loadingSpinner} />
        <div style={{fontSize: '1.25rem', color: 'rgba(255, 255, 255, 0.8)'}}>Loading dashboard...</div>
      </div>
    );
  }

  const chartData = data.predictions.map((pred, index) => ({
    time: index + 1,
    prediction: pred,
    actual: data.actuals[index],
    error: Math.abs(pred - data.actuals[index])
  }));

  const featureData = Object.entries(data.feature_importance).map(([feature, importance]) => ({
    feature,
    importance,
    percentage: (importance / Object.values(data.feature_importance).reduce((a, b) => a + b, 0) * 100)
  }));

  const pieColors = ['#3b82f6', '#06b6d4', '#22c55e', '#f59e0b'];

  const MetricCard = ({ title, value, icon: Icon, color, metric }) => (
    <div
      style={{
        ...styles.metricCard,
        ...(hoveredCard === metric ? styles.metricCardHover : {})
      }}
      onMouseEnter={() => setHoveredCard(metric)}
      onMouseLeave={() => setHoveredCard(null)}
      onClick={() => setActiveMetric(metric)}
    >
      <div style={{...styles.metricGlow, opacity: hoveredCard === metric ? 1 : 0}} />
      <div style={styles.metricContent}>
        <div>
          <p style={styles.metricTitle}>{title}</p>
          <p style={{...styles.metricValue, color: colors[color]}}>
            {typeof value === 'number' ? value.toFixed(4) : value}
          </p>
          <div style={{...styles.metricChange, color: colors[color]}}>
            <TrendingUp size={12} />
            <span>Prediction Horizon: {selectedHorizon}</span>
          </div>
        </div>
        <div style={{...styles.metricIconContainer, background: `linear-gradient(135deg, ${colors[color]}33, ${colors[color]}11)`, transform: hoveredCard === metric ? 'scale(1.1) rotate(5deg)' : 'scale(1)'}}>
          <Icon style={{...styles.metricIcon, color: colors[color]}} />
        </div>
      </div>
    </div>
  );

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(15, 23, 42, 0.95)',
          border: '1px solid rgba(59, 130, 246, 0.5)',
          borderRadius: '0.5rem',
          padding: '0.75rem',
          backdropFilter: 'blur(10px)'
        }}>
          {payload.map((entry, index) => (
            <div key={index} style={{color: entry.color, fontSize: '0.875rem', marginBottom: '0.25rem'}}>
              <strong>{entry.name}:</strong> {entry.value.toFixed(4)}
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div style={styles.dashboard}>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .upload-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
        }
        .horizon-button:hover {
          background: rgba(59, 130, 246, 0.2) !important;
          color: white !important;
        }
        .icon-button:hover {
          background: rgba(59, 130, 246, 0.3) !important;
          transform: scale(1.05);
          border-color: rgba(59, 130, 246, 0.5) !important;
        }
        .chart-container:hover {
          border-color: rgba(59, 130, 246, 0.4) !important;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        }
        .summary-item:hover {
          background: rgba(30, 41, 59, 0.8) !important;
          transform: translateY(-2px);
          border-color: rgba(59, 130, 246, 0.3) !important;
        }
        @media (max-width: 1024px) {
          .charts-main-row { grid-template-columns: 1fr !important; }
          .charts-bottom-row { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 768px) {
          .header-content { flex-direction: column !important; }
          .metrics-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
      
      <div style={styles.header}>
        <div style={styles.headerContent} className="header-content">
          <div style={styles.headerLeft}>
            <div style={styles.headerIconContainer}>
              <Wind style={styles.headerIcon} />
            </div>
            <div>
              <h1 style={styles.headerTitle}>Wind Power TFT Dashboard</h1>
              <p style={styles.headerSubtitle}>Advanced Temporal Fusion Transformer Analytics</p>
            </div>
          </div>
          <div style={styles.headerRight}>
            <div style={styles.horizonSelector}>
              {['12h', '24h', '36h', '48h'].map((horizon) => (
                <button
                  key={horizon}
                  className="horizon-button"
                  onClick={() => setSelectedHorizon(horizon)}
                  style={{
                    ...styles.horizonButton,
                    ...(selectedHorizon === horizon ? styles.horizonButtonActive : {})
                  }}
                >
                  {horizon}
                </button>
              ))}
            </div>
            <button
              className="icon-button"
              style={styles.iconButton}
              onClick={() => setSelectedHorizon(selectedHorizon)}
              title="Refresh"
            >
              <RefreshCw size={16} />
            </button>
            <button
              className="icon-button"
              style={styles.iconButton}
              onClick={exportData}
              title="Export Data"
            >
              <Download size={16} />
            </button>
            <label style={styles.uploadButton} className="upload-button">
              <Upload size={16} />
              <span>Upload</span>
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                style={styles.fileInput}
              />
            </label>
            {loading && <div style={styles.loadingSpinner} />}
          </div>
        </div>
      </div>

      <div style={styles.mainContent}>
        <div style={styles.metricsGrid} className="metrics-grid">
          <MetricCard
            title="Mean Absolute Error"
            value={data.metrics.mae}
            icon={Target}
            color="red"
            metric="mae"
          />
          <MetricCard
            title="Root Mean Square Error"
            value={data.metrics.rmse}
            icon={TrendingUp}
            color="orange"
            metric="rmse"
          />
          <MetricCard
            title="R² Score"
            value={data.metrics.r2}
            icon={Activity}
            color="purple"
            metric="r2"
          />
          <MetricCard
            title="Mean Square Error"
            value={data.metrics.mse}
            icon={Zap}
            color="green"
            metric="mse"
          />
        </div>

        <div style={styles.controlPanel}>
          <div>
            <span style={{color: 'rgba(255, 255, 255, 0.7)', fontSize: '0.875rem', marginRight: '1rem'}}>Chart Type:</span>
            <div style={styles.chartTypeSelector}>
              {['line', 'area'].map((type) => (
                <button
                  key={type}
                  onClick={() => setChartType(type)}
                  style={{
                    ...styles.chartTypeButton,
                    ...(chartType === type ? styles.chartTypeButtonActive : {})
                  }}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div style={{color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.875rem'}}>
            Active Horizon: <strong style={{color: colors.blue}}>{selectedHorizon}</strong> | 
            Data Points: <strong style={{color: colors.cyan}}>{data.predictions.length}</strong>
          </div>
        </div>

        <div style={styles.chartsMainRow} className="charts-main-row">
          <div style={styles.chartContainer} className="chart-container">
            <div style={styles.chartHeader}>
              <h3 style={styles.chartTitle}>
                <Activity size={20} />
                Wind Power: Predictions vs Actuals
              </h3>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              {chartType === 'line' ? (
                <LineChart data={chartData}>
                  <defs>
                    <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                    </linearGradient>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="time" 
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                    label={{ value: 'Time Steps', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.6)' }}
                  />
                  <YAxis 
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                    label={{ value: 'Power Output', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.6)' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Line 
                    type="monotone" 
                    dataKey="prediction" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    name="Predicted Power"
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#22c55e" 
                    strokeWidth={3}
                    name="Actual Power"
                    dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                  />
                </LineChart>
              ) : (
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                    </linearGradient>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="time" 
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                  />
                  <YAxis 
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Area 
                    type="monotone" 
                    dataKey="prediction" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    fill="url(#colorPred)"
                    name="Predicted Power"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#22c55e" 
                    strokeWidth={2}
                    fill="url(#colorActual)"
                    name="Actual Power"
                  />
                </AreaChart>
              )}
            </ResponsiveContainer>
          </div>

          <div style={styles.chartContainer} className="chart-container">
            <div style={styles.chartHeader}>
              <h3 style={styles.chartTitle}>
                <Target size={20} />
                Feature Importance
              </h3>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <PieChart>
                <Pie
                  data={featureData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({feature, percentage}) => `${feature}\n${percentage.toFixed(1)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="importance"
                >
                  {featureData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={pieColors[index % pieColors.length]}
                      stroke="rgba(255,255,255,0.2)"
                      strokeWidth={2}
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={styles.chartsBottomRow} className="charts-bottom-row">
          <div style={styles.chartContainer} className="chart-container">
            <div style={styles.chartHeader}>
              <h3 style={styles.chartTitle}>
                <TrendingUp size={20} />
                Prediction Error Over Time
              </h3>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData}>
                <defs>
                  <linearGradient id="colorError" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.9}/>
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0.4}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="time" 
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="error" 
                  fill="url(#colorError)" 
                  name="Absolute Error"
                  radius={[8, 8, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={styles.chartContainer} className="chart-container">
            <div style={styles.chartHeader}>
              <h3 style={styles.chartTitle}>
                <Zap size={20} />
                Feature Importance Values
              </h3>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={featureData} layout="vertical">
                <defs>
                  <linearGradient id="colorBar" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.9}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.7}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  type="number"
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                />
                <YAxis 
                  type="category"
                  dataKey="feature"
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12, fontWeight: 600 }}
                  width={80}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="importance" 
                  fill="url(#colorBar)" 
                  name="Importance Score"
                  radius={[0, 8, 8, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={styles.dataSummary}>
          <div style={styles.chartHeader}>
            <h3 style={styles.chartTitle}>
              <Info size={20} />
              Data Summary & Statistics
            </h3>
          </div>
          <div style={styles.summaryGrid}>
            <div style={styles.summaryItem} className="summary-item">
              <p style={styles.summaryLabel}>Total Data Points</p>
              <p style={styles.summaryValue}>{data.predictions.length}</p>
            </div>
            <div style={styles.summaryItem} className="summary-item">
              <p style={styles.summaryLabel}>Avg Predicted Power</p>
              <p style={styles.summaryValue}>
                {(data.predictions.reduce((a, b) => a + b, 0) / data.predictions.length).toFixed(4)}
              </p>
            </div>
            <div style={styles.summaryItem} className="summary-item">
              <p style={styles.summaryLabel}>Avg Actual Power</p>
              <p style={styles.summaryValue}>
                {(data.actuals.reduce((a, b) => a + b, 0) / data.actuals.length).toFixed(4)}
              </p>
            </div>
            <div style={styles.summaryItem} className="summary-item">
              <p style={styles.summaryLabel}>Max Prediction</p>
              <p style={styles.summaryValue}>
                {Math.max(...data.predictions).toFixed(4)}
              </p>
            </div>
            <div style={styles.summaryItem} className="summary-item">
              <p style={styles.summaryLabel}>Min Prediction</p>
              <p style={styles.summaryValue}>
                {Math.min(...data.predictions).toFixed(4)}
              </p>
            </div>
            <div style={styles.summaryItem} className="summary-item">
              <p style={styles.summaryLabel}>Prediction Horizon</p>
              <p style={{...styles.summaryValue, color: colors.blue}}>{selectedHorizon}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WindPowerDashboard;
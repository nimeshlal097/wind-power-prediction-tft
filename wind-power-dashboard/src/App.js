import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Wind, Upload, TrendingUp, Target, Activity, Zap } from 'lucide-react';

const WindPowerDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Styles object
  const styles = {
    dashboard: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1e3a8a 0%, #475569 50%, #1e3a8a 100%)',
      color: 'white',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif'
    },
    header: {
      background: 'rgba(255, 255, 255, 0.1)',
      backdropFilter: 'blur(10px)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.2)'
    },
    headerContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '1rem 1.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem'
    },
    headerLeft: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem'
    },
    headerIcon: {
      width: '2rem',
      height: '2rem',
      color: '#60a5fa'
    },
    headerTitle: {
      fontSize: '1.5rem',
      fontWeight: 'bold',
      margin: 0
    },
    headerSubtitle: {
      color: 'rgba(255, 255, 255, 0.7)',
      margin: 0,
      fontSize: '0.9rem'
    },
    headerRight: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem'
    },
    uploadButton: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      background: '#2563eb',
      padding: '0.5rem 1rem',
      borderRadius: '0.5rem',
      cursor: 'pointer',
      transition: 'background-color 0.3s',
      border: 'none',
      color: 'white'
    },
    fileInput: {
      display: 'none'
    },
    loadingSpinner: {
      width: '1.5rem',
      height: '1.5rem',
      border: '2px solid #60a5fa',
      borderTop: '2px solid transparent',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite'
    },
    mainContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem 1.5rem'
    },
    metricsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '1.5rem',
      marginBottom: '2rem'
    },
    metricCard: {
      background: 'rgba(255, 255, 255, 0.1)',
      backdropFilter: 'blur(10px)',
      borderRadius: '0.75rem',
      padding: '1.5rem',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      transition: 'all 0.3s ease'
    },
    metricContent: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    },
    metricTitle: {
      color: 'rgba(255, 255, 255, 0.7)',
      fontSize: '0.875rem',
      fontWeight: '500',
      marginBottom: '0.5rem'
    },
    metricValue: {
      fontSize: '1.5rem',
      fontWeight: 'bold',
      margin: 0
    },
    metricIcon: {
      width: '2rem',
      height: '2rem'
    },
    chartsMainRow: {
      display: 'grid',
      gridTemplateColumns: '2fr 1fr',
      gap: '1.5rem',
      marginBottom: '2rem'
    },
    chartsBottomRow: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '1.5rem',
      marginBottom: '2rem'
    },
    chartContainer: {
      background: 'rgba(255, 255, 255, 0.1)',
      backdropFilter: 'blur(10px)',
      borderRadius: '0.75rem',
      padding: '1.5rem',
      border: '1px solid rgba(255, 255, 255, 0.2)'
    },
    chartTitle: {
      fontSize: '1.25rem',
      fontWeight: '600',
      marginBottom: '1rem',
      color: 'white'
    },
    dataSummary: {
      background: 'rgba(255, 255, 255, 0.1)',
      backdropFilter: 'blur(10px)',
      borderRadius: '0.75rem',
      padding: '1.5rem',
      border: '1px solid rgba(255, 255, 255, 0.2)'
    },
    summaryGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem'
    },
    summaryLabel: {
      color: 'rgba(255, 255, 255, 0.7)',
      marginBottom: '0.5rem',
      fontSize: '0.875rem'
    },
    summaryValue: {
      color: 'white',
      fontWeight: '600',
      margin: 0
    }
  };

  const colors = {
    red: '#f87171',
    orange: '#fb923c',
    purple: '#a78bfa',
    green: '#34d399'
  };

  // Load initial data from the provided JSON
  useEffect(() => {
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
  }, []);

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

  if (!data) {
    return (
      <div style={{...styles.dashboard, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
        <div style={{fontSize: '1.25rem'}}>Loading dashboard...</div>
      </div>
    );
  }

  // Prepare chart data
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

  const pieColors = ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b'];

  const MetricCard = ({ title, value, icon: Icon, color, unit = '' }) => (
    <div style={styles.metricCard}>
      <div style={styles.metricContent}>
        <div>
          <p style={styles.metricTitle}>{title}</p>
          <p style={{...styles.metricValue, color: colors[color]}}>
            {typeof value === 'number' ? value.toFixed(4) : value}{unit}
          </p>
        </div>
        <Icon style={{...styles.metricIcon, color: colors[color]}} />
      </div>
    </div>
  );

  return (
    <div style={styles.dashboard}>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .loading-spinner {
          animation: spin 1s linear infinite;
        }
        .upload-button:hover {
          background: #1d4ed8 !important;
        }
        .metric-card:hover {
          background: rgba(255, 255, 255, 0.2) !important;
          transform: translateY(-2px);
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
      
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerContent} className="header-content">
          <div style={styles.headerLeft}>
            <Wind style={styles.headerIcon} />
            <div>
              <h1 style={styles.headerTitle}>Wind Power TFT Dashboard</h1>
              <p style={styles.headerSubtitle}>Temporal Fusion Transformer Analytics</p>
            </div>
          </div>
          <div style={styles.headerRight}>
            <label style={styles.uploadButton} className="upload-button">
              <Upload style={{width: '1rem', height: '1rem'}} />
              <span>Upload JSON</span>
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                style={styles.fileInput}
              />
            </label>
            {loading && <div style={styles.loadingSpinner} className="loading-spinner" />}
          </div>
        </div>
      </div>

      <div style={styles.mainContent}>
        {/* Metrics Cards */}
        <div style={styles.metricsGrid} className="metrics-grid">
          <MetricCard
            title="Mean Absolute Error"
            value={data.metrics.mae}
            icon={Target}
            color="red"
          />
          <MetricCard
            title="Root Mean Square Error"
            value={data.metrics.rmse}
            icon={TrendingUp}
            color="orange"
          />
          <MetricCard
            title="R² Score"
            value={data.metrics.r2}
            icon={Activity}
            color="purple"
          />
          <MetricCard
            title="Mean Square Error"
            value={data.metrics.mse}
            icon={Zap}
            color="green"
          />
        </div>

        {/* Main Charts Row */}
        <div style={styles.chartsMainRow} className="charts-main-row">
          {/* Predictions vs Actuals Chart */}
          <div style={styles.chartContainer}>
            <h3 style={styles.chartTitle}>Wind Power: Predictions vs Actuals</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="time" 
                  stroke="rgba(255,255,255,0.7)"
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)"
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px',
                    color: 'white'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="prediction" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Predicted Power"
                  dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="Actual Power"
                  dot={{ fill: '#10b981', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Feature Importance Pie Chart */}
          <div style={styles.chartContainer}>
            <h3 style={styles.chartTitle}>Feature Importance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={featureData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="importance"
                  label={({feature, percentage}) => `${feature} (${percentage.toFixed(1)}%)`}
                >
                  {featureData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={pieColors[index % pieColors.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px',
                    color: 'white'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bottom Row */}
        <div style={styles.chartsBottomRow} className="charts-bottom-row">
          {/* Error Distribution */}
          <div style={styles.chartContainer}>
            <h3 style={styles.chartTitle}>Prediction Error Over Time</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="time" 
                  stroke="rgba(255,255,255,0.7)"
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)"
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px',
                    color: 'white'
                  }}
                />
                <Bar dataKey="error" fill="#f59e0b" name="Absolute Error" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Feature Importance Bar Chart */}
          <div style={styles.chartContainer}>
            <h3 style={styles.chartTitle}>Feature Importance Values</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={featureData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  type="number"
                  stroke="rgba(255,255,255,0.7)"
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                />
                <YAxis 
                  type="category"
                  dataKey="feature"
                  stroke="rgba(255,255,255,0.7)"
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px',
                    color: 'white'
                  }}
                />
                <Bar dataKey="importance" fill="#06b6d4" name="Importance Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Data Summary */}
        <div style={styles.dataSummary}>
          <h3 style={styles.chartTitle}>Data Summary</h3>
          <div style={styles.summaryGrid}>
            <div>
              <p style={styles.summaryLabel}>Total Data Points:</p>
              <p style={styles.summaryValue}>{data.predictions.length}</p>
            </div>
            <div>
              <p style={styles.summaryLabel}>Average Predicted Power:</p>
              <p style={styles.summaryValue}>
                {(data.predictions.reduce((a, b) => a + b, 0) / data.predictions.length).toFixed(4)}
              </p>
            </div>
            <div>
              <p style={styles.summaryLabel}>Average Actual Power:</p>
              <p style={styles.summaryValue}>
                {(data.actuals.reduce((a, b) => a + b, 0) / data.actuals.length).toFixed(4)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WindPowerDashboard;
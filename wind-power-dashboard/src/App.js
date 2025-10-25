import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';
import { Wind, TrendingUp, Target, Activity, Zap, Info } from 'lucide-react';

const WindPowerDashboard = () => {
  const [hoveredCard, setHoveredCard] = useState(null);
  const [activeMetric, setActiveMetric] = useState('mae');
  const [chartType, setChartType] = useState('line');
  const [horizon, setHorizon] = useState('12h');

  // 12h prediction data
  const data12h = {
    "predictions": [
      0.6912519931793213,
  0.6626559495925903,
  0.6634369492530823,
  0.481275737285614,
  0.413384348154068,
  0.4181964695453644,
  0.30141928791999817,
  0.23546847701072693,
  0.2303375005722046,
  0.19868440926074982,
  0.16447556018829346,
  0.17340947687625885
    ],
    "actuals": [
      0.7116811871528625,
  0.5548350811004639,
  0.4498637318611145,
  0.5984399914741516,
  0.4386805593967438,
  0.2999717891216278,
  0.007139247842133045,
  0.03335271403193474,
  0.06389789283275604,
  0.035608142614364624,
  0.0,
  0.015964195132255554
    ],
    "feature_importance": {
      "V100": 14.416508674621582,
        "U100": 17.477462768554688,
        "V10": 27.72646141052246,
        "U10": 40.37956619262695
    },
    "metrics": {
      "mae": 0.13179951906204224,
        "mse": 0.033692056494167434,
        "rmse": 0.18355396071501,
        "r2": 0.6679900884628296
    }
  };

  // 24h prediction data
  const data24h = {
    "predictions": [
      0.6899523735046387,
      0.7357529997825623,
      0.7669937014579773,
      0.37692609429359436,
      0.2942606508731842,
      0.2868696451187134,
      0.21890410780906677,
      0.18667230010032654,
      0.1875278651714325,
      0.17056694626808167,
      0.12232732772827148,
      0.1295154094696045,
      0.17955313622951508,
      0.1955268383026123,
      0.10764575004577637,
      0.17592453956604004,
      0.27757835388183594,
      0.32385876774787903,
      0.37039369344711304,
      0.4662932753562927,
      0.6966075897216797,
      0.8472678661346436,
      0.8969017267227173,
      0.8065758943557739
    ],
    "actuals": [
      0.7116811871528625,
      0.5548350811004639,
      0.4498637318611145,
      0.5984399914741516,
      0.4386805593967438,
      0.2999717891216278,
      0.007139247842133045,
      0.03335271403193474,
      0.06389789283275604,
      0.035608142614364624,
      0.0,
      0.015964195132255554,
      0.0,
      0.0,
      0.0,
      0.3646244406700134,
      0.6539798974990845,
      0.5515459179878235,
      0.832910418510437,
      0.9739686250686646,
      0.976599931716919,
      0.9756601452827454,
      0.9413588643074036,
      0.9690818190574646
    ],
    "feature_importance": {
      "V10": 9.979222297668457,
      "U10": 13.241190910339355,
      "V100": 19.54136848449707,
      "U100": 57.238216400146484
    },
    "metrics": {
      "mae": 0.15163667500019073,
      "mse": 0.04494978934566169,
      "rmse": 0.21201365367744995,
      "r2": 0.5571480989456177
    }
  };

  // 36h prediction data
  const data36h = {
    "predictions": [
      0.7721378803253174,
      0.6834080219268799,
      0.681358814239502,
      0.47638440132141113,
      0.44748052954673767,
      0.44008713960647583,
      0.32815825939178467,
      0.2573516368865967,
      0.28271621465682983,
      0.23246291279792786,
      0.18952693045139313,
      0.21425274014472961,
      0.22603914141654968,
      0.2293836772441864,
      0.14268359541893005,
      0.20074525475502014,
      0.2815742492675781,
      0.303111732006073,
      0.31538331508636475,
      0.32104480266571045,
      0.36019420623779297,
      0.4660554528236389,
      0.5583506226539612,
      0.6008742451667786,
      0.5880221128463745,
      0.7595593929290771,
      0.81784987449646,
      0.8492470383644104,
      0.8188894987106323,
      0.7539854645729065,
      0.6622645854949951,
      0.5862269401550293,
      0.47471389174461365,
      0.3800160884857178,
      0.412802517414093,
      0.3023299276828766
    ],
    "actuals": [
      0.7116811871528625,
      0.5548350811004639,
      0.4498637318611145,
      0.5984399914741516,
      0.4386805593967438,
      0.2999717891216278,
      0.007139247842133045,
      0.03335271403193474,
      0.06389789283275604,
      0.035608142614364624,
      0.0,
      0.015964195132255554,
      0.0,
      0.0,
      0.0,
      0.3646244406700134,
      0.6539798974990845,
      0.5515459179878235,
      0.832910418510437,
      0.9739686250686646,
      0.976599931716919,
      0.9756601452827454,
      0.9413588643074036,
      0.9690818190574646,
      0.8817780017852783,
      0.86843341588974,
      0.9618456363677979,
      0.6835823655128479,
      0.5897002220153809,
      0.7412837147712708,
      0.6453340649604797,
      0.4965698719024658,
      0.7035992741584778,
      0.53058922290802,
      0.19659806787967682,
      0.07978573441505432
    ],
    "feature_importance": {
      "V10": 20.479034423828125,
      "V100": 23.473243713378906,
      "U100": 24.208791732788086,
      "U10": 31.838932037353516
    },
    "metrics": {
      "mae": 0.16314329206943512,
      "mse": 0.050677629311305226,
      "rmse": 0.22511692364481448,
      "r2": 0.5005544424057007
    }
  };

  // 48h prediction data
  const data48h = {
    "predictions": [
        0.7238050699234009,
        0.7115650177001953,
        0.6879153251647949,
        0.46189531683921814,
        0.43045732378959656,
        0.4246101379394531,
        0.22542525827884674,
        0.17221927642822266,
        0.19846780598163605,
        0.18005108833312988,
        0.15003201365470886,
        0.20829108357429504,
        0.21706488728523254,
        0.20415063202381134,
        0.08517131209373474,
        0.1319483518600464,
        0.2159515917301178,
        0.20129811763763428,
        0.2501648962497711,
        0.31073200702667236,
        0.43611612915992737,
        0.5708209276199341,
        0.6683470010757446,
        0.6799358129501343,
        0.7138911485671997,
        0.8252042531967163,
        0.8760359883308411,
        0.8864554166793823,
        0.8644571304321289,
        0.8119674921035767,
        0.7747354507446289,
        0.7521822452545166,
        0.6957850456237793,
        0.6147598624229431,
        0.677780270576477,
        0.5381155610084534,
        0.4793660640716553,
        0.4018334150314331,
        0.3677949011325836,
        0.4023028612136841,
        0.364796906709671,
        0.4649554193019867,
        0.39526239037513733,
        0.3631384074687958,
        0.35952308773994446,
        0.3362247347831726,
        0.3440941572189331,
        0.32404226064682007
    ],
    "actuals": [
        0.7116811871528625,
        0.5548350811004639,
        0.4498637318611145,
        0.5984399914741516,
        0.4386805593967438,
        0.2999717891216278,
        0.007139247842133045,
        0.03335271403193474,
        0.06389789283275604,
        0.035608142614364624,
        0.0,
        0.015964195132255554,
        0.0,
        0.0,
        0.0,
        0.3646244406700134,
        0.6539798974990845,
        0.5515459179878235,
        0.832910418510437,
        0.9739686250686646,
        0.976599931716919,
        0.9756601452827454,
        0.9413588643074036,
        0.9690818190574646,
        0.8817780017852783,
        0.86843341588974,
        0.9618456363677979,
        0.6835823655128479,
        0.5897002220153809,
        0.7412837147712708,
        0.6453340649604797,
        0.4965698719024658,
        0.7035992741584778,
        0.53058922290802,
        0.19659806787967682,
        0.07978573441505432,
        0.14773046970367432,
        0.2528897523880005,
        0.2512921690940857,
        0.3477116823196411,
        0.37346112728118896,
        0.47899630665779114,
        0.4383986294269562,
        0.3805093467235565,
        0.23663188517093658,
        0.11483883112668991,
        0.24781504273414612,
        0.30908748507499695
    ],
    "feature_importance": {
        "V10": 3.3493430614471436,
        "U100": 28.39495277404785,
        "V100": 29.357282638549805,
        "U10": 38.89841842651367
    },
    "metrics": {
        "mae": 0.15785089135169983,
        "mse": 0.047359965447444696,
        "rmse": 0.21762344875367795,
        "r2": 0.5337982177734375
    }
  };

  // Select data based on horizon
  const dataMap = {
    '12h': data12h,
    '24h': data24h,
    '36h': data36h,
    '48h': data48h
  };
  
  const data = dataMap[horizon];

  const styles = {
    dashboard: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 25%, #bae6fd 50%, #e0f2fe 75%, #f0f9ff 100%)',
      backgroundSize: '400% 400%',
      animation: 'gradientShift 15s ease infinite',
      color: '#0f172a',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif'
    },
    header: {
      background: 'rgba(255, 255, 255, 0.9)',
      backdropFilter: 'blur(20px)',
      borderBottom: '2px solid rgba(59, 130, 246, 0.2)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
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
      boxShadow: '0 8px 25px rgba(59, 130, 246, 0.3)'
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
      background: 'linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text'
    },
    headerSubtitle: {
      color: '#64748b',
      margin: 0,
      fontSize: '0.9rem',
      fontWeight: '500'
    },
    headerRight: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      padding: '0.75rem 1.5rem',
      background: 'rgba(239, 246, 255, 0.8)',
      borderRadius: '0.75rem',
      border: '2px solid rgba(59, 130, 246, 0.2)',
      backdropFilter: 'blur(10px)'
    },
    horizonBadge: {
      color: '#475569',
      fontSize: '0.875rem',
      fontWeight: '600'
    },
    horizonSelector: {
      position: 'relative'
    },
    horizonDropdown: {
      padding: '0.5rem 2.5rem 0.5rem 1rem',
      borderRadius: '0.5rem',
      border: '2px solid rgba(59, 130, 246, 0.3)',
      background: 'white',
      color: '#2563eb',
      cursor: 'pointer',
      fontSize: '1rem',
      fontWeight: '700',
      appearance: 'none',
      outline: 'none',
      transition: 'all 0.3s',
      backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'12\' height=\'8\' viewBox=\'0 0 12 8\' fill=\'none\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M1 1.5L6 6.5L11 1.5\' stroke=\'%232563eb\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'/%3E%3C/svg%3E")',
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'right 0.75rem center'
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
      background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.75rem',
      border: '2px solid rgba(59, 130, 246, 0.15)',
      transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      cursor: 'pointer',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.05)'
    },
    metricCardHover: {
      transform: 'translateY(-8px) scale(1.02)',
      borderColor: 'rgba(59, 130, 246, 0.4)',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.12), 0 0 40px rgba(59, 130, 246, 0.15)'
    },
    metricGlow: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 70%)',
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
      color: '#64748b',
      fontSize: '0.875rem',
      fontWeight: '700',
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
      gap: '0.25rem',
      fontWeight: '600'
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
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '2px solid rgba(59, 130, 246, 0.15)',
      marginBottom: '2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem',
      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.05)'
    },
    chartTypeSelector: {
      display: 'flex',
      gap: '0.5rem'
    },
    chartTypeButton: {
      padding: '0.5rem 1rem',
      borderRadius: '0.5rem',
      border: '2px solid rgba(59, 130, 246, 0.2)',
      background: 'rgba(239, 246, 255, 0.5)',
      color: '#64748b',
      cursor: 'pointer',
      transition: 'all 0.3s',
      fontSize: '0.875rem',
      fontWeight: '600'
    },
    chartTypeButtonActive: {
      background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
      color: 'white',
      borderColor: 'transparent',
      boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
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
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '2rem',
      border: '2px solid rgba(59, 130, 246, 0.15)',
      transition: 'all 0.3s',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.05)'
    },
    chartHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '1.5rem'
    },
    chartTitle: {
      fontSize: '1.25rem',
      fontWeight: '700',
      margin: 0,
      color: '#0f172a',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem'
    },
    dataSummary: {
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '2rem',
      border: '2px solid rgba(59, 130, 246, 0.15)',
      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.05)'
    },
    summaryGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1.5rem'
    },
    summaryItem: {
      padding: '1rem',
      borderRadius: '0.75rem',
      background: 'rgba(239, 246, 255, 0.6)',
      border: '2px solid rgba(59, 130, 246, 0.1)',
      transition: 'all 0.3s'
    },
    summaryLabel: {
      color: '#64748b',
      marginBottom: '0.5rem',
      fontSize: '0.875rem',
      fontWeight: '600'
    },
    summaryValue: {
      color: '#0f172a',
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
            <span>{horizon === '12h' ? '12-Hour' : horizon === '24h' ? '24-Hour' : horizon === '36h' ? '36-Hour' : '48-Hour'} Forecast</span>
          </div>
        </div>
        <div style={{...styles.metricIconContainer, background: `linear-gradient(135deg, ${colors[color]}22, ${colors[color]}08)`, transform: hoveredCard === metric ? 'scale(1.1) rotate(5deg)' : 'scale(1)'}}>
          <Icon style={{...styles.metricIcon, color: colors[color]}} />
        </div>
      </div>
    </div>
  );

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(255, 255, 255, 0.98)',
          border: '2px solid rgba(59, 130, 246, 0.3)',
          borderRadius: '0.5rem',
          padding: '0.75rem',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
        }}>
          {payload.map((entry, index) => (
            <div key={index} style={{color: entry.color, fontSize: '0.875rem', marginBottom: '0.25rem', fontWeight: '600'}}>
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
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .chart-container:hover {
          border-color: rgba(59, 130, 246, 0.35) !important;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12) !important;
        }
        .summary-item:hover {
          background: rgba(219, 234, 254, 0.8) !important;
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
            <span style={styles.horizonBadge}>Prediction Horizon:</span>
            <div style={styles.horizonSelector}>
              <select 
                value={horizon} 
                onChange={(e) => setHorizon(e.target.value)}
                style={styles.horizonDropdown}
              >
                <option value="12h">12 Hours</option>
                <option value="24h">24 Hours</option>
                <option value="36h">36 Hours</option>
                <option value="48h">48 Hours</option>
              </select>
            </div>
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
            <span style={{color: '#64748b', fontSize: '0.875rem', marginRight: '1rem', fontWeight: '600'}}>Chart Type:</span>
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
          <div style={{color: '#64748b', fontSize: '0.875rem', fontWeight: '600'}}>
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
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#64748b"
                    tick={{ fill: '#475569', fontSize: 12 }}
                    label={{ value: 'Time Steps', position: 'insideBottom', offset: -5, fill: '#64748b' }}
                  />
                  <YAxis 
                    stroke="#64748b"
                    tick={{ fill: '#475569', fontSize: 12 }}
                    label={{ value: 'Power Output', angle: -90, position: 'insideLeft', fill: '#64748b' }}
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
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#64748b"
                    tick={{ fill: '#475569', fontSize: 12 }}
                  />
                  <YAxis 
                    stroke="#64748b"
                    tick={{ fill: '#475569', fontSize: 12 }}
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
                      stroke="rgba(0,0,0,0.1)"
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
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                <XAxis 
                  dataKey="time" 
                  stroke="#64748b"
                  tick={{ fill: '#475569', fontSize: 11 }}
                />
                <YAxis 
                  stroke="#64748b"
                  tick={{ fill: '#475569', fontSize: 11 }}
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
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                <XAxis 
                  type="number"
                  stroke="#64748b"
                  tick={{ fill: '#475569', fontSize: 11 }}
                />
                <YAxis 
                  type="category"
                  dataKey="feature"
                  stroke="#64748b"
                  tick={{ fill: '#475569', fontSize: 12, fontWeight: 600 }}
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
              <p style={{...styles.summaryValue, color: colors.blue}}>
                {horizon === '12h' ? '12 Hours' : horizon === '24h' ? '24 Hours' : horizon === '36h' ? '36 Hours' : '48 Hours'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WindPowerDashboard;
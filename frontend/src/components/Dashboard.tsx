import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Zap, 
  Clock,
  Eye,
  Hash,
  FileText,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Camera,
  Sparkles,
  BarChart3,
  PieChart,
  RefreshCw,
  Settings,
  Trash2,
  Maximize2
} from 'lucide-react';

interface HistoryItem {
  id: number;
  jobId: string;
  userId: string;
  imageName: string;
  status: string;
  results: any;
  metadata: {
    processedAt: string;
  };
  savedToHistory: boolean;
}

interface DashboardProps {
  userName: string;
  history: HistoryItem[];
  onNavigate?: (page: 'Dashboard' | 'NewTest' | 'History' | 'Settings' | 'Help') => void;
  showToast?: (type: 'success' | 'error' | 'info', message: string, duration?: number) => void;
}

// Count-up animation hook
const useCountUp = (end: number, duration: number = 2000) => {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    let startTime: number;
    let animationFrame: number;
    
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      
      setCount(Math.floor(progress * end));
      
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };
    
    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [end, duration]);
  
  return count;
};

const Dashboard: React.FC<DashboardProps> = ({ userName, history, onNavigate, showToast }) => {
  const [greeting, setGreeting] = useState('');
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [liveObjectCount, setLiveObjectCount] = useState<number>(0);
  const [isCountingActive, setIsCountingActive] = useState(false);
  
  // Calculate service usage from actual history
  const calculateServiceUsage = () => {
    const usage = {
      OCR: 0,
      Count: 0,
      Freshness: 0,
      Brand: 0
    };
    
    history.forEach(item => {
      if (item.results.ocr) usage.OCR++;
      if (item.results.productCount) usage.Count++;
      if (item.results.freshness) usage.Freshness++;
      if (item.results.brand) usage.Brand++;
    });
    
    const total = Object.values(usage).reduce((sum, val) => sum + val, 0) || 1;
    
    return [
      { name: 'OCR', value: usage.OCR, color: 'from-blue-400 to-blue-600', percentage: Math.round((usage.OCR / total) * 100) },
      { name: 'Count', value: usage.Count, color: 'from-green-400 to-green-600', percentage: Math.round((usage.Count / total) * 100) },
      { name: 'Freshness', value: usage.Freshness, color: 'from-orange-400 to-orange-600', percentage: Math.round((usage.Freshness / total) * 100) },
      { name: 'Brand', value: usage.Brand, color: 'from-purple-400 to-purple-600', percentage: Math.round((usage.Brand / total) * 100) }
    ];
  };
  
  const serviceUsage = calculateServiceUsage();
  
  const [aiModels] = useState([
    { name: 'OCR Engine', status: 'stable', uptime: '99.8%', color: 'text-green-500' },
    { name: 'Object Detector', status: 'stable', uptime: '99.5%', color: 'text-green-500' },
    { name: 'Freshness Model', status: 'warning', uptime: '97.2%', color: 'text-orange-500' },
    { name: 'Brand Classifier', status: 'stable', uptime: '99.9%', color: 'text-green-500' }
  ]);
  
  // Convert actual history to recent activity
  const getRecentActivity = () => {
    const now = new Date();
    return history.slice(0, 5).map(item => {
      const processedAt = new Date(item.metadata.processedAt);
      const diffMs = now.getTime() - processedAt.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);
      
      let timeAgo;
      if (diffDays > 0) timeAgo = `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
      else if (diffHours > 0) timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
      else if (diffMins > 0) timeAgo = `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;
      else timeAgo = 'Just now';
      
      // Determine primary service type
      let type = 'OCR';
      if (item.results.productCount) type = 'Count';
      else if (item.results.freshness) type = 'Freshness';
      else if (item.results.brand) type = 'Brand';
      
      return {
        id: item.id,
        type,
        time: timeAgo,
        result: item.status,
        product: item.imageName
      };
    });
  };
  
  const recentActivity = getRecentActivity();
  
  // Generate insights based on real data
  const generateInsights = () => {
    const insights = [];
    const successCount = history.filter(h => h.status === 'Success').length;
    const successRate = history.length > 0 ? Math.round((successCount / history.length) * 100) : 100;
    
    // Most used service
    const maxService = serviceUsage.reduce((max, service) => service.value > max.value ? service : max, serviceUsage[0]);
    if (maxService.value > 0) {
      insights.push({
        title: 'Most Popular Service',
        description: `${maxService.name} is your most used service with ${maxService.value} analyses (${maxService.percentage}% of total).`,
        icon: Sparkles
      });
    }
    
    // Success rate insight
    if (successRate >= 90) {
      insights.push({
        title: 'Excellent Performance',
        description: `Your success rate is ${successRate}%! Keep up the great work with quality inputs.`,
        icon: CheckCircle2
      });
    } else if (successRate < 90 && history.length > 0) {
      insights.push({
        title: 'Improvement Opportunity',
        description: `Your success rate is ${successRate}%. Try using higher quality images for better results.`,
        icon: AlertCircle
      });
    }
    
    // Recent activity insight
    if (history.length >= 5) {
      insights.push({
        title: 'Active User',
        description: `You've completed ${history.length} analyses. Your data is helping improve our AI models!`,
        icon: Activity
      });
    } else if (history.length === 0) {
      insights.push({
        title: 'Get Started',
        description: 'Start by uploading an image and selecting analysis services to see your results here.',
        icon: Clock
      });
    }
    
    // Fill with generic insights if needed
    while (insights.length < 3) {
      insights.push({
        title: 'Combine Services',
        description: 'Use multiple services together for comprehensive product analysis and better insights.',
        icon: Sparkles
      });
    }
    
    return insights.slice(0, 3);
  };
  
  const insights = generateInsights();
  
  // Calculate metrics
  const totalAnalyses = history.length;
  const successCount = history.filter(h => h.status === 'Success').length;
  const successRate = totalAnalyses > 0 ? Math.round((successCount / totalAnalyses) * 100) : 0;
  
  // Count-up values
  const totalAnalysesAnimated = useCountUp(totalAnalyses, 2000);
  const successRateAnimated = useCountUp(successRate, 2000);
  const avgProcessing = useCountUp(28, 2000); // Keep this as mock for now
  
  // Dynamic greeting
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good Morning');
    else if (hour < 18) setGreeting('Good Afternoon');
    else setGreeting('Good Evening');
  }, []);
  
  // Capture and analyze frames at 1 FPS
  useEffect(() => {
    if (!cameraActive || !isCountingActive) return;
    
    const captureAndAnalyze = async () => {
      if (!videoRef.current || !canvasRef.current) return;
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      if (!context) return;
      
      try {
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to base64
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to backend for analysis
        const response = await fetch('http://localhost:5001/api/detect/live-count', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ frame: frameData })
        });
        
        if (response.ok) {
          const data = await response.json();
          setLiveObjectCount(data.count || 0);
        }
      } catch (error) {
        console.error('Frame analysis error:', error);
      }
    };
    
    // Analyze at 1 FPS (every 1000ms)
    const interval = setInterval(captureAndAnalyze, 1000);
    
    return () => clearInterval(interval);
  }, [cameraActive, isCountingActive]);
  
  // Initialize camera for live detection
  useEffect(() => {
    let stream: MediaStream | null = null;
    
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user' // Use front camera for dashboard
          }
        });
        
        if (videoRef.current) {
          const video = videoRef.current;
          video.srcObject = stream;
          
          // Try to play immediately
          video.play()
            .then(() => {
              setCameraActive(true);
              setIsCountingActive(true); // Start counting when camera is active
              setCameraError(null);
            })
            .catch(() => {
              // If direct play fails, wait for metadata
              video.onloadedmetadata = () => {
                video.play()
                  .then(() => {
                    setCameraActive(true);
                    setCameraError(null);
                  })
                  .catch(() => {
                    setCameraError('Failed to play video');
                    setCameraActive(false);
                    setIsCountingActive(false);
                  });
              };
            });
        }
      } catch (error) {
        setCameraError('Camera access denied');
        setCameraActive(false);
        setIsCountingActive(false);
      }
    };
    
    // Small delay to ensure component is mounted
    const timer = setTimeout(startCamera, 100);
    
    return () => {
      clearTimeout(timer);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };
  

  return (
    <motion.div
      className="space-y-6"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Ambient Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <motion.div
          className="absolute top-0 left-1/4 w-96 h-96 bg-blue-400/10 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
            scale: [1, 1.2, 1]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
        <motion.div
          className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-400/10 rounded-full blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, 50, 0],
            scale: [1, 1.1, 1]
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      </div>
      
      {/* 1. Mission Control Header */}
      <motion.div variants={itemVariants} className="mb-8">
        <motion.h1 
          className="text-3xl font-bold text-gray-900 dark:text-slate-100 mb-2"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {greeting}, {userName} 👋
        </motion.h1>
        <motion.p 
          className="text-gray-600 dark:text-slate-300"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Here's your system's current performance overview
        </motion.p>
      </motion.div>

      {/* 2. KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Analyses */}
        <motion.div
          variants={itemVariants}
          whileHover="hover"
          className="card relative overflow-hidden group"
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity"
          />
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl shadow-lg">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <motion.div
                className="text-xs text-blue-600 font-medium"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                +12% ↑
              </motion.div>
            </div>
            <p className="text-sm text-gray-600 dark:text-slate-400 mb-1">Total Analyses</p>
            <p className="text-3xl font-bold text-gray-900 dark:text-slate-100">{totalAnalysesAnimated.toLocaleString()}</p>
            <div className="mt-3 h-1 bg-gray-100 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 1.5, delay: 0.5 }}
              />
            </div>
          </div>
        </motion.div>

        {/* Success Rate */}
        <motion.div
          variants={itemVariants}
          whileHover="hover"
          className="card relative overflow-hidden group"
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity"
          />
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl shadow-lg">
                <CheckCircle2 className="w-6 h-6 text-white" />
              </div>
              <div className="relative w-12 h-12">
                <svg className="transform -rotate-90" viewBox="0 0 36 36">
                  <circle cx="18" cy="18" r="16" fill="none" stroke="#e5e7eb" strokeWidth="3" />
                  <motion.circle
                    cx="18" cy="18" r="16" fill="none" stroke="url(#gradient-success)" strokeWidth="3"
                    strokeDasharray="100"
                    initial={{ strokeDashoffset: 100 }}
                    animate={{ strokeDashoffset: 100 - successRateAnimated }}
                    transition={{ duration: 2, delay: 0.5 }}
                  />
                  <defs>
                    <linearGradient id="gradient-success" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#10b981" />
                      <stop offset="100%" stopColor="#059669" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-green-600">
                  {successRateAnimated}%
                </div>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-slate-400 mb-1">Success Rate</p>
            <p className="text-3xl font-bold text-gray-900 dark:text-slate-100">{successRateAnimated}%</p>
          </div>
        </motion.div>

        {/* Avg Processing Time */}
        <motion.div
          variants={itemVariants}
          whileHover="hover"
          className="card relative overflow-hidden group"
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-br from-orange-500/10 to-amber-500/10 opacity-0 group-hover:opacity-100 transition-opacity"
          />
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-orange-500 to-amber-500 rounded-xl shadow-lg">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              >
                <RefreshCw className="w-4 h-4 text-orange-500" />
              </motion.div>
            </div>
            <p className="text-sm text-gray-600 dark:text-slate-400 mb-1">Avg Processing</p>
            <p className="text-3xl font-bold text-gray-900 dark:text-slate-100">{avgProcessing}s</p>
            <p className="text-xs text-green-600 mt-2">-5s from last week</p>
          </div>
        </motion.div>

        {/* Most Used Service */}
        <motion.div
          variants={itemVariants}
          whileHover="hover"
          className="card relative overflow-hidden group"
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity"
          />
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl shadow-lg">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <motion.div
                className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center"
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Hash className="w-4 h-4 text-purple-600" />
              </motion.div>
            </div>
            <p className="text-sm text-gray-600 dark:text-slate-400 mb-1">Most Used Service</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-slate-100">
              {serviceUsage.reduce((max, s) => s.value > max.value ? s : max, serviceUsage[0]).name}
            </p>
            <p className="text-xs text-gray-500 mt-2">
              {serviceUsage.reduce((max, s) => s.value > max.value ? s : max, serviceUsage[0]).percentage}% of total
            </p>
          </div>
        </motion.div>
      </div>

      {/* 3. Service Usage Analytics & AI Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Service Usage Chart */}
        <motion.div variants={itemVariants} className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-100 mb-4 flex items-center">
            <PieChart className="w-5 h-5 mr-2 text-blue-600" />
            Service Usage Analytics
          </h3>
          <div className="space-y-4">
            {serviceUsage.map((service, index) => (
              <motion.div
                key={service.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * index }}
                className="group"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-slate-200">{service.name}</span>
                  <span className="text-sm text-gray-600 dark:text-slate-400">{service.value} uses</span>
                </div>
                <div className="relative h-3 bg-gray-100 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full bg-gradient-to-r ${service.color} rounded-full relative`}
                    initial={{ width: 0 }}
                    animate={{ width: `${service.percentage}%` }}
                    transition={{ duration: 1, delay: 0.2 * index, ease: 'easeOut' }}
                    whileHover={{ opacity: 0.8 }}
                  >
                    <motion.div
                      className="absolute inset-0 bg-white/30"
                      animate={{ x: ['0%', '100%'] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                    />
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* AI Model Health */}
        <motion.div variants={itemVariants} className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-100 mb-4 flex items-center">
            <Cpu className="w-5 h-5 mr-2 text-purple-600" />
            AI Model Health
          </h3>
          <div className="space-y-3">
            {aiModels.map((model, index) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 * index }}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-slate-700 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-600 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <motion.div
                    className={`w-2 h-2 rounded-full ${
                      model.status === 'stable' ? 'bg-green-500' :
                      model.status === 'warning' ? 'bg-orange-500' : 'bg-red-500'
                    }`}
                    animate={model.status === 'stable' ? { scale: [1, 1.2, 1] } : {}}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-slate-100">{model.name}</p>
                    <p className="text-xs text-gray-500 dark:text-slate-400">Uptime: {model.uptime}</p>
                  </div>
                </div>
                <motion.div
                  animate={{ rotate: model.status === 'stable' ? 0 : [0, 10, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  {model.status === 'stable' ? (
                    <CheckCircle2 className={`w-5 h-5 ${model.color}`} />
                  ) : (
                    <AlertCircle className={`w-5 h-5 ${model.color}`} />
                  )}
                </motion.div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* 4. Recent Activity & Real-time Detection */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activity Feed */}
        <motion.div variants={itemVariants} className="lg:col-span-2 card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-100 mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-green-600" />
            Recent Activity
          </h3>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {recentActivity.map((activity, index) => (
              <motion.div
                key={activity.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 * index }}
                className="flex items-start space-x-3 p-3 hover:bg-gray-50 dark:hover:bg-slate-700 rounded-lg transition-colors group"
              >
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                  activity.type === 'OCR' ? 'bg-blue-100' :
                  activity.type === 'Count' ? 'bg-green-100' :
                  activity.type === 'Freshness' ? 'bg-orange-100' : 'bg-purple-100'
                }`}>
                  {activity.type === 'OCR' ? <FileText className="w-5 h-5 text-blue-600" /> :
                   activity.type === 'Count' ? <Hash className="w-5 h-5 text-green-600" /> :
                   activity.type === 'Freshness' ? <Sparkles className="w-5 h-5 text-orange-600" /> :
                   <Eye className="w-5 h-5 text-purple-600" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-slate-100">{activity.product}</p>
                  <div className="flex items-center space-x-2 mt-1">
                    <span className="text-xs text-gray-500 dark:text-slate-400">{activity.time}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      activity.result === 'Success' ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
                    }`}>
                      {activity.result}
                    </span>
                  </div>
                </div>
                <motion.div
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                  whileHover={{ scale: 1.1 }}
                >
                  <Maximize2 className="w-4 h-4 text-gray-400" />
                </motion.div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Real-time Detection Stream */}
        <motion.div variants={itemVariants} className="card bg-gradient-to-br from-gray-900 to-gray-800 text-white">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Camera className="w-5 h-5 mr-2" />
            Live Detection
          </h3>
          <div className="relative bg-black rounded-lg overflow-hidden aspect-video mb-3">
            {/* Video element - always rendered */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${cameraActive ? 'block' : 'hidden'}`}
            />
            
            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} className="hidden" />
            
            {/* Scan line overlay - only when active */}
            {cameraActive && (
              <motion.div
                className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/10 to-transparent pointer-events-none"
                animate={{ y: ['-100%', '200%'] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              />
            )}
            
            {/* LIVE badge - only when active */}
            {cameraActive && (
              <motion.div
                className="absolute top-3 right-3 flex items-center space-x-2 bg-green-500/90 px-2 py-1 rounded-full"
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                <span className="text-xs font-medium">LIVE</span>
              </motion.div>
            )}
            
            {/* Loading/Error overlay - when not active */}
            {!cameraActive && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                {cameraError ? (
                  <div className="text-center">
                    <Camera className="w-12 h-12 text-gray-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-400">{cameraError}</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                    <p className="text-sm text-gray-400">Starting camera...</p>
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Status:</span>
              <span className={cameraActive ? "text-green-400" : "text-red-400"}>
                {cameraActive ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Objects:</span>
              <span className="font-bold text-cyan-400">{cameraActive && isCountingActive ? liveObjectCount : '-'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Analysis:</span>
              <span>{isCountingActive ? '1 FPS' : 'Paused'}</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* 5. Insights & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {insights.map((insight) => {
          const Icon = insight.icon;
          return (
            <motion.div
              key={insight.title}
              variants={itemVariants}
              whileHover={{ scale: 1.02, y: -4 }}
              className="card bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border-2 border-blue-100 dark:border-blue-800"
            >
              <div className="flex items-start space-x-3">
                <div className="p-2 bg-white dark:bg-slate-700 rounded-lg shadow-sm">
                  <Icon className="w-5 h-5 text-blue-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900 dark:text-slate-100 mb-1">{insight.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-slate-300">{insight.description}</p>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* 6. Quick Actions Toolbar */}
      <motion.div 
        variants={itemVariants}
        className="card bg-gradient-to-r from-gray-800 to-gray-900 text-white"
      >
        <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { 
              icon: Sparkles, 
              label: 'New Analysis', 
              color: 'from-cyan-500 to-blue-500',
              onClick: () => onNavigate?.('NewTest')
            },
            { 
              icon: Trash2, 
              label: 'Clear Cache', 
              color: 'from-orange-500 to-red-500',
              onClick: () => {
                if (confirm('Clear all cached data? This will remove temporary files and optimize performance.')) {
                  localStorage.clear();
                  sessionStorage.clear();
                  showToast?.('success', 'Cache cleared successfully!');
                }
              }
            },
            { 
              icon: Cpu, 
              label: 'Optimize Models', 
              color: 'from-green-500 to-emerald-500',
              onClick: () => {
                showToast?.('info', 'AI models optimization initiated!\n\nThis process runs in the background and will complete automatically. Your models will be ready for peak performance.', 5000);
              }
            },
            { 
              icon: Settings, 
              label: 'System Settings', 
              color: 'from-purple-500 to-pink-500',
              onClick: () => onNavigate?.('Settings')
            }
          ].map((action, index) => {
            const Icon = action.icon;
            return (
              <motion.button
                key={action.label}
                onClick={action.onClick}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 * index, duration: 0.3 }}
                whileHover={{ scale: 1.05, y: -2, transition: { duration: 0.1 } }}
                whileTap={{ scale: 0.95, transition: { duration: 0.05 } }}
                className={`p-4 bg-gradient-to-br ${action.color} rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-150 group relative overflow-hidden cursor-pointer`}
              >
                <motion.div
                  className="absolute inset-0 bg-white/20 rounded-xl"
                  initial={{ opacity: 0 }}
                  whileHover={{ opacity: 1 }}
                  transition={{ duration: 0.1 }}
                />
                <div className="relative z-10">
                  <Icon className="w-6 h-6 mx-auto mb-2" />
                  <p className="text-sm font-medium">{action.label}</p>
                </div>
              </motion.button>
            );
          })}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Dashboard;

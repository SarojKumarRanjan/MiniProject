import React, { useRef, useEffect, useState } from 'react';
import { Camera, X, Play, Pause, Smartphone } from 'lucide-react';

interface LiveDetectionModalProps {
  onClose: () => void;
  useIPWebcam?: boolean;
  webcamUrl?: string;
  embedded?: boolean;
}

const LiveDetectionModal: React.FC<LiveDetectionModalProps> = ({ 
  onClose,
  useIPWebcam = false,
  webcamUrl,
  embedded = false
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const ipWebcamImgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [objectCount, setObjectCount] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [selectedFps, setSelectedFps] = useState<number>(5); // Default 5 FPS
  const [useMaxFps, setUseMaxFps] = useState<boolean>(false);

  // Scroll into view when camera starts
  useEffect(() => {
    if (cameraStarted && containerRef.current) {
      // Smooth scroll to bring the video section into view
      setTimeout(() => {
        containerRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 100);
    }
  }, [cameraStarted]);

  // Initialize camera based on mode - only when camera is started
  useEffect(() => {
    if (!cameraStarted) return;
    
    if (useIPWebcam) {
      startIPWebcam();
    } else {
      startLaptopCamera();
    }
    
    return () => {
      stopCamera();
    };
  }, [cameraStarted]); // Only depend on cameraStarted to prevent re-initialization


  // Preview loop - shows live feed only when NOT detecting
  useEffect(() => {
    if (!displayCanvasRef.current) return;
    if (isLoading || error) return;
    if (isDetecting) return; // Don't run preview when detecting

    let animationId: number;

    const previewLoop = () => {
      if (!displayCanvasRef.current) return;
      
      const displayCanvas = displayCanvasRef.current;
      const displayContext = displayCanvas.getContext('2d');
      
      if (!displayContext) return;

      if (useIPWebcam) {
        // Show IP Webcam feed
        if (ipWebcamImgRef.current && ipWebcamImgRef.current.complete) {
          displayCanvas.width = ipWebcamImgRef.current.naturalWidth;
          displayCanvas.height = ipWebcamImgRef.current.naturalHeight;
          displayContext.drawImage(ipWebcamImgRef.current, 0, 0);
        }
      } else {
        // Show laptop camera feed
        if (videoRef.current && videoRef.current.readyState >= 2) {
          displayCanvas.width = videoRef.current.videoWidth;
          displayCanvas.height = videoRef.current.videoHeight;
          displayContext.drawImage(videoRef.current, 0, 0);
        }
      }

      animationId = requestAnimationFrame(previewLoop);
    };

    previewLoop();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [useIPWebcam, isLoading, error, isDetecting]);

  // Live detection loop
  useEffect(() => {
    if (!isDetecting) return;

    // Calculate interval based on selected FPS
    // If useMaxFps is true, use requestAnimationFrame (max system FPS)
    // Otherwise use selected FPS (interval in ms = 1000 / fps)
    const DETECTION_INTERVAL = useMaxFps ? 0 : Math.floor(1000 / selectedFps);
    let intervalId: number;
    let lastTime = Date.now();
    let frameCount = 0;

    const detectLoop = async () => {
      try {
        let frameData: string;

        if (useIPWebcam) {
          // IP Webcam mode
          if (!ipWebcamImgRef.current || !canvasRef.current) return;
          
          const img = ipWebcamImgRef.current;
          const canvas = canvasRef.current;
          const context = canvas.getContext('2d');
          
          if (!context || !img.complete) return;

          // Set canvas size to match image
          if (canvas.width !== img.naturalWidth) {
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
          }

          // Draw current frame
          context.drawImage(img, 0, 0);
          frameData = canvas.toDataURL('image/jpeg', 0.8);
        } else {
          // Laptop camera mode
          if (!videoRef.current || !canvasRef.current) return;
          
          const video = videoRef.current;
          const canvas = canvasRef.current;
          const context = canvas.getContext('2d');
          
          if (!context) return;

          // Set canvas sizes
          if (canvas.width !== video.videoWidth) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          // Draw current frame
          context.drawImage(video, 0, 0);
          frameData = canvas.toDataURL('image/jpeg', 0.8);
        }

        // Send for detection with boxes
        const response = await fetch('http://localhost:5001/api/detect/live-count', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ 
            frame: frameData,
            include_boxes: true 
          })
        });

        if (response.ok) {
          const data = await response.json();
          setObjectCount(data.count || 0);

          // Display annotated frame with boxes
          if (data.annotated_frame && displayCanvasRef.current) {
            const displayContext = displayCanvasRef.current.getContext('2d');
            if (displayContext) {
              const annotatedImg = new Image();
              annotatedImg.onload = () => {
                if (displayCanvasRef.current) {
                  displayCanvasRef.current.width = annotatedImg.naturalWidth;
                  displayCanvasRef.current.height = annotatedImg.naturalHeight;
                }
                displayContext.drawImage(annotatedImg, 0, 0);
              };
              annotatedImg.src = data.annotated_frame;
            }
          }

          // Calculate FPS
          frameCount++;
          const currentTime = Date.now();
          if (currentTime - lastTime >= 1000) {
            setFps(frameCount);
            frameCount = 0;
            lastTime = currentTime;
          }

          setError(null);
        }
      } catch (err) {
        console.error('Detection error:', err);
        if (useIPWebcam) {
          setError('Connection lost. Check if IP Webcam is running.');
        }
      }
    };

    // Start detection loop
    if (useMaxFps) {
      // Use requestAnimationFrame for max FPS
      let rafId: number;
      const rafLoop = () => {
        detectLoop();
        rafId = requestAnimationFrame(rafLoop);
      };
      rafId = requestAnimationFrame(rafLoop);
      
      return () => {
        if (rafId) {
          cancelAnimationFrame(rafId);
        }
      };
    } else {
      // Use setInterval for specific FPS
      intervalId = setInterval(detectLoop, DETECTION_INTERVAL);
      
      return () => {
        if (intervalId) {
          clearInterval(intervalId);
        }
      };
    }
  }, [isDetecting, useIPWebcam, webcamUrl, selectedFps, useMaxFps]);

  const startIPWebcam = async () => {
    // Validate webcamUrl first
    if (!webcamUrl || webcamUrl.trim() === '') {
      setError('No IP Webcam URL provided. Please enter the URL and try again.');
      setIsLoading(false);
      return;
    }
    
    console.log('Starting IP Webcam with URL:', webcamUrl);
    setIsLoading(true);
    setError(null);

    // Normalize URL: remove trailing slash, ensure /video endpoint
    let normalizedUrl = webcamUrl.trim().replace(/\/+$/, '');
    
    // If URL doesn't end with /video, add it
    if (!normalizedUrl.endsWith('/video')) {
      normalizedUrl = `${normalizedUrl}/video`;
    }
    
    console.log('Normalized URL:', normalizedUrl);
    
    // Poll for image readiness and start detection when ready
    let attempts = 0;
    const maxAttempts = 30; // 3 seconds total
    
    const checkImageReady = setInterval(() => {
      attempts++;
      
      if (ipWebcamImgRef.current && ipWebcamImgRef.current.complete && ipWebcamImgRef.current.naturalWidth > 0) {
        console.log('IP Webcam stream is ready, starting detection');
        clearInterval(checkImageReady);
        setIsLoading(false);
        setIsDetecting(true);
      } else if (attempts >= maxAttempts) {
        console.log('Timeout waiting for stream, starting anyway');
        clearInterval(checkImageReady);
        setIsLoading(false);
        setIsDetecting(true);
      }
    }, 100); // Check every 100ms
  };

  const startLaptopCamera = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Try with environment camera (back camera)
      let mediaStream;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'environment'
          }
        });
      } catch (err) {
        // If environment camera fails, try user camera (front camera)
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }
        });
      }

      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        
        // Wait for video to be ready
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play()
            .then(() => {
              setIsLoading(false);
              setIsDetecting(true); // Auto-start detection
            })
            .catch(() => {
              setError('Failed to start video playback.');
              setIsLoading(false);
            });
        };
      }
    } catch (err: any) {
      setError(`Camera access denied: ${err.message || 'Please allow camera permissions.'}`);
      setIsLoading(false);
    }
  };

  const stopCamera = () => {
    setIsDetecting(false);
    
    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
      });
      setStream(null);
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const toggleDetection = () => {
    setIsDetecting(!isDetecting);
  };

  const handleStartCamera = () => {
    setCameraStarted(true);
  };

  const handleStopDetection = () => {
    stopCamera();
    setCameraStarted(false);
    setIsDetecting(false);
    setObjectCount(0);
    setFps(0);
    setError(null);
  };

  const handleClose = () => {
    stopCamera();
    setCameraStarted(false);
    onClose();
  };


  const content = (
    <>
        {/* Header - Simplified for embedded mode */}
        <div className={`flex items-center justify-between p-4 border-b dark:border-slate-700 ${
          embedded ? 'rounded-t-lg' : ''
        } ${
          useIPWebcam 
            ? 'bg-gradient-to-r from-green-600 to-emerald-600'
            : 'bg-gradient-to-r from-blue-600 to-purple-600'
        } text-white`}>
          <div className="flex items-center space-x-3">
            {useIPWebcam ? (
              <Smartphone className="w-6 h-6" />
            ) : (
              <Camera className="w-6 h-6" />
            )}
            <div>
              <h3 className="text-lg font-semibold">
                {useIPWebcam ? 'IP Webcam Live Detection' : 'Live Object Detection'}
              </h3>
              <p className="text-xs text-blue-100">
                {useIPWebcam 
                  ? 'Real-time detection via smartphone camera'
                  : 'Real-time counting with bounding boxes'}
              </p>
            </div>
          </div>
          {!embedded && (
            <button
              onClick={handleClose}
              className="text-white hover:text-gray-200 p-1 transition-colors">
              <X className="w-6 h-6" />
            </button>
          )}
        </div>

        {/* Detection Display */}
        <div className="bg-black relative overflow-hidden" style={{ height: '600px' }}>
          {!cameraStarted && !error && (
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-900 to-black z-10">
              <div className="text-white text-center max-w-md px-4">
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-green-500 to-blue-500 rounded-full flex items-center justify-center">
                  {useIPWebcam ? (
                    <Smartphone className="w-10 h-10" />
                  ) : (
                    <Camera className="w-10 h-10" />
                  )}
                </div>
                <h3 className="text-2xl font-bold mb-2">
                  {useIPWebcam ? 'IP Webcam Live Detection' : 'Live Object Detection'}
                </h3>
                <p className="text-gray-300 mb-6">
                  {useIPWebcam 
                    ? 'Ready to connect to your smartphone camera for real-time object detection'
                    : 'Ready to use your laptop camera for real-time object detection'}
                </p>
                <button
                  onClick={handleStartCamera}
                  className="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white rounded-lg font-medium transition-all duration-200 transform hover:scale-105">
                  <div className="flex items-center space-x-2">
                    <Play className="w-5 h-5" />
                    <span>Start Live Detection</span>
                  </div>
                </button>
                <p className="text-xs text-gray-400 mt-4">
                  {useIPWebcam 
                    ? `Connecting to: ${webcamUrl}`
                    : 'Camera permission will be requested'}
                </p>
              </div>
            </div>
          )}
          
          {isLoading && !error && cameraStarted && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
              <div className="text-white text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
                <p className="text-lg">
                  {useIPWebcam ? 'Connecting to IP Webcam...' : 'Starting camera...'}
                </p>
                <p className="text-sm text-gray-300 mt-2">
                  {useIPWebcam ? webcamUrl : 'Please allow camera access'}
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
              <div className="text-center text-white p-4 max-w-md">
                <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Camera className="w-8 h-8" />
                </div>
                <p className="text-lg font-semibold mb-2">Connection Error</p>
                <p className="mb-4 text-gray-300">{error}</p>
                {useIPWebcam && (
                  <div className="text-xs text-gray-400 bg-black/50 p-3 rounded">
                    <p className="mb-2">Make sure:</p>
                    <ul className="text-left space-y-1">
                      <li>• IP Webcam app is running on phone</li>
                      <li>• Server is started in the app</li>
                      <li>• Both devices on same WiFi</li>
                      <li>• URL is correct: {webcamUrl}</li>
                    </ul>
                  </div>
                )}
                <button
                  onClick={() => {
                    setError(null);
                    setIsDetecting(true);
                  }}
                  className="mt-4 px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors">
                  Retry Connection
                </button>
              </div>
            </div>
          )}

          {/* Hidden video element for laptop camera */}
          {!useIPWebcam && (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="hidden"
            />
          )}

          {/* IP Webcam MJPEG stream as img */}
          {useIPWebcam && webcamUrl && (() => {
            // Normalize URL: remove trailing slash, ensure /video endpoint
            let streamUrl = webcamUrl.trim().replace(/\/+$/, '');
            
            // If URL doesn't already end with /video, add it
            if (!streamUrl.endsWith('/video')) {
              streamUrl = `${streamUrl}/video`;
            }
            
            return (
              <img
                ref={ipWebcamImgRef}
                src={streamUrl}
                alt="IP Webcam Stream"
                className="hidden"
                onLoad={() => {
                  console.log('IP Webcam stream loaded successfully from:', streamUrl);
                  // Don't change loading state here - timeout handles it
                }}
                onError={(e) => {
                  console.error('IP Webcam stream error from:', streamUrl, e);
                  setError(`Cannot connect to IP Webcam. Check the URL and network.`);
                  setIsLoading(false);
                }}
              />
            );
          })()}

          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} className="hidden" />

          {/* Display canvas with annotations */}
          <canvas
            ref={displayCanvasRef}
            className="w-full h-full object-contain"
          />

          {/* Live indicator */}
          {isDetecting && !error && (
            <div className="absolute top-4 right-4 flex items-center space-x-2 bg-green-500 px-3 py-1.5 rounded-full shadow-lg">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-xs font-bold text-white">LIVE</span>
            </div>
          )}

          {/* Camera source badge */}
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${useIPWebcam ? 'bg-green-400' : 'bg-blue-400'}`} />
              <span className="text-xs text-white font-medium">
                {useIPWebcam ? 'IP Webcam' : 'Laptop Camera'}
              </span>
            </div>
          </div>

          {/* Stats overlay */}
          <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg p-4 text-white">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400">{objectCount}</div>
                <div className="text-xs text-gray-300">Objects Detected</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cyan-400">{fps}</div>
                <div className="text-xs text-gray-300">FPS</div>
              </div>
              <div className="text-center">
                <div className={`text-3xl font-bold ${isDetecting ? 'text-green-400' : 'text-orange-400'}`}>
                  {isDetecting ? 'ON' : 'OFF'}
                </div>
                <div className="text-xs text-gray-300">Detection</div>
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className={`p-4 bg-gray-50 dark:bg-slate-700 border-t dark:border-slate-600 ${
          embedded ? 'rounded-b-lg' : ''
        }`}>
          {/* Info Text */}
          <div className="text-sm text-gray-600 dark:text-slate-300 mb-3">
            <p className="font-medium">✓ Green boxes show detected objects</p>
            <p className="text-xs mt-1">
              {useIPWebcam 
                ? `✓ Using smartphone camera at ${useMaxFps ? 'max' : `~${selectedFps}`} FPS` 
                : `✓ Using laptop camera at ${useMaxFps ? 'max' : `~${selectedFps}`} FPS`}
            </p>
          </div>

          {/* FPS Control & Action Buttons - Fixed widths to prevent jumping */}
          <div className="grid grid-cols-[1fr_auto] gap-3 items-center">
            {/* FPS Control - Fixed space with border */}
            <div className="flex items-center gap-3 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-600 rounded-lg px-3 py-2">
              <label className="text-xs font-medium text-gray-600 dark:text-slate-400 uppercase whitespace-nowrap">
                Speed:
              </label>
              <div className="flex-1 flex items-center gap-2">
                <input
                  type="range"
                  min="5"
                  max="40"
                  value={selectedFps}
                  onChange={(e) => {
                    setSelectedFps(Number(e.target.value));
                    setUseMaxFps(false);
                  }}
                  disabled={useMaxFps || !cameraStarted}
                  className="flex-1 h-2 rounded-lg appearance-none cursor-pointer disabled:opacity-40"
                  style={{
                    background: useMaxFps || !cameraStarted 
                      ? '#d1d5db' 
                      : `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((selectedFps - 5) / 35) * 100}%, #d1d5db ${((selectedFps - 5) / 35) * 100}%, #d1d5db 100%)`
                  }}
                />
                <span className="text-base font-bold text-blue-600 dark:text-blue-400 w-8 text-center">
                  {useMaxFps ? '-' : selectedFps}
                </span>
                <div className="relative group">
                  <button
                    onClick={() => setUseMaxFps(!useMaxFps)}
                    disabled={!cameraStarted}
                    className={`px-3 py-2 rounded-lg text-xs font-bold transition-all ${
                      useMaxFps
                        ? 'bg-green-600 text-white shadow-sm'
                        : 'bg-gray-200 dark:bg-slate-600 text-gray-700 dark:text-slate-200 hover:bg-gray-300'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}>
                    {useMaxFps ? '⚡' : 'MAX'}
                  </button>
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                    {useMaxFps ? 'Using max FPS supported by your device' : 'Use maximum FPS'}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons - Fixed space (always reserves width) */}
            <div className="flex gap-3 min-w-[230px] justify-center">
              {cameraStarted ? (
                <>
                  <div className="relative group">
                    <button
                      onClick={toggleDetection}
                      disabled={!!error}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all transform hover:scale-105 hover:shadow-lg ${
                        isDetecting
                          ? 'bg-orange-500 hover:bg-orange-600 text-white'
                          : 'bg-green-500 hover:bg-green-600 text-white'
                      } disabled:opacity-50 disabled:hover:scale-100`}>
                      {isDetecting ? (
                        <>
                          <Pause className="w-4 h-4" />
                          <span>Pause</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          <span>Resume</span>
                        </>
                      )}
                    </button>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      {isDetecting ? 'Pause object detection' : 'Resume object detection'}
                      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                    </div>
                  </div>
                  <div className="relative group">
                    <button
                      onClick={handleStopDetection}
                      className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg font-medium transition-all transform hover:scale-105 hover:shadow-lg">
                      <X className="w-4 h-4" />
                      <span>Stop</span>
                    </button>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Stop and close camera
                      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                    </div>
                  </div>
                </>
              ) : (
                // Reserve space even when buttons not shown
                <div className="w-[230px]"></div>
              )}
              {!embedded && (
                <button
                  onClick={handleClose}
                  className="px-4 py-2 bg-gray-200 dark:bg-slate-600 hover:bg-gray-300 text-gray-700 dark:text-slate-100 rounded-lg font-medium">
                  Close
                </button>
              )}
            </div>
          </div>
        </div>
    </>
  );

  return embedded ? (
    // Embedded mode - no modal wrapper
    <div ref={containerRef} className="bg-white dark:bg-slate-800 rounded-lg w-full flex flex-col overflow-hidden shadow-md">
      {content}
    </div>
  ) : (
    // Modal mode - full modal wrapper
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-lg max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {content}
      </div>
    </div>
  );
};

export default LiveDetectionModal;

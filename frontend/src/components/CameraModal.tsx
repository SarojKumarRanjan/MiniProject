import React, { useRef, useEffect, useState } from 'react';
import { Camera, X, RotateCcw } from 'lucide-react';

interface CameraModalProps {
  onCapture: (imageData: string) => void;
  onClose: () => void;
  useIPWebcam?: boolean;
  webcamUrl?: string;
}

const CameraModal: React.FC<CameraModalProps> = ({ 
  onCapture, 
  onClose,
  useIPWebcam = false,
  webcamUrl
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');

  // Initialize camera or IP Webcam
  useEffect(() => {
    if (useIPWebcam) {
      startIPWebcam();
    } else {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [useIPWebcam]);

  // IP Webcam frame update
  useEffect(() => {
    if (!useIPWebcam) return;
    
    const streamUrl = `${webcamUrl}/shot.jpg`;
    const updateFrame = () => {
      setImageUrl(`${streamUrl}?t=${Date.now()}`);
    };

    updateFrame();
    const intervalId = setInterval(updateFrame, 100); // 10 FPS

    return () => {
      clearInterval(intervalId);
    };
  }, [useIPWebcam, webcamUrl]);

  const startIPWebcam = () => {
    setIsLoading(true);
    setError(null);
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  const startCamera = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment' // Use back camera on mobile
        }
      });
      
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      
      setIsLoading(false);
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Unable to access camera. Please check permissions.');
      setIsLoading(false);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const capturePhoto = () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;

    if (useIPWebcam) {
      // IP Webcam mode
      if (!imgRef.current) return;
      
      const img = imgRef.current;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      context.drawImage(img, 0, 0);
    } else {
      // Laptop camera mode
      if (!videoRef.current) return;
      
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    // Convert canvas to base64 image
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    onCapture(imageData);
    stopCamera();
  };

  const handleImageLoad = () => {
    setIsLoading(false);
    setError(null);
  };

  const handleImageError = () => {
    setError('Cannot connect to IP Webcam. Make sure the app is running.');
    setIsLoading(false);
  };

  const handleClose = () => {
    stopCamera();
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-lg max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b dark:border-slate-700">
          <h3 className="text-lg font-semibold flex items-center text-gray-900 dark:text-slate-100">
            <Camera className="w-5 h-5 mr-2" />
            Camera Capture
          </h3>
          <button
            onClick={handleClose}
            className="text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 p-1">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Camera View */}
        <div className="relative bg-black flex-1 flex items-center justify-center min-h-[400px]">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-white text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                <p className="text-lg">Starting camera...</p>
                <p className="text-sm text-gray-300 mt-2">Please allow camera access when prompted</p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center text-white p-4 max-w-md">
                <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Camera className="w-8 h-8" />
                </div>
                <p className="text-lg font-semibold mb-2">Camera Access Error</p>
                <p className="mb-4 text-gray-300">{error}</p>
                <button 
                  onClick={startCamera}
                  className="btn-primary flex items-center mx-auto">
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Try Again
                </button>
                <p className="text-xs text-gray-400 mt-4">
                  Make sure you've allowed camera access in your browser settings
                </p>
              </div>
            </div>
          )}

          {!isLoading && !error && (
            <>
              {/* Laptop camera */}
              {!useIPWebcam && (
                <video
                  ref={videoRef}
                  className="w-full h-full object-contain"
                  autoPlay
                  playsInline
                  muted
                />
              )}

              {/* IP Webcam */}
              {useIPWebcam && (
                <img
                  ref={imgRef}
                  src={imageUrl}
                  alt="IP Webcam Feed"
                  className="w-full h-full object-contain"
                  onLoad={handleImageLoad}
                  onError={handleImageError}
                />
              )}
              
              {/* Capture overlay */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-4 border-2 border-white border-dashed rounded-lg opacity-50"></div>
                <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                  {useIPWebcam ? 'Position phone to frame product' : 'Position your image within the frame'}
                </div>
                {useIPWebcam && (
                  <div className="absolute top-4 right-4 bg-green-500 px-2 py-1 rounded text-xs text-white font-bold">
                    IP WEBCAM
                  </div>
                )}
              </div>
            </>
          )}

          {/* Hidden canvas for capture */}
          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Controls */}
        <div className="p-4 flex items-center justify-center space-x-4 bg-white dark:bg-slate-800 border-t dark:border-slate-700">
          <button
            onClick={handleClose}
            className="btn-secondary">
            Cancel
          </button>
          
          <button
            onClick={capturePhoto}
            disabled={isLoading || !!error}
            className="btn-primary flex items-center disabled:opacity-50 disabled:cursor-not-allowed">
            <Camera className="w-4 h-4 mr-2" />
            Capture Photo
          </button>
        </div>

        {/* Tips */}
        <div className="px-4 pb-4 text-xs text-gray-500 dark:text-slate-400">
          <p>• Ensure good lighting for better analysis results</p>
          <p>• Hold camera steady and wait for focus</p>
          <p>• Make sure the entire product is visible in the frame</p>
        </div>
      </div>
    </div>
  );
};

export default CameraModal;

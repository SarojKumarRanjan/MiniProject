import React, { useState, useRef } from 'react';
import { Upload, Camera, X, Check, Smartphone } from 'lucide-react';
import type { InputMethod } from '../types';
import IPWebcamUrlInput from './IPWebcamUrlInput';

interface UploadAreaProps {
  onFile: (file: File | null) => void;
  onOpenCamera: () => void;
  selectedFile: File | null;
  inputMethod: InputMethod;
  onInputMethodChange: (method: InputMethod) => void;
  capturedImage: string | null;
  isUploading?: boolean;
  useIPWebcamForCamera?: boolean;
  onToggleIPWebcamForCamera?: () => void;
  useIPWebcamForLiveDetection?: boolean;
  onToggleIPWebcamForLiveDetection?: () => void;
  ipWebcamUrl?: string;
  onIpWebcamUrlChange?: (url: string) => void;
  onSaveIpWebcamUrl?: () => void;
}

const UploadArea: React.FC<UploadAreaProps> = ({
  onFile,
  onOpenCamera,
  selectedFile,
  inputMethod,
  onInputMethodChange,
  capturedImage,
  isUploading = false,
  useIPWebcamForCamera = false,
  onToggleIPWebcamForCamera,
  useIPWebcamForLiveDetection = false,
  onToggleIPWebcamForLiveDetection,
  ipWebcamUrl = '',
  onIpWebcamUrlChange,
  onSaveIpWebcamUrl
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        onFile(file);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFile(files[0]);
    }
  };

  const handleClickUpload = () => {
    fileInputRef.current?.click();
  };

  const borderStyle = isDragOver
    ? "border-blue-400 bg-blue-50"
    : isUploading
    ? "border-gray-300 bg-gray-50"
    : "border-gray-300 hover:border-gray-400";

  return (
    <div className="card">
      <div className="space-y-6">
        {/* Input method selector */}
        <div className="flex items-center justify-center space-x-3">
          <button
            onClick={() => onInputMethodChange('Upload')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              inputMethod === 'Upload' 
                ? "bg-blue-100 text-blue-700" 
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}>
            <div className="flex items-center space-x-2">
              <Upload className="w-4 h-4" />
              <span>Upload Image</span>
            </div>
          </button>
          
          <button
            onClick={() => onInputMethodChange('Camera')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              inputMethod === 'Camera' 
                ? "bg-blue-100 text-blue-700" 
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}>
            <div className="flex items-center space-x-2">
              <Camera className="w-4 h-4" />
              <span>Capture Photo</span>
            </div>
          </button>
          
          <button
            onClick={() => onInputMethodChange('LiveDetection')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              inputMethod === 'LiveDetection' 
                ? "bg-blue-100 text-blue-700" 
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}>
            <div className="flex items-center space-x-2">
              <Camera className="w-4 h-4" />
              <span>Live Detection</span>
            </div>
          </button>
        </div>

        {inputMethod === 'Upload' ? (
          <div
            className={`rounded-lg border-2 border-dashed p-8 flex flex-col items-center gap-4 cursor-pointer transition-all ${borderStyle}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClickUpload}>
            
            {isUploading ? (
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            ) : (
              <Upload className="w-12 h-12 text-gray-400" />
            )}
            
            <div className="text-center">
              <p className="text-lg font-medium text-gray-700">
                {isUploading ? "Processing..." : "Drop image here or click to upload"}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Supports JPG, PNG, WEBP up to 10MB
              </p>
            </div>

            {!isUploading && (
              <div className="flex gap-3">
                <button className="btn-primary" onClick={(e) => { e.stopPropagation(); handleClickUpload(); }}>
                  Choose File
                </button>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        ) : inputMethod === 'Camera' ? (
          // Camera capture section with toggle
          <div className="text-center space-y-4">
            <div className="bg-gray-100 dark:bg-slate-700 rounded-lg p-8 border-2 border-dashed border-gray-300 dark:border-slate-600">
              {useIPWebcamForCamera ? (
                <Smartphone className="w-16 h-16 text-green-600 mx-auto mb-4" />
              ) : (
                <Camera className="w-16 h-16 text-gray-400 dark:text-slate-500 mx-auto mb-4" />
              )}
              <p className="text-lg font-medium text-gray-700 dark:text-slate-200 mb-2">
                {useIPWebcamForCamera ? 'IP Webcam Photo Capture' : 'Camera Ready'}
              </p>
              <p className="text-sm text-gray-500 dark:text-slate-400 mb-4">
                {useIPWebcamForCamera 
                  ? 'Use your smartphone camera to capture photos'
                  : 'Click below to open camera and capture an image'}
              </p>
              <button 
                type="button"
                className="btn-primary"
                onClick={onOpenCamera}>
                Open Camera
              </button>
            </div>

            {/* Camera Source Toggle */}
            {onToggleIPWebcamForCamera && (
              <div className="space-y-3">
                <div className="bg-gray-50 dark:bg-slate-700/30 border border-gray-200 dark:border-slate-600 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Smartphone className="w-5 h-5 text-green-600" />
                      <div className="text-left">
                        <p className="text-sm font-medium text-gray-900 dark:text-slate-100">Use IP Webcam</p>
                        <p className="text-xs text-gray-500 dark:text-slate-400">
                          {useIPWebcamForCamera ? 'Using smartphone camera' : 'Using aptop webCam'}
                        </p>
                      </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input 
                        type="checkbox" 
                        className="sr-only peer" 
                        checked={useIPWebcamForCamera}
                        onChange={onToggleIPWebcamForCamera}
                      />
                      <div className="w-11 h-6 bg-gray-200 dark:bg-slate-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-green-300 dark:peer-focus:ring-green-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                    </label>
                  </div>
                </div>
                
                {/* IP Webcam URL Input */}
                {useIPWebcamForCamera && onIpWebcamUrlChange && (
                  <IPWebcamUrlInput
                    url={ipWebcamUrl}
                    onUrlChange={onIpWebcamUrlChange}
                    onSave={onSaveIpWebcamUrl}
                  />
                )}
              </div>
            )}
          </div>
        ) : (
          // Live Detection section
          <div className="text-center space-y-4">
            <div className="bg-gradient-to-br from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-8 border-2 border-dashed border-green-300 dark:border-green-700">
              {useIPWebcamForLiveDetection ? (
                <Smartphone className="w-16 h-16 text-green-500 mx-auto mb-4 animate-pulse" />
              ) : (
                <Camera className="w-16 h-16 text-green-500 mx-auto mb-4 animate-pulse" />
              )}
              <p className="text-lg font-medium text-gray-700 dark:text-slate-200 mb-2">
                {useIPWebcamForLiveDetection ? 'IP Webcam Live Detection' : 'Live Object Detection'}
              </p>
              <p className="text-sm text-gray-500 dark:text-slate-400 mb-4">
                {useIPWebcamForLiveDetection 
                  ? 'Real-time detection using smartphone camera'
                  : 'Real-time object counting with bounding boxes'}
              </p>
              <button 
                className="btn-primary bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600"
                onClick={onOpenCamera}>
                Start Live Detection
              </button>
            </div>
            
            {/* Camera Source Toggle */}
            {onToggleIPWebcamForLiveDetection && (
              <div className="space-y-3">
                <div className="bg-gray-50 dark:bg-slate-700/30 border border-gray-200 dark:border-slate-600 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Smartphone className="w-5 h-5 text-green-600" />
                      <div className="text-left">
                        <p className="text-sm font-medium text-gray-900 dark:text-slate-100">Use IP Webcam</p>
                        <p className="text-xs text-gray-500 dark:text-slate-400">
                          {useIPWebcamForLiveDetection ? 'Using smartphone camera' : 'Using laptop camera'}
                        </p>
                      </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input 
                        type="checkbox" 
                        className="sr-only peer" 
                        checked={useIPWebcamForLiveDetection}
                        onChange={onToggleIPWebcamForLiveDetection}
                      />
                      <div className="w-11 h-6 bg-gray-200 dark:bg-slate-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-green-300 dark:peer-focus:ring-green-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                    </label>
                  </div>
                </div>
                
                {/* IP Webcam URL Input */}
                {useIPWebcamForLiveDetection && onIpWebcamUrlChange && (
                  <IPWebcamUrlInput
                    url={ipWebcamUrl}
                    onUrlChange={onIpWebcamUrlChange}
                    onSave={onSaveIpWebcamUrl}
                  />
                )}
              </div>
            )}
            
            <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg p-4 text-left">
              <p className="text-sm font-medium text-blue-900 dark:text-blue-200 mb-2">✨ Live Detection Features:</p>
              <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1">
                <li>• Real-time object counting at 5 FPS</li>
                <li>• Green bounding boxes around detected objects</li>
                <li>• Live FPS and count display</li>
                <li>• Pause/Resume functionality</li>
                {useIPWebcamForLiveDetection && <li>• Uses high-quality smartphone camera</li>}
              </ul>
            </div>
          </div>
        )}

        {/* Selected file preview */}
        {(selectedFile || capturedImage) && (
          <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <Check className="w-8 h-8 text-green-600" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 dark:text-slate-100">
                  {selectedFile ? selectedFile.name : "Captured Image"}
                </p>
                <p className="text-sm text-gray-500 dark:text-slate-400">
                  {selectedFile 
                    ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                    : "Ready for analysis"
                  }
                </p>
              </div>
              <button 
                className="text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300"
                onClick={() => onFile(null)}
                aria-label="Remove file">
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadArea;

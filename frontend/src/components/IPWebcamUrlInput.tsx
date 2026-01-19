import React from 'react';

interface IPWebcamUrlInputProps {
  url: string;
  onUrlChange: (url: string) => void;
  onSave?: () => void;
}

const IPWebcamUrlInput: React.FC<IPWebcamUrlInputProps> = ({
  url,
  onUrlChange,
  onSave
}) => {
  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
      <label className="block text-sm font-medium text-gray-900 dark:text-slate-100 mb-2">
        IP Webcam URL
      </label>
      <div className="flex space-x-2">
        <input
          type="text"
          value={url}
          onChange={(e) => onUrlChange(e.target.value)}
          placeholder="http://192.168.1.100:8080"
          className="flex-1 px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-100 text-sm focus:ring-2 focus:ring-green-500 focus:border-transparent"
        />
        {onSave && (
          <button
            onClick={onSave}
            className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg text-sm font-medium transition-colors">
            Save
          </button>
        )}
      </div>
      <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
        Enter your phone's IP Webcam URL (e.g., http://192.168.1.100:8080)
      </p>
    </div>
  );
};

export default IPWebcamUrlInput;

import React, { useState } from 'react';
import { FileText, Hash, Leaf, Shield, Loader, AlertCircle, Copy, Eye, Download, RotateCcw } from 'lucide-react';
import type { ServiceType, AnalysisStatus } from '../types';

interface ResultCardProps {
  service: ServiceType;
  status: AnalysisStatus;
  result?: any;
  onAction: (action: string) => void;
}

const ResultCard: React.FC<ResultCardProps> = ({ service, status, result, onAction }) => {
  const [showRawData, setShowRawData] = useState(false);

  const getServiceInfo = (service: ServiceType) => {
    switch (service) {
      case 'OCR':
        return { title: 'Product Description', shortTitle: 'OCR', colorClass: 'text-blue-600' };
      case 'ProductCount':
        return { title: 'Product Count', shortTitle: 'COUNT', colorClass: 'text-green-600' };
      case 'Freshness':
        return { title: 'Freshness Analysis', shortTitle: 'FRESH', colorClass: 'text-orange-600' };
      case 'BrandRecognition':
        return { title: 'Brand Recognition', shortTitle: 'BRAND', colorClass: 'text-purple-600' };
    }
  };

  const getStatusBadge = (status: AnalysisStatus) => {
    switch (status) {
      case 'Success':
        return { text: 'Success', className: 'status-badge-success' };
      case 'Warning':
        return { text: 'Warning', className: 'status-badge-warning' };
      case 'Failed':
        return { text: 'Failed', className: 'status-badge-error' };
      case 'Processing':
        return { text: 'Processing', className: 'status-badge-warning' };
      case 'Pending':
        return { text: 'Pending', className: 'status-badge-warning' };
    }
  };

  const getServiceIcon = (service: ServiceType) => {
    switch (service) {
      case 'OCR':
        return <FileText className="w-5 h-5" />;
      case 'ProductCount':
        return <Hash className="w-5 h-5" />;
      case 'Freshness':
        return <Leaf className="w-5 h-5" />;
      case 'BrandRecognition':
        return <Shield className="w-5 h-5" />;
    }
  };

  const renderResultContent = () => {
    switch (service) {
      case 'OCR':
        if (status === 'Success') {
          return (
            <div className="space-y-3">
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-sm text-gray-700 font-mono leading-relaxed">
                  Sample extracted text: Product Name, Brand, Expiration Date: 2024-12-31, Ingredients: Water, Sugar, Natural Flavors
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <button 
                  className="btn-secondary text-xs py-1 px-2"
                  onClick={() => onAction('copy')}>
                  <Copy className="w-3 h-3 mr-1" />
                  Copy Text
                </button>
                <button 
                  className="btn-secondary text-xs py-1 px-2"
                  onClick={() => onAction('highlight')}>
                  Highlight in Image
                </button>
              </div>
            </div>
          );
        }
        break;

      case 'ProductCount':
        if (status === 'Success') {
          const productCount = result?.productCount?.total || 0;
          const confidence = Math.min(95, Math.max(70, 70 + (productCount * 2))); // Dynamic confidence based on count
          return (
            <div className="space-y-3">
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-900">{productCount}</div>
                <div className="text-sm text-gray-500">Objects Detected</div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: `${confidence}%` }}></div>
              </div>
              <div className="text-xs text-gray-500 text-center">{confidence}% confidence</div>
            </div>
          );
        }
        break;

      case 'Freshness':
        if (status === 'Success') {
          return (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-lg font-semibold text-green-600">Fresh</div>
                  <div className="text-sm text-gray-500">Quality Score</div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-gray-900">8.7</div>
                  <div className="text-sm text-gray-500">/10</div>
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full" style={{ width: '87%' }}></div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="text-center">
                  <div className="font-medium">Color</div>
                  <div className="text-green-600">Good</div>
                </div>
                <div className="text-center">
                  <div className="font-medium">Texture</div>
                  <div className="text-green-600">Firm</div>
                </div>
                <div className="text-center">
                  <div className="font-medium">Spots</div>
                  <div className="text-yellow-600">Few</div>
                </div>
              </div>
            </div>
          );
        }
        break;

      case 'BrandRecognition':
        if (status === 'Success') {
          const brandData = result?.brand;
          const matches = brandData?.matches || [];
          
          if (matches.length === 0) {
            return (
              <div className="text-center py-6">
                <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Shield className="w-6 h-6 text-gray-400" />
                </div>
                <p className="text-sm text-gray-600 mb-1">No brands detected</p>
                <p className="text-xs text-gray-500">Try uploading an image with visible brand text</p>
              </div>
            );
          }
          
          return (
            <div className="space-y-3">
              <div className="space-y-2">
                {matches.map((match: any, index: number) => {
                  const confidencePercent = Math.round((match.confidence || 0) * 100);
                  const isCounterfeit = match.isCounterfeit || false;
                  const brandInitial = match.brand?.charAt(0) || 'B';
                  
                  return (
                    <div 
                      key={index}
                      className={`flex items-center justify-between p-3 rounded-lg ${
                        isCounterfeit 
                          ? 'bg-red-50 border border-red-200' 
                          : 'bg-green-50 border border-green-200'
                      }`}>
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          isCounterfeit ? 'bg-red-100' : 'bg-green-100'
                        }`}>
                          <span className={`text-xs font-bold ${
                            isCounterfeit ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {brandInitial}
                          </span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{match.brand}</div>
                          <div className="text-sm text-gray-500">
                            Confidence: {confidencePercent}%
                            {match.matched_text && match.matched_text.length > 0 && (
                              <span className="ml-2 text-xs">
                                (found: {match.matched_text.join(', ')})
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <span className={isCounterfeit ? 'status-badge-error' : 'status-badge-success'}>
                        {isCounterfeit ? 'Counterfeit' : 'Authentic'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        }
        break;
    }

    // Default states
    if (status === 'Processing') {
      return (
        <div className="flex items-center justify-center py-8">
          <Loader className="w-8 h-8 text-blue-600 animate-spin" />
          <span className="ml-3 text-gray-600">Processing...</span>
        </div>
      );
    }

    if (status === 'Failed') {
      return (
        <div className="text-center py-6">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-3" />
          <p className="text-sm text-gray-600">Analysis failed. Please try again.</p>
          <button 
            className="btn-secondary text-xs mt-3 mx-auto flex items-center"
            onClick={() => onAction('retry')}>
            <RotateCcw className="w-3 h-3 mr-1" />
            Retry
          </button>
        </div>
      );
    }

    return (
      <div className="text-center py-6 text-gray-500">
        <p className="text-sm">No results available</p>
      </div>
    );
  };

  const { title, shortTitle, colorClass } = getServiceInfo(service);
  const { text: statusText, className: statusClass } = getStatusBadge(status);

  return (
    <div className="card animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg bg-gray-100 ${colorClass}`}>
            {getServiceIcon(service)}
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">{shortTitle}</p>
          </div>
        </div>
        <span className={statusClass}>{statusText}</span>
      </div>

      <div className="mb-4">
        {renderResultContent()}
      </div>

      <div className="flex items-center justify-between pt-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <button 
            className="flex items-center btn-secondary text-xs py-1 px-2 gap-1"
            onClick={() => onAction('details')}>
            <Eye className="w-3 h-3 mr-1" />
            View Details
          </button>
          {status === 'Success' && (
            <button 
              className="btn-secondary text-xs py-1 px-2"
              onClick={() => onAction('export')}>
              <Download className="w-3 h-3 mr-1" />
              Export
            </button>
          )}
        </div>
        
        <button 
          className={`text-xs px-2 py-1 rounded transition-colors ${
            showRawData ? "bg-gray-200 text-gray-700" : "text-gray-500 hover:text-gray-700"
          }`}
          onClick={() => setShowRawData(!showRawData)}>
          Raw JSON
        </button>
      </div>

      {showRawData && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="bg-gray-900 rounded-lg p-3 text-xs">
            <pre className="text-green-400 overflow-x-auto">
              {`{
  "service": "${shortTitle}",
  "status": "${statusText}",
  "confidence": 0.92,
  "timestamp": "${Date.now()}",
  "data": { ... }
}`}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultCard;

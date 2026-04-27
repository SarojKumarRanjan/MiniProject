// Type definitions for the Smart Quality Test System

export interface Overlay {
  id: string;
  bbox: [number, number, number, number]; // [x, y, w, h]
  label: string;
  confidence: number;
}

export type ServiceType = 'OCR' | 'ProductCount' | 'Freshness' | 'BrandRecognition';

export type AnalysisStatus = 'Pending' | 'Processing' | 'Success' | 'Warning' | 'Failed';

export interface OcrBox {
  bbox: [number, number, number, number];
  text: string;
}

export interface OcrResult {
  text: string;
  boxes: OcrBox[];
}

export interface Detection {
  label: string;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface ProductCountResult {
  total: number;
  detections: Detection[];
}

export interface FreshnessResult {
  score: number;
  label: string;
  regions: Overlay[];
  color?: string;
  texture?: string;
  spots?: string;
}

export interface BrandMatch {
  brand: string;
  confidence: number;
  bbox: [number, number, number, number];
  isCounterfeit: boolean;
}

export interface BrandResult {
  matches: BrandMatch[];
}

export interface AnalysisResults {
  ocr?: OcrResult;
  productCount?: ProductCountResult;
  freshness?: FreshnessResult;
  brand?: BrandResult;
}

export interface AnalysisMetadata {
  processedAt: string;
}

export interface AnalysisResponse {
  id?: number;
  jobId: string;
  status: AnalysisStatus;
  results: AnalysisResults;
  metadata: AnalysisMetadata;
  imageName?: string;
}

export type InputMethod = 'Camera' | 'Upload' | 'LiveDetection' | 'PreviewStatic';

export type NavigationPage = 'Dashboard' | 'NewTest' | 'History' | 'Settings' | 'Help';

export interface ServiceInfo {
  service: ServiceType;
  title: string;
  description: string;
  estimatedTime: string;
  tooltip: string;
}

export interface NavItem {
  page: NavigationPage;
  label: string;
  icon: React.ReactNode;
  description: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string | null;
  createdAt: string;
  lastLogin?: string;
}

export interface HistoryItem extends AnalysisResponse {
  id: number;
  userId: string;
  imageName: string;
  imageUrl?: string;
  savedToHistory: boolean;
}

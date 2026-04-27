import type { ServiceType, AnalysisResponse, AnalysisStatus, User as UserType } from '../types';

const BASE_URL = 'http://localhost:5001';

// Convert ServiceType to string for API
export const serviceTypeToString = (service: ServiceType): string => {
  switch (service) {
    case 'OCR':
      return 'ocr';
    case 'ProductCount':
      return 'product_count';
    case 'Freshness':
      return 'freshness';
    case 'BrandRecognition':
      return 'brand';
  }
};

// Convert analysis status from string
export const stringToAnalysisStatus = (status: string): AnalysisStatus => {
  switch (status.toLowerCase()) {
    case 'pending':
      return 'Pending';
    case 'processing':
      return 'Processing';
    case 'success':
      return 'Success';
    case 'warning':
      return 'Warning';
    case 'failed':
      return 'Failed';
    default:
      return 'Failed';
  }
};

// Analyze image endpoint
export const analyzeImage = async (
  file: File | null,
  capturedImage: string | null,
  staticImagePath: string | null,
  services: ServiceType[],
  _onProgress?: (message: string) => void
): Promise<AnalysisResponse> => {
  try {
    if (!file && !capturedImage && !staticImagePath) {
      throw new Error('No image provided');
    }

    const formData = new FormData();
    
    // Add file or captured image or static image path
    if (file) {
      formData.append('image', file);
    } else if (capturedImage) {
      formData.append('captured_image', capturedImage);
    } else if (staticImagePath) {
      formData.append('static_image_path', staticImagePath);
    }
    
    // Add services
    services.forEach(service => {
      formData.append('services', serviceTypeToString(service));
    });
    
    const response = await fetch(`${BASE_URL}/capture`, {
      method: 'POST',
      credentials: 'include', // Include cookies for session
      body: formData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }
    
    const data = await response.json();
    
    // Transform API response to our types
    const result: AnalysisResponse = {
      jobId: data.job_id || 'unknown',
      status: stringToAnalysisStatus(data.status),
      results: {
        ocr: data.results?.ocr ? {
          text: data.results.ocr.text || '',
          boxes: data.results.ocr.boxes || []
        } : undefined,
        productCount: data.results?.product_count ? {
          total: data.results.product_count.total || 0,
          detections: data.results.product_count.detections || []
        } : undefined,
        freshness: data.results?.freshness ? {
          score: data.results.freshness.score || 0,
          label: data.results.freshness.label || '',
          regions: data.results.freshness.regions || [],
          color: data.results.freshness.color,
          texture: data.results.freshness.texture,
          spots: data.results.freshness.spots
        } : undefined,
        brand: data.results?.brand ? {
          matches: data.results.brand.matches || []
        } : undefined,
      },
      metadata: {
        processedAt: data.metadata?.processed_at || new Date().toISOString()
      }
    };
    
    return result;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

// Get static assets
export const getStaticAssets = async (): Promise<Record<string, string[]>> => {
  try {
    const response = await fetch(`${BASE_URL}/api/static-assets`, {
      method: 'GET',
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch static assets:', error);
    return {};
  }
};

// Get analysis status (for polling)
export const getAnalysisStatus = async (jobId: string): Promise<AnalysisResponse> => {
  try {
    const response = await fetch(`${BASE_URL}/api/analyze/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    
    // Transform response similar to analyzeImage
    const result: AnalysisResponse = {
      jobId: data.job_id || jobId,
      status: stringToAnalysisStatus(data.status),
      results: data.results || {},
      metadata: {
        processedAt: data.metadata?.processed_at || new Date().toISOString()
      }
    };
    
    return result;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

// Delete history item
export const deleteHistoryItem = async (historyId: number): Promise<void> => {
  try {
    const response = await fetch(`${BASE_URL}/api/history/${historyId}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP ${response.status}`);
    }
  } catch (error) {
    console.error('Delete history item error:', error);
    throw error;
  }
};

// Get history
export const getHistory = async (): Promise<AnalysisResponse[]> => {
  try {
    const response = await fetch(`${BASE_URL}/api/history`, {
      method: 'GET',
      credentials: 'include', // Include cookies for session
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    
  // Transform array of responses and normalize result keys
  return data.map((item: any) => {
    const results = item.results || {};
    
    // Transform product_count to productCount for consistency
    const normalizedResults = {
      ...results,
      productCount: results.product_count || results.productCount,
    };
    
    // Remove the old key to avoid confusion
    delete normalizedResults.product_count;
    
    return {
      id: item.id, // Include the database ID
      jobId: item.jobId || 'unknown',
      status: stringToAnalysisStatus(item.status),
      results: normalizedResults,
      imageName: item.imageName || 'Analysis Result',
      metadata: {
        processedAt: item.metadata?.processedAt || new Date().toISOString()
      }
    } as any;
  });
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

// Authentication functions
export const login = async (email: string, password: string): Promise<UserType> => {
  try {
    const response = await fetch(`${BASE_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include', // Include cookies for session
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Login failed');
    }

    const data = await response.json();
    return data.user;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

export const signup = async (name: string, email: string, password: string): Promise<UserType> => {
  try {
    const response = await fetch(`${BASE_URL}/api/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ name, email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Signup failed');
    }

    const data = await response.json();
    return data.user;
  } catch (error) {
    console.error('Signup error:', error);
    throw error;
  }
};

export const logout = async (): Promise<void> => {
  try {
    await fetch(`${BASE_URL}/api/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    });
  } catch (error) {
    console.error('Logout error:', error);
  }
};

export const getCurrentUser = async (): Promise<UserType | null> => {
  try {
    const response = await fetch(`${BASE_URL}/api/auth/me`, {
      method: 'GET',
      credentials: 'include',
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data.user;
  } catch (error) {
    console.error('Get current user error:', error);
    return null;
  }
};

export const updateProfile = async (name: string, email: string): Promise<UserType> => {
  try {
    const response = await fetch(`${BASE_URL}/api/auth/update-profile`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ name, email }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Profile update failed');
    }

    const data = await response.json();
    return data.user;
  } catch (error) {
    console.error('Update profile error:', error);
    throw error;
  }
};

export const uploadAvatar = async (file: File): Promise<UserType> => {
  try {
    const formData = new FormData();
    formData.append('avatar', file);

    const response = await fetch(`${BASE_URL}/api/auth/upload-avatar`, {
      method: 'POST',
      credentials: 'include',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Avatar upload failed');
    }

    const data = await response.json();
    return data.user;
  } catch (error) {
    console.error('Upload avatar error:', error);
    throw error;
  }
};

export const deleteAvatar = async (): Promise<UserType> => {
  try {
    const response = await fetch(`${BASE_URL}/api/auth/delete-avatar`, {
      method: 'DELETE',
      credentials: 'include',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Avatar deletion failed');
    }

    const data = await response.json();
    return data.user;
  } catch (error) {
    console.error('Delete avatar error:', error);
    throw error;
  }
};

// Test API connection
export const testConnection = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${BASE_URL}/api/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    return response.ok;
  } catch (error) {
    console.error('Connection test failed:', error);
    return false;
  }
};

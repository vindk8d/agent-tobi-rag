/**
 * API Client for RAG-Tobi Frontend
 * Centralized API communication with the backend
 */

import { 
  APIResponse, 
  SystemHealth, 
  ConversationRequest, 
  ConversationResponse,
  DocumentModel,
  DataSourceModel,
  DashboardStats
} from '@/types';

// Get API URL from environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class APIClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  /**
   * Make a generic API request
   */
  private async request<T = any>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    const requestOptions = { ...defaultOptions, ...options };

    try {
      const response = await fetch(url, requestOptions);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Health check endpoint
   */
  async getHealth(): Promise<APIResponse<SystemHealth>> {
    return this.request<SystemHealth>('/health');
  }

  /**
   * Get dashboard statistics
   */
  async getDashboardStats(): Promise<APIResponse<DashboardStats>> {
    return this.request<DashboardStats>('/api/dashboard/stats');
  }

  /**
   * Chat/conversation endpoints
   */
  async sendMessage(request: ConversationRequest): Promise<APIResponse<ConversationResponse>> {
    return this.request<ConversationResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Document management endpoints
   */
  async getDocuments(): Promise<APIResponse<DocumentModel[]>> {
    return this.request<DocumentModel[]>('/api/documents');
  }

  async uploadDocument(formData: FormData): Promise<APIResponse<DocumentModel>> {
    return this.request<DocumentModel>('/api/documents/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it for FormData
    });
  }

  async deleteDocument(documentId: string): Promise<APIResponse<void>> {
    return this.request<void>(`/api/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Data source management endpoints
   */
  async getDataSources(): Promise<APIResponse<DataSourceModel[]>> {
    return this.request<DataSourceModel[]>('/api/datasources');
  }

  async createDataSource(dataSource: Partial<DataSourceModel>): Promise<APIResponse<DataSourceModel>> {
    return this.request<DataSourceModel>('/api/datasources', {
      method: 'POST',
      body: JSON.stringify(dataSource),
    });
  }

  async updateDataSource(id: string, dataSource: Partial<DataSourceModel>): Promise<APIResponse<DataSourceModel>> {
    return this.request<DataSourceModel>(`/api/datasources/${id}`, {
      method: 'PUT',
      body: JSON.stringify(dataSource),
    });
  }

  async deleteDataSource(id: string): Promise<APIResponse<void>> {
    return this.request<void>(`/api/datasources/${id}`, {
      method: 'DELETE',
    });
  }

  async scrapeDataSource(id: string): Promise<APIResponse<any>> {
    return this.request<any>(`/api/datasources/${id}/scrape`, {
      method: 'POST',
    });
  }

  /**
   * Memory debug endpoints
   */
  async getUsers(): Promise<APIResponse<any[]>> {
    return this.request<any[]>('/api/memory/users');
  }

  async getUserMemories(userId: string): Promise<APIResponse<any[]>> {
    return this.request<any[]>(`/api/memory/users/${userId}/memories`);
  }

  /**
   * Test connectivity
   */
  async testConnection(): Promise<boolean> {
    try {
      const response = await this.getHealth();
      return response.success;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient(API_BASE_URL);

// Export class for testing or custom instances
export { APIClient };

// Export types for convenience
export type { APIResponse, SystemHealth, ConversationRequest, ConversationResponse };

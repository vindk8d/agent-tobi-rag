/**
 * TypeScript type definitions for RAG-Tobi application
 */

// Base types
export interface BaseEntity {
  id: string;
  created_at: string;
  updated_at?: string;
}

// API Response types
export interface APIResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  error?: string;
}

// Conversation types
export enum MessageRole {
  USER = "user",
  ASSISTANT = "assistant",
  SYSTEM = "system",
}

export enum ConversationType {
  CHAT = "chat",
  QUERY = "query",
  FEEDBACK = "feedback",
}

export interface Message {
  role: MessageRole;
  content: string;
  timestamp?: string;
  metadata?: Record<string, any>;
}

export interface ConversationRequest {
  message: string;
  conversation_id?: string;
  conversation_type?: ConversationType;
  context?: Record<string, any>;
  user_id?: string;
}

export interface ConversationResponse {
  message: string;
  conversation_id: string;
  response_metadata?: Record<string, any>;
  sources?: Array<{
    title: string;
    url?: string;
    content: string;
    confidence?: number;
  }>;
  suggestions?: string[];
  confidence_score?: number;
}

export interface ConversationHistory extends BaseEntity {
  conversation_id: string;
  user_id?: string;
  messages: Message[];
  conversation_type: ConversationType;
  metadata?: Record<string, any>;
  is_active: boolean;
}

// Document types
export enum DocumentType {
  PDF = "pdf",
  WORD = "word",
  TEXT = "text",
  HTML = "html",
  MARKDOWN = "markdown",
  WEB_PAGE = "web_page",
}

export enum DocumentStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
  INDEXED = "indexed",
}

export interface DocumentModel extends BaseEntity {
  title: string;
  content?: string;
  document_type: DocumentType;
  file_path?: string;
  url?: string;
  file_size?: number;
  page_count?: number;
  status: DocumentStatus;
  embedding_count?: number;
  metadata?: Record<string, any>;
  data_source_id?: string;
}

export interface DocumentUploadRequest {
  title: string;
  document_type: DocumentType;
  metadata?: Record<string, any>;
}

export interface DocumentUploadResponse {
  document_id: string;
  upload_url?: string;
  status: DocumentStatus;
  message: string;
}

// Data Source types
export enum DataSourceType {
  WEBSITE = "website",
  FILE_UPLOAD = "file_upload",
  API = "api",
  DATABASE = "database",
}

export enum DataSourceStatus {
  ACTIVE = "active",
  INACTIVE = "inactive",
  ERROR = "error",
  PENDING = "pending",
}

export enum ScrapingFrequency {
  DAILY = "daily",
  WEEKLY = "weekly",
  MONTHLY = "monthly",
  MANUAL = "manual",
}

export interface DataSourceModel extends BaseEntity {
  name: string;
  description?: string;
  source_type: DataSourceType;
  url?: string;
  status: DataSourceStatus;
  scraping_frequency: ScrapingFrequency;
  last_scraped?: string;
  last_success?: string;
  document_count: number;
  error_count: number;
  last_error?: string;
  configuration?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface DataSourceRequest {
  name: string;
  description?: string;
  source_type: DataSourceType;
  url?: string;
  scraping_frequency?: ScrapingFrequency;
  configuration?: Record<string, any>;
}

export interface ScrapingResult {
  data_source_id: string;
  success: boolean;
  documents_found: number;
  documents_processed: number;
  documents_failed: number;
  error_message?: string;
  scraped_at: string;
  processing_time?: number;
}

// UI Component types
export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface ErrorState {
  hasError: boolean;
  message?: string;
  details?: string;
}

export interface PaginationState {
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
}

export interface FilterState {
  search?: string;
  status?: string;
  type?: string;
  date_range?: {
    start: string;
    end: string;
  };
}

// Dashboard types
export interface DashboardStats {
  total_documents: number;
  total_data_sources: number;
  active_conversations: number;
  processing_documents: number;
  failed_documents: number;
  last_updated: string;
}

export interface SystemHealth {
  database: 'healthy' | 'unhealthy' | 'warning';
  embeddings: 'healthy' | 'unhealthy' | 'warning';
  scraping: 'healthy' | 'unhealthy' | 'warning';
  api: 'healthy' | 'unhealthy' | 'warning';
  last_check: string;
}

// Form types
export interface FormField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'url' | 'select' | 'textarea' | 'file' | 'checkbox';
  required?: boolean;
  placeholder?: string;
  options?: Array<{ value: string; label: string }>;
  validation?: {
    pattern?: string;
    min?: number;
    max?: number;
    minLength?: number;
    maxLength?: number;
  };
}

export interface FormState {
  values: Record<string, any>;
  errors: Record<string, string>;
  touched: Record<string, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

// Theme types
export interface ThemeConfig {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    error: string;
    warning: string;
    success: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      xs: string;
      sm: string;
      base: string;
      lg: string;
      xl: string;
      '2xl': string;
    };
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
} 
/**
 * Shared file upload utilities for RAG-Tobi Frontend
 * Extracted from manage page for reuse across upload interfaces
 */

export interface FileValidation {
  isValid: boolean;
  error?: string;
}

export interface FileUploadConfig {
  maxSize: number; // in bytes
  allowedTypes: string[];
  allowedExtensions: string[];
  maxSizeLabel: string; // human readable, e.g., "10MB"
}

// Default configurations for different upload types
export const FILE_UPLOAD_CONFIGS = {
  // General document upload (from manage page)
  general: {
    maxSize: 50 * 1024 * 1024, // 50MB
    allowedTypes: [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'text/markdown',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ],
    allowedExtensions: ['.pdf', '.doc', '.docx', '.txt', '.md', '.xls', '.xlsx'],
    maxSizeLabel: '50MB'
  },
  
  // Vehicle specification upload (more restrictive)
  vehicleSpec: {
    maxSize: 10 * 1024 * 1024, // 10MB
    allowedTypes: [
      'text/plain',
      'text/markdown',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/pdf'
    ],
    allowedExtensions: ['.txt', '.md', '.doc', '.docx', '.pdf'],
    maxSizeLabel: '10MB'
  }
} as const;

/**
 * Comprehensive file validation function extracted from manage page
 * Validates both MIME type and file extension for better security
 */
export function validateFile(file: File, config: FileUploadConfig = FILE_UPLOAD_CONFIGS.general): FileValidation {
  // Check file size
  if (file.size > config.maxSize) {
    return {
      isValid: false,
      error: `File size must be less than ${config.maxSizeLabel}`
    };
  }
  
  // Check file extension
  const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
  if (!config.allowedExtensions.includes(fileExtension)) {
    const extensionList = config.allowedExtensions.join(', ');
    return {
      isValid: false,
      error: `File type not supported. Allowed extensions: ${extensionList}`
    };
  }
  
  // Check MIME type (if provided by browser)
  if (file.type && !config.allowedTypes.includes(file.type)) {
    const typeDescription = getFileTypeDescription(config.allowedExtensions);
    return {
      isValid: false,
      error: `File type not supported. Please upload ${typeDescription}.`
    };
  }
  
  return { isValid: true };
}

/**
 * Generate human-readable file type description from extensions
 */
function getFileTypeDescription(extensions: string[]): string {
  const typeMap: Record<string, string> = {
    '.pdf': 'PDF',
    '.doc': 'Word',
    '.docx': 'Word',
    '.txt': 'text',
    '.md': 'Markdown',
    '.xls': 'Excel',
    '.xlsx': 'Excel'
  };
  
  const types = [...new Set(extensions.map(ext => typeMap[ext] || ext))];
  
  if (types.length === 1) {
    return `${types[0]} files`;
  } else if (types.length === 2) {
    return `${types[0]} and ${types[1]} files`;
  } else {
    return `${types.slice(0, -1).join(', ')}, and ${types[types.length - 1]} files`;
  }
}

/**
 * Standard drag and drop event handlers
 * Reusable across different upload interfaces
 */
export const createDragHandlers = (
  onFileSelect: (file: File) => void,
  setIsDragOver?: (isDragOver: boolean) => void,
  onValidationError?: (error: string) => void
) => {
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver?.(true);
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver?.(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver?.(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver?.(false);
    
    const files = Array.from(e.dataTransfer.files);
    
    if (files.length > 1) {
      onValidationError?.('Please upload only one file at a time.');
      return;
    }
    
    if (files.length === 1) {
      onFileSelect(files[0]);
    }
  };

  return {
    handleDragOver,
    handleDragEnter,
    handleDragLeave,
    handleDrop
  };
};

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Generate accept attribute for file input from config
 */
export function getAcceptAttribute(config: FileUploadConfig): string {
  return config.allowedExtensions.join(',');
}

/**
 * Create a standardized file selection handler
 */
export const createFileSelectHandler = (
  config: FileUploadConfig,
  onValidFile: (file: File) => void,
  onValidationError: (error: string) => void,
  onClearError?: () => void
) => {
  return (file: File) => {
    onClearError?.();
    
    const validation = validateFile(file, config);
    if (!validation.isValid) {
      onValidationError(validation.error!);
      return;
    }
    
    onValidFile(file);
  };
};

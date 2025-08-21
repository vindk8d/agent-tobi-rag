'use client';

import Link from 'next/link';
import { useState, useEffect, useRef, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import { 
  validateFile, 
  createDragHandlers, 
  createFileSelectHandler,
  FILE_UPLOAD_CONFIGS,
  formatFileSize,
  getAcceptAttribute,
  type FileValidation 
} from '@/lib/fileUtils';

// ============================================================================
// TYPES AND INTERFACES
// ============================================================================

interface Vehicle {
  id: string;
  brand: string;
  model: string;
  year: number;
  type: string;
  variant?: string;
  key_features?: string;
  is_available: boolean;
}

interface VehicleSpecificationResponse {
  exists: boolean;
  document?: {
    id: string;
    original_filename: string;
    created_at: string;
    file_size?: number;
    chunk_count?: number;
    embedding_count?: number;
  };
}



interface UploadProgress {
  vehicleId: string;
  fileName: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  chunkCount?: number;
  embeddingCount?: number;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Use shared vehicle spec upload configuration
const uploadConfig = FILE_UPLOAD_CONFIGS.vehicleSpec;

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function VehicleSpecUploadPage() {
  // State management
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(null);
  const [existingSpec, setExistingSpec] = useState<VehicleSpecificationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // ============================================================================
  // DATA FETCHING
  // ============================================================================

  // Load vehicles on component mount
  useEffect(() => {
    const fetchVehicles = async () => {
      try {
        const response = await apiClient.getVehicles();
        if (response.success && response.data) {
          setVehicles(response.data.vehicles || []);
        }
      } catch (err) {
        console.error('Error fetching vehicles:', err);
        setError('Failed to load vehicles');
      } finally {
        setLoading(false);
      }
    };

    fetchVehicles();
  }, []);

  // Check for existing specification when vehicle is selected
  useEffect(() => {
    const checkExistingSpec = async () => {
      if (!selectedVehicle) {
        setExistingSpec(null);
        return;
      }

      try {
        const response = await apiClient.getVehicleSpecification(selectedVehicle.id);
        setExistingSpec(response.data);
      } catch (err) {
        console.error('Error checking existing specification:', err);
        setExistingSpec(null);
      }
    };

    checkExistingSpec();
  }, [selectedVehicle]);

  // ============================================================================
  // FILE UPLOAD HANDLERS
  // ============================================================================

  // Create file selection handler using shared utility
  const handleFileSelect = useCallback(
    createFileSelectHandler(
      uploadConfig,
      (file: File) => {
        if (!selectedVehicle) {
          setError('Please select a vehicle first');
          return;
        }

        // Check if vehicle already has a specification
        if (existingSpec?.exists) {
          setPendingFile(file);
          setShowConfirmDialog(true);
        } else {
          uploadFile(file);
        }
      },
      (error: string) => {
        if (!selectedVehicle) {
          setError('Please select a vehicle first');
        } else {
          setError(error);
        }
      },
      () => setError(null)
    ),
    [selectedVehicle, existingSpec]
  );

  const uploadFile = async (file: File) => {
    if (!selectedVehicle) return;

    setUploadProgress({
      vehicleId: selectedVehicle.id,
      fileName: file.name,
      progress: 0,
      status: 'uploading'
    });

    try {
      // Upload the file
      const uploadResponse = await apiClient.uploadVehicleSpecification(
        selectedVehicle.id,
        file,
        (progress) => {
          setUploadProgress(prev => prev ? { ...prev, progress } : null);
        }
      );

      if (uploadResponse.success) {
        setUploadProgress(prev => prev ? { 
          ...prev, 
          progress: 100, 
          status: 'processing' 
        } : null);

        // Poll for processing completion and get counts
        let attempts = 0;
        const maxAttempts = 30; // 30 seconds max wait
        const pollInterval = 1000; // 1 second

        const pollForCompletion = async () => {
          try {
            const specResponse = await apiClient.getVehicleSpecification(selectedVehicle.id);
            const doc = specResponse.data?.document;
            
            // Check if processing is complete (has chunk and embedding counts)
            if (doc && doc.chunk_count !== undefined && doc.embedding_count !== undefined) {
              setUploadProgress(prev => prev ? { 
                ...prev, 
                status: 'completed',
                chunkCount: doc.chunk_count,
                embeddingCount: doc.embedding_count
              } : null);
              
              setExistingSpec(specResponse.data);
              
              // Clear upload progress after showing results
              setTimeout(() => {
                setUploadProgress(null);
              }, 5000);
              
              return;
            }
            
            // Continue polling if not complete and within attempt limit
            attempts++;
            if (attempts < maxAttempts) {
              setTimeout(pollForCompletion, pollInterval);
            } else {
              // Timeout - show completed but without counts
              setUploadProgress(prev => prev ? { 
                ...prev, 
                status: 'completed'
              } : null);
              
              setExistingSpec(specResponse.data);
              
              setTimeout(() => {
                setUploadProgress(null);
              }, 3000);
            }
          } catch (error) {
            console.error('Error polling for completion:', error);
            // Fallback to completed status
            setUploadProgress(prev => prev ? { 
              ...prev, 
              status: 'completed'
            } : null);
            
            setTimeout(() => {
              setUploadProgress(null);
            }, 3000);
          }
        };

        // Start polling after a short delay
        setTimeout(pollForCompletion, 2000);
      } else {
        throw new Error(uploadResponse.error || 'Upload failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setUploadProgress(prev => prev ? {
        ...prev,
        status: 'error',
        error: err instanceof Error ? err.message : 'Upload failed'
      } : null);
    }
  };

  const handleConfirmOverwrite = () => {
    if (pendingFile) {
      uploadFile(pendingFile);
      setPendingFile(null);
    }
    setShowConfirmDialog(false);
  };

  const handleCancelOverwrite = () => {
    setPendingFile(null);
    setShowConfirmDialog(false);
  };

  // ============================================================================
  // DRAG AND DROP HANDLERS
  // ============================================================================

  // Create drag handlers using shared utility
  const [isDragOver, setIsDragOver] = useState(false);
  const dragHandlers = createDragHandlers(
    handleFileSelect,
    setIsDragOver,
    setError
  );

  // ============================================================================
  // RENDER HELPERS
  // ============================================================================

  const renderVehicleDropdown = () => (
    <div className="mb-6">
      <label htmlFor="vehicle-select" className="block text-sm font-medium text-gray-700 mb-2">
        Select Vehicle
      </label>
      <select
        id="vehicle-select"
        value={selectedVehicle?.id || ''}
        onChange={(e) => {
          const vehicle = vehicles.find(v => v.id === e.target.value);
          setSelectedVehicle(vehicle || null);
        }}
        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        <option value="">Choose a vehicle...</option>
        {vehicles.map((vehicle) => (
          <option key={vehicle.id} value={vehicle.id}>
            {vehicle.brand} {vehicle.model} {vehicle.year} {vehicle.variant ? `(${vehicle.variant})` : ''}
          </option>
        ))}
      </select>
    </div>
  );

  const renderExistingSpecStatus = () => {
    if (!selectedVehicle || !existingSpec?.exists) return null;

    const doc = existingSpec.document;
    return (
      <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3 flex-1">
            <h3 className="text-sm font-medium text-yellow-800">
              Existing Specification Found
            </h3>
            <div className="mt-2 text-sm text-yellow-700">
              <p>
                This vehicle already has a specification document: <strong>{doc?.original_filename}</strong>
              </p>
              <p className="mt-1">
                Uploaded: {doc?.created_at ? new Date(doc.created_at).toLocaleDateString() : 'Unknown'}
              </p>
              
              {/* Processing Stats */}
              {(doc?.chunk_count !== undefined || doc?.embedding_count !== undefined) && (
                <div className="mt-2 flex items-center space-x-4 text-xs">
                  {doc?.chunk_count !== undefined && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full bg-yellow-100 text-yellow-800">
                      📄 {doc.chunk_count} chunks
                    </span>
                  )}
                  {doc?.embedding_count !== undefined && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full bg-yellow-100 text-yellow-800">
                      🔗 {doc.embedding_count} embeddings
                    </span>
                  )}
                </div>
              )}
              
              <p className="mt-2 text-xs">
                Uploading a new file will replace the existing specification.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderUploadArea = () => (
    <div
      ref={dropZoneRef}
      onDragOver={dragHandlers.handleDragOver}
      onDragEnter={dragHandlers.handleDragEnter}
      onDragLeave={dragHandlers.handleDragLeave}
      onDrop={dragHandlers.handleDrop}
      className={`relative border-2 border-dashed rounded-lg p-6 text-center ${
        isDragOver 
          ? 'border-blue-400 bg-blue-50' 
          : selectedVehicle 
            ? 'border-gray-300 hover:border-gray-400' 
            : 'border-gray-200 bg-gray-50'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={getAcceptAttribute(uploadConfig)}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileSelect(file);
        }}
        className="hidden"
        disabled={!selectedVehicle}
      />

      <div className="space-y-2">
        <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        
        <div className="text-gray-600">
          <p className="text-lg font-medium">
            {selectedVehicle ? 'Drop your specification file here' : 'Select a vehicle first'}
          </p>
          <p className="text-sm">
            {selectedVehicle ? 'or click to browse' : 'Choose a vehicle from the dropdown above'}
          </p>
        </div>

        {selectedVehicle && (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Choose File
          </button>
        )}

        <p className="text-xs text-gray-500">
          Supported formats: {uploadConfig.allowedExtensions.join(', ').toUpperCase()} (max {uploadConfig.maxSizeLabel})
        </p>
      </div>
    </div>
  );

  const renderUploadProgress = () => {
    if (!uploadProgress) return null;

    return (
      <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-blue-900">
            {uploadProgress.fileName}
          </span>
          <span className="text-sm text-blue-700">
            {uploadProgress.status === 'uploading' && `${uploadProgress.progress}%`}
            {uploadProgress.status === 'processing' && 'Processing...'}
            {uploadProgress.status === 'completed' && 'Completed!'}
            {uploadProgress.status === 'error' && 'Failed'}
          </span>
        </div>
        
        {uploadProgress.status === 'uploading' && (
          <div className="w-full bg-blue-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress.progress}%` }}
            />
          </div>
        )}

        {/* Processing Results */}
        {uploadProgress.status === 'completed' && (uploadProgress.chunkCount !== undefined || uploadProgress.embeddingCount !== undefined) && (
          <div className="mt-3 flex items-center space-x-4 text-xs">
            {uploadProgress.chunkCount !== undefined && (
              <span className="inline-flex items-center px-2 py-1 rounded-full bg-green-100 text-green-800">
                📄 {uploadProgress.chunkCount} chunks created
              </span>
            )}
            {uploadProgress.embeddingCount !== undefined && (
              <span className="inline-flex items-center px-2 py-1 rounded-full bg-green-100 text-green-800">
                🔗 {uploadProgress.embeddingCount} embeddings generated
              </span>
            )}
          </div>
        )}

        {uploadProgress.status === 'error' && uploadProgress.error && (
          <p className="text-sm text-red-600 mt-2">{uploadProgress.error}</p>
        )}
      </div>
    );
  };

  const renderConfirmDialog = () => {
    if (!showConfirmDialog || !pendingFile || !existingSpec?.document) return null;

    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
        <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
          <div className="mt-3 text-center">
            <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-yellow-100">
              <svg className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h3 className="text-lg leading-6 font-medium text-gray-900 mt-4">
              Replace Existing Specification?
            </h3>
            <div className="mt-2 px-7 py-3">
              <p className="text-sm text-gray-500">
                This will replace the existing specification "{existingSpec.document.original_filename}" 
                for {selectedVehicle?.brand} {selectedVehicle?.model}.
              </p>
              <p className="text-sm text-gray-500 mt-2">
                The old file will be backed up for 30 days before permanent deletion.
              </p>
            </div>
            <div className="items-center px-4 py-3">
              <button
                onClick={handleConfirmOverwrite}
                className="px-4 py-2 bg-red-600 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300 mb-2"
              >
                Replace Specification
              </button>
              <button
                onClick={handleCancelOverwrite}
                className="px-4 py-2 bg-gray-300 text-gray-700 text-base font-medium rounded-md w-full shadow-sm hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-300"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading vehicles...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Vehicle Specification Upload
              </h1>
              <p className="text-gray-600">
                Upload and manage vehicle specification documents for RAG processing
              </p>
            </div>
            <Link
              href="/dev"
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              <span>Back to Dev Tools</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setError(null)}
                  className="inline-flex text-red-400 hover:text-red-600"
                >
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Upload Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            Upload Vehicle Specification
          </h2>

          {/* Vehicle Selection */}
          {renderVehicleDropdown()}

          {/* Existing Specification Status */}
          {renderExistingSpecStatus()}

          {/* Upload Area */}
          {renderUploadArea()}

          {/* Upload Progress */}
          {renderUploadProgress()}

          {/* Instructions */}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
            <h3 className="text-sm font-medium text-blue-900 mb-2">
              📋 Upload Instructions
            </h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Select a vehicle from the dropdown menu</li>
              <li>• Upload one specification document per vehicle</li>
              <li>• Supported formats: Text, Markdown, Word documents, PDF</li>
              <li>• Documents will be processed using section-based chunking</li>
              <li>• Each section will be embedded separately for better RAG performance</li>
              <li>• Uploading a new file will replace any existing specification</li>
            </ul>
          </div>
        </div>

        {/* Vehicle List */}
        {vehicles.length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Available Vehicles ({vehicles.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {vehicles.map((vehicle) => (
                <div
                  key={vehicle.id}
                  className={`p-4 border rounded-md cursor-pointer transition-colors ${
                    selectedVehicle?.id === vehicle.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedVehicle(vehicle)}
                >
                  <h3 className="font-medium text-gray-900">
                    {vehicle.brand} {vehicle.model}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {vehicle.year} • {vehicle.type}
                  </p>
                  {vehicle.variant && (
                    <p className="text-sm text-gray-500">{vehicle.variant}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Confirmation Dialog */}
      {renderConfirmDialog()}
    </div>
  );
}

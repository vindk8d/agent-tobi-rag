"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// ============================================================================
// ACTIVE INTERFACES AND TYPES
// ============================================================================

interface DataSource {
  id: string;
  name: string;
  description?: string;
  url?: string;
  file_path?: string;
  status: string;
  document_type: string;
  chunk_count: number;
  scraping_frequency?: string;
  last_scraped_at?: string;
  created_at: string;
  updated_at: string;
}

interface FileValidation {
  isValid: boolean;
  error?: string;
}

// Simple browser-safe UUID generator
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// File validation
function validateFile(file: File): FileValidation {
  const maxSize = 50 * 1024 * 1024; // 50MB
  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'text/markdown',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  ];
  
  const allowedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.xls', '.xlsx'];
  
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: 'File size must be less than 50MB'
    };
  }
  
  const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
  if (!allowedExtensions.includes(fileExtension)) {
    return {
      isValid: false,
      error: 'File type not supported. Please upload PDF, Word, Excel, or text files.'
    };
  }
  
  if (!allowedTypes.includes(file.type) && file.type !== '') {
    return {
      isValid: false,
      error: 'File type not supported. Please upload PDF, Word, Excel, or text files.'
    };
  }
  
  return { isValid: true };
}

export default function ManagePage() {
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [dataSourcesLoading, setDataSourcesLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [processingDataSourceId, setProcessingDataSourceId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Add delete state
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Fetch data sources with chunk counts
  useEffect(() => {
    async function fetchDataSources() {
      setDataSourcesLoading(true);
      try {
        const { data: dataSourcesData, error: dataSourcesError } = await supabase
          .from("data_sources")
          .select("id, name, description, url, file_path, status, document_type, chunk_count, scraping_frequency, last_scraped_at, created_at, updated_at")
          .order("created_at", { ascending: false });

        if (dataSourcesError) {
          console.error("Error fetching data sources:", dataSourcesError);
          setDataSources([]);
        } else {
          setDataSources(dataSourcesData || []);
        }
      } catch (error) {
        console.error("Error fetching data sources:", error);
        setDataSources([]);
      }
      setDataSourcesLoading(false);
    }
    fetchDataSources();
  }, [refresh]);

  // Real-time updates for data sources
  useEffect(() => {
    const channel = supabase
      .channel('data-sources-changes')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'data_sources' }, 
        (payload) => {
          console.log('Data source changed:', payload);
          
          // Check if this is the document we're currently processing
          if (processingDataSourceId && payload.new && typeof payload.new === 'object') {
            const newRecord = payload.new as any;
            
            if (newRecord.id === processingDataSourceId) {
              const newStatus = newRecord.status;
              const newChunkCount = newRecord.chunk_count;
              const documentName = newRecord.name;
              
              // If processing is complete (status is 'active' and has chunks)
              if (newStatus === 'active' && newChunkCount > 0) {
                setUploadStatus(`Processing complete! Document "${documentName}" has been successfully processed with ${newChunkCount} chunks.`);
                setProcessingDataSourceId(null);
                
                // Clear the completion message after 5 seconds
                setTimeout(() => {
                  setUploadStatus(null);
                }, 5000);
              }
              // If processing failed
              else if (newStatus === 'failed') {
                setUploadStatus(`Processing failed for document "${documentName}". Please try again.`);
                setProcessingDataSourceId(null);
                
                // Clear the error message after 8 seconds
                setTimeout(() => {
                  setUploadStatus(null);
                }, 8000);
              }
            }
          }
          
          // Refresh data sources when changes occur
          setRefresh((r) => r + 1);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [processingDataSourceId]);

  // Auto-refresh timer for periodic updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRefresh((r) => r + 1);
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback((selectedFile: File) => {
    setValidationError(null);
    setUploadStatus(null);
    setProcessingDataSourceId(null);
    
    const validation = validateFile(selectedFile);
    if (!validation.isValid) {
      setValidationError(validation.error!);
      setFile(null);
      return;
    }
    
    setFile(selectedFile);
  }, []);

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 1) {
      setValidationError('Please upload only one file at a time.');
      return;
    }
    
    if (files.length === 1) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  // Handle file upload with progress
  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    setUploadProgress(0);
    
    try {
      const fileExt = file.name.split(".").pop();
      const filePath = `${uuidv4()}.${fileExt}`;
      
      // Simulate upload progress (Supabase doesn't provide real progress callbacks)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);
      
      const { data, error } = await supabase.storage
        .from("documents")
        .upload(filePath, file);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      if (error) {
        console.error("Upload error:", error);
        if (error.message.includes("Bucket not found")) {
          setUploadStatus(
            "Upload failed: Documents bucket not found. Please create a 'documents' bucket in your Supabase Storage dashboard first."
          );
        } else {
          setUploadStatus("Upload failed: " + error.message);
        }
        setUploading(false);
        setUploadProgress(0);
        return;
      }
      
      try {
        const response = await fetch("http://localhost:8000/api/v1/data-sources/process-uploaded", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_path: data.path,
            file_name: file.name,
            file_size: file.size,
          }),
        });

        if (response.ok) {
          const result = await response.json();
          
          // Set the processing data source ID to track completion
          if (result.data_source_id) {
            setProcessingDataSourceId(result.data_source_id);
          }
          
          setUploadStatus(
            `Upload successful! Document "${file.name}" is being processed. Processing will complete in the background.`
          );
          setTimeout(() => setRefresh((r) => r + 1), 2000);
        } else {
          const errorData = await response.json();
          setUploadStatus(
            `Upload successful, but processing failed: ${errorData.detail || 'Unknown error'}`
          );
        }
      } catch (processError) {
        setUploadStatus(
          `Upload successful, but couldn't start processing. Check if backend is running.`
        );
      }
      
      setFile(null);
      setUploadProgress(0);
      if (fileInputRef.current) fileInputRef.current.value = "";
      
    } catch (e: any) {
      setUploadStatus("Upload error: " + e.message);
      setUploadProgress(0);
    }
    setUploading(false);
  }

  function formatDate(dateString: string) {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Handle data source deletion
  async function handleDeleteDataSource(dataSourceId: string, name: string) {
    if (!confirm(`Are you sure you want to delete "${name}"? This will also delete all related chunks and embeddings. This action cannot be undone.`)) {
      return;
    }

    setDeletingId(dataSourceId);
    setUploadStatus(null);

    try {
      const response = await fetch(`http://localhost:8000/api/v1/data-sources/${dataSourceId}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setUploadStatus(`Document "${name}" deleted successfully`);
        
        // Clear processing state if the deleted document was being processed
        if (processingDataSourceId === dataSourceId) {
          setProcessingDataSourceId(null);
        }
        
        // Clear the success message after 3 seconds
        setTimeout(() => {
          setUploadStatus(null);
        }, 3000);
        
        setRefresh((r) => r + 1);
      } else {
        const errorData = await response.json();
        setUploadStatus(
          `Failed to delete document: ${errorData.detail || 'Unknown error'}`
        );
      }
    } catch (error: any) {
      setUploadStatus(
        `Failed to delete document: ${error.message || 'Network error'}`
      );
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="max-w-6xl mx-auto py-8 px-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Document Management</h1>
        <button
          onClick={() => setRefresh((r) => r + 1)}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Refresh
        </button>
      </div>

      {/* Enhanced Document Upload Section */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-4">Upload Documents</h2>
        <div className="bg-white p-6 rounded-lg shadow">
          {/* Drag and Drop Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragOver
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              <div className="flex justify-center">
                <svg
                  className={`w-12 h-12 ${isDragOver ? 'text-blue-500' : 'text-gray-400'}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900">
                  Drop files here or <span className="text-blue-600">click to browse</span>
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Supports PDF, Word documents, Excel files, and text files (max 50MB)
                </p>
              </div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => {
                  const selectedFile = e.target.files?.[0];
                  if (selectedFile) {
                    handleFileSelect(selectedFile);
                  }
                }}
                accept=".pdf,.doc,.docx,.txt,.md,.xls,.xlsx"
                className="hidden"
                aria-label="File upload"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Select File
              </button>
            </div>
          </div>

          {/* Selected File Info */}
          {file && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-blue-600 font-semibold text-sm">📄</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setFile(null);
                    setValidationError(null);
                    setUploadStatus(null);
                    setProcessingDataSourceId(null);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {/* Upload Progress Bar */}
          {uploading && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Uploading...</span>
                <span className="text-sm text-gray-500">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-in-out"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Upload Button */}
          <div className="mt-4">
            <button
              onClick={handleUpload}
              disabled={uploading || !file || !!validationError}
              className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {uploading ? "Uploading..." : "Upload & Process Document"}
            </button>
          </div>

          {/* Status Messages */}
          {validationError && (
            <div className="mt-4 p-3 rounded-lg bg-red-50 text-red-700 border border-red-200">
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {validationError}
              </div>
            </div>
          )}

          {uploadStatus && (
            <div className={`mt-4 p-3 rounded-lg ${
              uploadStatus.includes("failed") || uploadStatus.includes("error") 
                ? "bg-red-50 text-red-700 border border-red-200" 
                : "bg-green-50 text-green-700 border border-green-200"
            }`}>
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                    d={uploadStatus.includes("failed") || uploadStatus.includes("error") 
                      ? "M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      : "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    } 
                  />
                </svg>
                {uploadStatus}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Processed Documents List */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-4">Processed Documents</h2>
        {dataSourcesLoading ? (
          <div className="text-center py-8">Loading data sources...</div>
        ) : dataSources.length === 0 ? (
          <div className="text-center py-8 text-gray-500">No documents uploaded yet.</div>
        ) : (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Document
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Chunks
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {dataSources.map((ds) => (
                    <tr key={ds.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <div className="font-medium text-gray-900">{ds.name}</div>
                        <div className="text-sm text-gray-500 break-all">
                          {ds.file_path || ds.url || 'N/A'}
                        </div>
                        {ds.description && (
                          <div className="text-sm text-gray-600 mt-1">{ds.description}</div>
                        )}
                      </td>
                      <td className="px-6 py-4">
                        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                          {ds.document_type}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          ds.status === 'active' 
                            ? 'bg-green-100 text-green-800' 
                            : ds.status === 'error' || ds.status === 'failed'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {ds.status}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          ds.chunk_count > 0 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {ds.chunk_count} chunks
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        {formatDate(ds.created_at)}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleDeleteDataSource(ds.id, ds.name)}
                            disabled={deletingId === ds.id}
                            className="inline-flex items-center px-3 py-1 border border-transparent text-xs leading-4 font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 disabled:opacity-50"
                          >
                            {deletingId === ds.id ? 'Deleting...' : 'Delete'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

// ============================================================================
// DEPRIORITIZED CODE - Website scraping functionality has been deprioritized
// ============================================================================

// interface DataSource {
//   id: string;
//   name: string;
//   description?: string;
//   url: string;
//   status: string;
//   scraping_frequency: string;
//   last_scraped_at?: string;
//   last_success?: string;
//   document_count: number;
//   error_count: number;
//   last_error?: string;
//   created_at: string;
// }

// interface DataSourceForm {
//   name: string;
//   description: string;
//   url: string;
//   scraping_frequency: string;
// }

// // State variables for data sources (commented out)
// // const [dataSources, setDataSources] = useState<DataSource[]>([]);
// // const [loading, setLoading] = useState(false);
// // const [showDataSourceForm, setShowDataSourceForm] = useState(false);
// // const [dataSourceForm, setDataSourceForm] = useState<DataSourceForm>({
// //   name: '',
// //   description: '',
// //   url: '',
// //   scraping_frequency: 'daily'
// // });
// // const [dataSourceSubmitting, setDataSourceSubmitting] = useState(false);
// // const [dataSourceStatus, setDataSourceStatus] = useState<string | null>(null);
// // const [testingUrl, setTestingUrl] = useState(false);
// // const [stats, setStats] = useState<any>(null);

// // Fetch data sources from Supabase (commented out)
// // useEffect(() => {
// //   async function fetchDataSources() {
// //     setLoading(true);
// //     try {
// //       const { data, error } = await supabase
// //         .from("data_sources")
// //         .select("*")
// //         .order("created_at", { ascending: false });
// //       
// //       if (error) {
// //         console.error("Error fetching data sources:", error);
// //         setDataSources([]);
// //       } else {
// //         setDataSources(data || []);
// //       }
// //     } catch (error) {
// //       console.error("Error fetching data sources:", error);
// //       setDataSources([]);
// //     }
// //     setLoading(false);
// //   }
// //   fetchDataSources();
// // }, [refresh]);

// // Fetch statistics (commented out)
// // useEffect(() => {
// //   async function fetchStats() {
// //     try {
// //       const response = await fetch("http://localhost:8000/api/v1/datasources/stats/overview");
// //       if (response.ok) {
// //         const result = await response.json();
// //         setStats(result.data);
// //       }
// //     } catch (error) {
// //       console.error("Error fetching stats:", error);
// //     }
// //   }
// //   fetchStats();
// // }, [refresh]);

// // Real-time updates for data sources (commented out)
// // useEffect(() => {
// //   const channel = supabase
// //     .channel('data-sources-changes')
// //     .on('postgres_changes', 
// //       { event: '*', schema: 'public', table: 'data_sources' }, 
// //       (payload) => {
// //         console.log('Data source changed:', payload);
// //         // Refresh data sources when changes occur
// //         setRefresh((r) => r + 1);
// //       }
// //     )
// //     .subscribe();

// //   return () => {
// //     supabase.removeChannel(channel);
// //   };
// // }, []);

// // Handle data source creation (commented out)
// // async function handleDataSourceSubmit(e: React.FormEvent) {
// //   e.preventDefault();
// //   setDataSourceSubmitting(true);
// //   setDataSourceStatus(null);
// //   
// //   try {
// //     const response = await fetch("http://localhost:8000/api/v1/datasources/", {
// //       method: "POST",
// //       headers: {
// //         "Content-Type": "application/json",
// //       },
// //       body: JSON.stringify(dataSourceForm),
// //     });

// //     if (response.ok) {
// //       const result = await response.json();
// //       setDataSourceStatus(
// //         `Data source "${dataSourceForm.name}" created successfully! Testing scraping capability...`
// //       );
// //       setDataSourceForm({
// //         name: '',
// //         description: '',
// //         url: '',
// //         scraping_frequency: 'daily'
// //       });
// //       setShowDataSourceForm(false);
// //       setTimeout(() => setRefresh((r) => r + 1), 2000);
// //     } else {
// //       const errorData = await response.json();
// //       setDataSourceStatus(
// //         `Failed to create data source: ${errorData.detail || 'Unknown error'}`
// //       );
// //     }
// //   } catch (error) {
// //     setDataSourceStatus(
// //       `Failed to create data source: ${error.message || 'Network error'}`
// //     );
// //   }
// //   setDataSourceSubmitting(false);
// // }

// // Test URL functionality (commented out)
// // async function testUrl() {
// //   if (!dataSourceForm.url) return;
// //   setTestingUrl(true);
// //   
// //   try {
// //     const response = await fetch("http://localhost:8000/api/v1/datasources/test", {
// //       method: "POST",
// //       headers: {
// //         "Content-Type": "application/json",
// //       },
// //       body: JSON.stringify({ url: dataSourceForm.url }),
// //     });

// //     if (response.ok) {
// //       const result = await response.json();
// //       const testResult = result.data.test_result;
// //       if (testResult.success) {
// //         setDataSourceStatus(
// //           `✅ URL test successful! Found ${testResult.content_length} characters of content. Title: "${testResult.title}"`
// //         );
// //       } else {
// //         setDataSourceStatus(
// //           `❌ URL test failed: ${testResult.error}`
// //         );
// //       }
// //     } else {
// //       const errorData = await response.json();
// //       setDataSourceStatus(
// //         `URL test failed: ${errorData.detail || 'Unknown error'}`
// //       );
// //     }
// //   } catch (error) {
// //     setDataSourceStatus(
// //       `URL test failed: ${error.message || 'Network error'}`
// //     );
// //   }
// //   setTestingUrl(false);
// // }

// // Delete data source (commented out)
// // async function deleteDataSource(dataSourceId: string, name: string) {
// //   if (!confirm(`Are you sure you want to delete "${name}"? This will also delete all related documents.`)) {
// //     return;
// //   }

// //   try {
// //     const response = await fetch(`http://localhost:8000/api/v1/datasources/${dataSourceId}`, {
// //       method: "DELETE",
// //     });

// //     if (response.ok) {
// //       setDataSourceStatus(`Data source "${name}" deleted successfully`);
// //       setRefresh((r) => r + 1);
// //     } else {
// //       const errorData = await response.json();
// //       setDataSourceStatus(
// //         `Failed to delete data source: ${errorData.detail || 'Unknown error'}`
// //       );
// //     }
// //   } catch (error) {
// //     setDataSourceStatus(
// //       `Failed to delete data source: ${error.message || 'Network error'}`
// //     );
// //   }
// // }

// // Trigger manual scrape (commented out)
// // async function triggerScrape(dataSourceId: string, name: string) {
// //   try {
// //     const response = await fetch(`http://localhost:8000/api/v1/datasources/${dataSourceId}/scrape`, {
// //       method: "POST",
// //     });

// //     if (response.ok) {
// //       setDataSourceStatus(`Manual scraping triggered for "${name}"`);
// //       setTimeout(() => setRefresh((r) => r + 1), 3000);
// //     } else {
// //       const errorData = await response.json();
// //       setDataSourceStatus(
// //         `Failed to trigger scraping: ${errorData.detail || 'Unknown error'}`
// //       );
// //     }
// //   } catch (error) {
// //     setDataSourceStatus(
// //       `Failed to trigger scraping: ${error.message || 'Network error'}`
// //     );
// //   }
// // }

// // Status color helper (commented out)
// // function getStatusColor(status: string) {
// //   switch (status.toLowerCase()) {
// //     case 'active':
// //       return 'bg-green-100 text-green-700';
// //     case 'pending':
// //       return 'bg-yellow-100 text-yellow-700';
// //     case 'error':
// //       return 'bg-red-100 text-red-700';
// //     case 'inactive':
// //       return 'bg-gray-100 text-gray-700';
// //     default:
// //       return 'bg-gray-100 text-gray-700';
// //   }
// // }

// // JSX for Statistics Overview (commented out)
// // {stats && (
// //   <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
// //     <div className="bg-white p-4 rounded-lg shadow">
// //       <div className="text-2xl font-bold text-green-600">{stats.status_counts?.active || 0}</div>
// //       <div className="text-sm text-gray-600">Active Sources</div>
// //     </div>
// //     <div className="bg-white p-4 rounded-lg shadow">
// //       <div className="text-2xl font-bold text-yellow-600">{stats.status_counts?.pending || 0}</div>
// //       <div className="text-sm text-gray-600">Pending Sources</div>
// //     </div>
// //     <div className="bg-white p-4 rounded-lg shadow">
// //       <div className="text-2xl font-bold text-red-600">{stats.status_counts?.error || 0}</div>
// //       <div className="text-sm text-gray-600">Error Sources</div>
// //     </div>
// //     <div className="bg-white p-4 rounded-lg shadow">
// //       <div className="text-2xl font-bold text-blue-600">{stats.total_documents || 0}</div>
// //       <div className="text-sm text-gray-600">Total Documents</div>
// //     </div>
// //   </div>
// // )}

// // JSX for Status Messages (commented out)
// // {dataSourceStatus && (
// //   <div className={`mb-6 p-4 rounded-lg ${
// //     dataSourceStatus.includes("successful") || dataSourceStatus.includes("✅") 
// //       ? "bg-green-50 text-green-700 border border-green-200" 
// //       : dataSourceStatus.includes("failed") || dataSourceStatus.includes("❌") 
// //       ? "bg-red-50 text-red-700 border border-red-200"
// //       : "bg-blue-50 text-blue-700 border border-blue-200"
// //   }`}>
// //     {dataSourceStatus}
// //   </div>
// // )}

// // JSX for Data Source Management Section (commented out)
// // <section className="mb-10">
// //   <div className="flex justify-between items-center mb-4">
// //     <h2 className="text-xl font-semibold">Website Data Sources</h2>
// //     <button
// //       onClick={() => setShowDataSourceForm(!showDataSourceForm)}
// //       className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
// //     >
// //       {showDataSourceForm ? 'Cancel' : 'Add New Source'}
// //     </button>
// //   </div>

// //   {/* Data Source Creation Form */}
// //   {showDataSourceForm && (
// //     <div className="bg-white p-6 rounded-lg shadow mb-6">
// //       <h3 className="text-lg font-semibold mb-4">Add New Data Source</h3>
// //       <form onSubmit={handleDataSourceSubmit} className="space-y-4">
// //         <div>
// //           <label className="block text-sm font-medium text-gray-700 mb-1">
// //             Name *
// //           </label>
// //           <input
// //             type="text"
// //             value={dataSourceForm.name}
// //             onChange={(e) => setDataSourceForm(prev => ({ ...prev, name: e.target.value }))}
// //             className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
// //             placeholder="e.g., Company Blog"
// //             required
// //           />
// //         </div>
// //         
// //         <div>
// //           <label className="block text-sm font-medium text-gray-700 mb-1">
// //             Description
// //           </label>
// //           <input
// //             type="text"
// //             value={dataSourceForm.description}
// //             onChange={(e) => setDataSourceForm(prev => ({ ...prev, description: e.target.value }))}
// //             className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
// //             placeholder="Optional description"
// //           />
// //         </div>
// //         
// //         <div>
// //           <label className="block text-sm font-medium text-gray-700 mb-1">
// //             Website URL *
// //           </label>
// //           <div className="flex gap-2">
// //             <input
// //               type="url"
// //               value={dataSourceForm.url}
// //               onChange={(e) => setDataSourceForm(prev => ({ ...prev, url: e.target.value }))}
// //               className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
// //               placeholder="https://example.com"
// //               required
// //             />
// //             <button
// //               type="button"
// //               onClick={testUrl}
// //               disabled={!dataSourceForm.url || testingUrl}
// //               className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
// //             >
// //               {testingUrl ? 'Testing...' : 'Test URL'}
// //             </button>
// //           </div>
// //         </div>
// //         
// //         <div>
// //           <label className="block text-sm font-medium text-gray-700 mb-1">
// //             Scraping Frequency
// //           </label>
// //           <select
// //             value={dataSourceForm.scraping_frequency}
// //             onChange={(e) => setDataSourceForm(prev => ({ ...prev, scraping_frequency: e.target.value }))}
// //             className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
// //           >
// //             <option value="daily">Daily</option>
// //             <option value="weekly">Weekly</option>
// //             <option value="monthly">Monthly</option>
// //             <option value="manual">Manual Only</option>
// //           </select>
// //         </div>
// //         
// //         <div className="flex gap-2">
// //           <button
// //             type="submit"
// //             disabled={dataSourceSubmitting}
// //             className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
// //           >
// //             {dataSourceSubmitting ? 'Creating...' : 'Create Data Source'}
// //           </button>
// //           <button
// //             type="button"
// //             onClick={() => setShowDataSourceForm(false)}
// //             className="px-6 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
// //           >
// //             Cancel
// //           </button>
// //         </div>
// //       </form>
// //     </div>
// //   )}

// //   {/* Data Sources List */}
// //   {loading ? (
// //     <div className="text-center py-8">Loading data sources...</div>
// //   ) : dataSources.length === 0 ? (
// //     <div className="text-center py-8 text-gray-500">
// //       No data sources found. Add your first data source to get started!
// //     </div>
// //   ) : (
// //     <div className="bg-white rounded-lg shadow overflow-hidden">
// //       <div className="overflow-x-auto">
// //         <table className="min-w-full divide-y divide-gray-200">
// //           <thead className="bg-gray-50">
// //             <tr>
// //               <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
// //                 Data Source
// //               </th>
// //               <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
// //                 Status
// //               </th>
// //               <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
// //                 Documents
// //               </th>
// //               <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
// //                 Last Scraped
// //               </th>
// //               <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
// //                 Actions
// //               </th>
// //             </tr>
// //           </thead>
// //           <tbody className="bg-white divide-y divide-gray-200">
// //             {dataSources.map((ds) => (
// //               <tr key={ds.id} className="hover:bg-gray-50">
// //                 <td className="px-6 py-4">
// //                   <div className="flex flex-col">
// //                     <div className="font-medium text-gray-900">{ds.name}</div>
// //                     <div className="text-sm text-blue-600 break-all">{ds.url}</div>
// //                     {ds.description && (
// //                       <div className="text-sm text-gray-500">{ds.description}</div>
// //                     )}
// //                     <div className="text-xs text-gray-400 mt-1">
// //                       {ds.scraping_frequency} • Created {formatDate(ds.created_at)}
// //                     </div>
// //                   </div>
// //                 </td>
// //                 <td className="px-6 py-4">
// //                   <div className="flex flex-col gap-1">
// //                     <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(ds.status)}`}>
// //                       {ds.status}
// //                     </span>
// //                     {ds.error_count > 0 && (
// //                       <span className="text-xs text-red-600">
// //                         {ds.error_count} error(s)
// //                       </span>
// //                     )}
// //                     {ds.last_error && (
// //                       <span className="text-xs text-red-600 truncate" title={ds.last_error}>
// //                         {ds.last_error.length > 30 ? ds.last_error.substring(0, 30) + '...' : ds.last_error}
// //                       </span>
// //                     )}
// //                   </div>
// //                 </td>
// //                 <td className="px-6 py-4">
// //                   <div className="flex flex-col">
// //                     <span className="text-sm font-medium">{ds.document_count}</span>
// //                     <span className="text-xs text-gray-500">documents</span>
// //                   </div>
// //                 </td>
// //                 <td className="px-6 py-4">
// //                   <div className="flex flex-col text-sm text-gray-900">
// //                     <span>Scraped: {formatDate(ds.last_scraped_at)}</span>
// //                     <span className="text-xs text-gray-500">
// //                       Success: {formatDate(ds.last_success)}
// //                     </span>
// //                   </div>
// //                 </td>
// //                 <td className="px-6 py-4">
// //                   <div className="flex flex-col gap-1">
// //                     <button
// //                       onClick={() => triggerScrape(ds.id, ds.name)}
// //                       className="inline-flex items-center px-3 py-1 border border-transparent text-xs leading-4 font-medium rounded text-blue-700 bg-blue-100 hover:bg-blue-200"
// //                     >
// //                       Scrape Now
// //                     </button>
// //                     <button
// //                       onClick={() => deleteDataSource(ds.id, ds.name)}
// //                       className="inline-flex items-center px-3 py-1 border border-transparent text-xs leading-4 font-medium rounded text-red-700 bg-red-100 hover:bg-red-200"
// //                     >
// //                       Delete
// //                     </button>
// //                   </div>
// //                 </td>
// //               </tr>
// //             ))}
// //           </tbody>
// //         </table>
// //       </div>
// //     </div>
// //   )}
// // </section> 
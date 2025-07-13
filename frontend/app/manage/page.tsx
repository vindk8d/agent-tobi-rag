"use client";
import { useEffect, useState, useRef } from "react";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

interface DataSource {
  id: string;
  name: string;
  description?: string;
  url: string;
  status: string;
  scraping_frequency: string;
  last_scraped_at?: string;
  last_success?: string;
  document_count: number;
  error_count: number;
  last_error?: string;
  created_at: string;
}

interface Document {
  id: string;
  title: string;
  document_type: string;
  storage_path: string;
  created_at: string;
  embedding_count: number;
}

interface DataSourceForm {
  name: string;
  description: string;
  url: string;
  scraping_frequency: string;
}

// Simple browser-safe UUID generator
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

export default function ManagePage() {
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [showDataSourceForm, setShowDataSourceForm] = useState(false);
  const [dataSourceForm, setDataSourceForm] = useState<DataSourceForm>({
    name: '',
    description: '',
    url: '',
    scraping_frequency: 'daily'
  });
  const [dataSourceSubmitting, setDataSourceSubmitting] = useState(false);
  const [dataSourceStatus, setDataSourceStatus] = useState<string | null>(null);
  const [testingUrl, setTestingUrl] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch data sources from Supabase
  useEffect(() => {
    async function fetchDataSources() {
      setLoading(true);
      try {
        const { data, error } = await supabase
          .from("data_sources")
          .select("*")
          .order("created_at", { ascending: false });
        
        if (error) {
          console.error("Error fetching data sources:", error);
          setDataSources([]);
        } else {
          setDataSources(data || []);
        }
      } catch (error) {
        console.error("Error fetching data sources:", error);
        setDataSources([]);
      }
      setLoading(false);
    }
    fetchDataSources();
  }, [refresh]);

  // Fetch documents with embedding counts
  useEffect(() => {
    async function fetchDocuments() {
      setDocumentsLoading(true);
      try {
        const { data: documentsData, error: documentsError } = await supabase
          .from("documents")
          .select("id, title, document_type, storage_path, created_at, embedding_count")
          .order("created_at", { ascending: false });

        if (documentsError) {
          console.error("Error fetching documents:", documentsError);
          setDocuments([]);
        } else {
          setDocuments(documentsData || []);
        }
      } catch (error) {
        console.error("Error fetching documents:", error);
        setDocuments([]);
      }
      setDocumentsLoading(false);
    }
    fetchDocuments();
  }, [refresh]);

  // Fetch statistics
  useEffect(() => {
    async function fetchStats() {
      try {
        const response = await fetch("http://localhost:8000/api/v1/datasources/stats/overview");
        if (response.ok) {
          const result = await response.json();
          setStats(result.data);
        }
      } catch (error) {
        console.error("Error fetching stats:", error);
      }
    }
    fetchStats();
  }, [refresh]);

  // Real-time updates for data sources
  useEffect(() => {
    const channel = supabase
      .channel('data-sources-changes')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'data_sources' }, 
        (payload) => {
          console.log('Data source changed:', payload);
          // Refresh data sources when changes occur
          setRefresh((r) => r + 1);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  // Real-time updates for documents
  useEffect(() => {
    const channel = supabase
      .channel('documents-changes')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'documents' }, 
        (payload) => {
          console.log('Document changed:', payload);
          // Refresh documents when changes occur
          setRefresh((r) => r + 1);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  // Auto-refresh timer for periodic updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRefresh((r) => r + 1);
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Handle file upload
  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    try {
      const fileExt = file.name.split(".").pop();
      const filePath = `${uuidv4()}.${fileExt}`;
      const { data, error } = await supabase.storage
        .from("documents")
        .upload(filePath, file);
      
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
        return;
      }
      
      try {
        const response = await fetch("http://localhost:8000/api/v1/documents/process-uploaded", {
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
      if (fileInputRef.current) fileInputRef.current.value = "";
      
    } catch (e: any) {
      setUploadStatus("Upload error: " + e.message);
    }
    setUploading(false);
  }

  // Handle data source creation
  async function handleDataSourceSubmit(e: React.FormEvent) {
    e.preventDefault();
    setDataSourceSubmitting(true);
    setDataSourceStatus(null);
    
    try {
      const response = await fetch("http://localhost:8000/api/v1/datasources/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(dataSourceForm),
      });

      if (response.ok) {
        const result = await response.json();
        setDataSourceStatus(
          `Data source "${dataSourceForm.name}" created successfully! Testing scraping capability...`
        );
        setDataSourceForm({
          name: '',
          description: '',
          url: '',
          scraping_frequency: 'daily'
        });
        setShowDataSourceForm(false);
        setTimeout(() => setRefresh((r) => r + 1), 2000);
      } else {
        const errorData = await response.json();
        setDataSourceStatus(
          `Failed to create data source: ${errorData.detail || 'Unknown error'}`
        );
      }
    } catch (error) {
      setDataSourceStatus(
        `Failed to create data source: ${error.message || 'Network error'}`
      );
    }
    setDataSourceSubmitting(false);
  }

  // Test URL functionality
  async function testUrl() {
    if (!dataSourceForm.url) return;
    setTestingUrl(true);
    
    try {
      const response = await fetch("http://localhost:8000/api/v1/datasources/test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: dataSourceForm.url }),
      });

      if (response.ok) {
        const result = await response.json();
        const testResult = result.data.test_result;
        if (testResult.success) {
          setDataSourceStatus(
            `✅ URL test successful! Found ${testResult.content_length} characters of content. Title: "${testResult.title}"`
          );
        } else {
          setDataSourceStatus(
            `❌ URL test failed: ${testResult.error}`
          );
        }
      } else {
        const errorData = await response.json();
        setDataSourceStatus(
          `URL test failed: ${errorData.detail || 'Unknown error'}`
        );
      }
    } catch (error) {
      setDataSourceStatus(
        `URL test failed: ${error.message || 'Network error'}`
      );
    }
    setTestingUrl(false);
  }

  // Delete data source
  async function deleteDataSource(dataSourceId: string, name: string) {
    if (!confirm(`Are you sure you want to delete "${name}"? This will also delete all related documents.`)) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/api/v1/datasources/${dataSourceId}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setDataSourceStatus(`Data source "${name}" deleted successfully`);
        setRefresh((r) => r + 1);
      } else {
        const errorData = await response.json();
        setDataSourceStatus(
          `Failed to delete data source: ${errorData.detail || 'Unknown error'}`
        );
      }
    } catch (error) {
      setDataSourceStatus(
        `Failed to delete data source: ${error.message || 'Network error'}`
      );
    }
  }

  // Trigger manual scrape
  async function triggerScrape(dataSourceId: string, name: string) {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/datasources/${dataSourceId}/scrape`, {
        method: "POST",
      });

      if (response.ok) {
        setDataSourceStatus(`Manual scraping triggered for "${name}"`);
        setTimeout(() => setRefresh((r) => r + 1), 3000);
      } else {
        const errorData = await response.json();
        setDataSourceStatus(
          `Failed to trigger scraping: ${errorData.detail || 'Unknown error'}`
        );
      }
    } catch (error) {
      setDataSourceStatus(
        `Failed to trigger scraping: ${error.message || 'Network error'}`
      );
    }
  }

  function getStatusColor(status: string) {
    switch (status.toLowerCase()) {
      case 'active':
        return 'bg-green-100 text-green-700';
      case 'pending':
        return 'bg-yellow-100 text-yellow-700';
      case 'error':
        return 'bg-red-100 text-red-700';
      case 'inactive':
        return 'bg-gray-100 text-gray-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  }

  function formatDate(dateString: string) {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  }

  return (
    <div className="max-w-6xl mx-auto py-8 px-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Data Source Management</h1>
        <button
          onClick={() => setRefresh((r) => r + 1)}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Refresh All
        </button>
      </div>

      {/* Statistics Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white p-4 rounded-lg shadow">
            <div className="text-2xl font-bold text-green-600">{stats.status_counts?.active || 0}</div>
            <div className="text-sm text-gray-600">Active Sources</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <div className="text-2xl font-bold text-yellow-600">{stats.status_counts?.pending || 0}</div>
            <div className="text-sm text-gray-600">Pending Sources</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <div className="text-2xl font-bold text-red-600">{stats.status_counts?.error || 0}</div>
            <div className="text-sm text-gray-600">Error Sources</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <div className="text-2xl font-bold text-blue-600">{stats.total_documents || 0}</div>
            <div className="text-sm text-gray-600">Total Documents</div>
          </div>
        </div>
      )}

      {/* Status Messages */}
      {dataSourceStatus && (
        <div className={`mb-6 p-4 rounded-lg ${
          dataSourceStatus.includes("successful") || dataSourceStatus.includes("✅") 
            ? "bg-green-50 text-green-700 border border-green-200" 
            : dataSourceStatus.includes("failed") || dataSourceStatus.includes("❌") 
            ? "bg-red-50 text-red-700 border border-red-200"
            : "bg-blue-50 text-blue-700 border border-blue-200"
        }`}>
          {dataSourceStatus}
        </div>
      )}

      {/* Data Source Management */}
      <section className="mb-10">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Website Data Sources</h2>
          <button
            onClick={() => setShowDataSourceForm(!showDataSourceForm)}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            {showDataSourceForm ? 'Cancel' : 'Add New Source'}
          </button>
        </div>

        {/* Data Source Creation Form */}
        {showDataSourceForm && (
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h3 className="text-lg font-semibold mb-4">Add New Data Source</h3>
            <form onSubmit={handleDataSourceSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name *
                </label>
                <input
                  type="text"
                  value={dataSourceForm.name}
                  onChange={(e) => setDataSourceForm(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Company Blog"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  value={dataSourceForm.description}
                  onChange={(e) => setDataSourceForm(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Optional description"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Website URL *
                </label>
                <div className="flex gap-2">
                  <input
                    type="url"
                    value={dataSourceForm.url}
                    onChange={(e) => setDataSourceForm(prev => ({ ...prev, url: e.target.value }))}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="https://example.com"
                    required
                  />
                  <button
                    type="button"
                    onClick={testUrl}
                    disabled={!dataSourceForm.url || testingUrl}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingUrl ? 'Testing...' : 'Test URL'}
                  </button>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Scraping Frequency
                </label>
                <select
                  value={dataSourceForm.scraping_frequency}
                  onChange={(e) => setDataSourceForm(prev => ({ ...prev, scraping_frequency: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                  <option value="manual">Manual Only</option>
                </select>
              </div>
              
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={dataSourceSubmitting}
                  className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                >
                  {dataSourceSubmitting ? 'Creating...' : 'Create Data Source'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowDataSourceForm(false)}
                  className="px-6 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Data Sources List */}
        {loading ? (
          <div className="text-center py-8">Loading data sources...</div>
        ) : dataSources.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No data sources found. Add your first data source to get started!
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Data Source
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Documents
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Scraped
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
                        <div className="flex flex-col">
                          <div className="font-medium text-gray-900">{ds.name}</div>
                          <div className="text-sm text-blue-600 break-all">{ds.url}</div>
                          {ds.description && (
                            <div className="text-sm text-gray-500">{ds.description}</div>
                          )}
                          <div className="text-xs text-gray-400 mt-1">
                            {ds.scraping_frequency} • Created {formatDate(ds.created_at)}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-col gap-1">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(ds.status)}`}>
                            {ds.status}
                          </span>
                          {ds.error_count > 0 && (
                            <span className="text-xs text-red-600">
                              {ds.error_count} error(s)
                            </span>
                          )}
                          {ds.last_error && (
                            <span className="text-xs text-red-600 truncate" title={ds.last_error}>
                              {ds.last_error.length > 30 ? ds.last_error.substring(0, 30) + '...' : ds.last_error}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-col">
                          <span className="text-sm font-medium">{ds.document_count}</span>
                          <span className="text-xs text-gray-500">documents</span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-col text-sm text-gray-900">
                          <span>Scraped: {formatDate(ds.last_scraped_at)}</span>
                          <span className="text-xs text-gray-500">
                            Success: {formatDate(ds.last_success)}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-col gap-1">
                          <button
                            onClick={() => triggerScrape(ds.id, ds.name)}
                            className="inline-flex items-center px-3 py-1 border border-transparent text-xs leading-4 font-medium rounded text-blue-700 bg-blue-100 hover:bg-blue-200"
                          >
                            Scrape Now
                          </button>
                          <button
                            onClick={() => deleteDataSource(ds.id, ds.name)}
                            className="inline-flex items-center px-3 py-1 border border-transparent text-xs leading-4 font-medium rounded text-red-700 bg-red-100 hover:bg-red-200"
                          >
                            Delete
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

      {/* Document Upload Section */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-4">Upload Documents</h2>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center gap-4 mb-4">
            <input
              type="file"
              accept=".pdf,.doc,.docx,.txt,.md"
              ref={fileInputRef}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="flex-1"
            />
            <button
              onClick={handleUpload}
              disabled={uploading || !file}
              className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {uploading ? "Uploading..." : "Upload & Process"}
            </button>
          </div>
          {uploadStatus && (
            <div className={`p-3 rounded ${
              uploadStatus.includes("failed") || uploadStatus.includes("error") 
                ? "bg-red-50 text-red-700 border border-red-200" 
                : "bg-green-50 text-green-700 border border-green-200"
            }`}>
              {uploadStatus}
            </div>
          )}
        </div>
      </section>

      {/* Uploaded Documents List */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-4">Processed Documents</h2>
        {documentsLoading ? (
          <div className="text-center py-8">Loading documents...</div>
        ) : documents.length === 0 ? (
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
                      Embeddings
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {documents.map((doc) => (
                    <tr key={doc.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <div className="font-medium text-gray-900">{doc.title}</div>
                        <div className="text-sm text-gray-500 break-all">{doc.storage_path}</div>
                      </td>
                      <td className="px-6 py-4">
                        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                          {doc.document_type}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          doc.embedding_count > 0 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {doc.embedding_count} chunks
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        {formatDate(doc.created_at)}
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
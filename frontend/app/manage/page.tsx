"use client";
import { useEffect, useState, useRef } from "react";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

interface DataSource {
  id: string;
  name: string;
  url: string;
  status: string;
  scraping_frequency: string;
}

interface Document {
  id: string;
  title: string;
  document_type: string;
  storage_path: string;
  created_at: string;
  embedding_count: number;
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
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch data sources from Supabase
  useEffect(() => {
    async function fetchDataSources() {
      setLoading(true);
      const { data, error } = await supabase
        .from("data_sources")
        .select("id, name, url, status, scraping_frequency");
      if (error) {
        setDataSources([]);
      } else {
        setDataSources(data || []);
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
        // Fetch documents with embedding count already included
        const { data: documentsData, error: documentsError } = await supabase
          .from("documents")
          .select("id, title, document_type, storage_path, created_at, embedding_count")
          .order("created_at", { ascending: false });

        if (documentsError) {
          console.error("Error fetching documents:", documentsError);
          setDocuments([]);
          setDocumentsLoading(false);
          return;
        }

        setDocuments(documentsData || []);
      } catch (error) {
        console.error("Error fetching documents:", error);
        setDocuments([]);
      }
      setDocumentsLoading(false);
    }
    fetchDocuments();
  }, [refresh]);

  // Handle file upload
  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    try {
      // Upload file to Supabase Storage (bucket: 'documents')
      const fileExt = file.name.split(".").pop();
      const filePath = `${uuidv4()}.${fileExt}`;
      const { data, error } = await supabase.storage
        .from("documents")
        .upload(filePath, file);
      if (error) {
        console.error("Upload error:", error);
        
        // Provide specific guidance for bucket not found error
        if (error.message.includes("Bucket not found")) {
          setUploadStatus(
            "Upload failed: Documents bucket not found. Please create a 'documents' bucket in your Supabase Storage dashboard first. " +
            "Go to your Supabase project → Storage → Create bucket → Name it 'documents' and set it as private."
          );
        } else {
          setUploadStatus("Upload failed: " + error.message);
        }
        setUploading(false);
        return;
      }
      
      // File uploaded successfully to Supabase Storage
      console.log("File uploaded successfully to storage:", data);
      
      // Now trigger backend processing
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
          console.log("Processing started:", result);
          
          setUploadStatus(
            `Upload successful! Document "${file.name}" is being processed. ` +
            `Document ID: ${result.data?.document_id}. Processing will complete in the background.`
          );
          
          // Refresh the documents list after a short delay
          setTimeout(() => setRefresh((r) => r + 1), 2000);
        } else {
          let errorMessage = "Unknown error";
          try {
            const errorData = await response.json();
            if (errorData.detail) {
              // Handle case where detail might be an object or array
              errorMessage = typeof errorData.detail === 'string' 
                ? errorData.detail 
                : JSON.stringify(errorData.detail);
            } else if (errorData.message) {
              errorMessage = errorData.message;
            } else if (errorData.error) {
              errorMessage = errorData.error;
            }
          } catch (jsonError) {
            // If JSON parsing fails, try to get error as text
            try {
              const errorText = await response.text();
              errorMessage = errorText || `HTTP ${response.status}: ${response.statusText}`;
            } catch (textError) {
              errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
          }
          
          setUploadStatus(
            `Upload successful, but processing failed: ${errorMessage}. ` +
            `File is stored but not indexed for search.`
          );
        }
      } catch (processError) {
        console.error("Processing request failed:", processError);
        setUploadStatus(
          `Upload successful, but couldn't start processing: ${processError.message}. ` +
          `File is stored but not indexed for search. Check if backend is running.`
        );
      }
      
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      
    } catch (e: any) {
      console.error("Upload exception:", e);
      setUploadStatus("Upload error: " + e.message);
    }
    setUploading(false);
  }

  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-6">Manage Data Sources & Documents</h1>

      {/* Document Upload */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-2">Upload PDF or Document</h2>
        <input
          type="file"
          accept=".pdf,.doc,.docx,.txt,.md"
          ref={fileInputRef}
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="mb-2"
        />
        <button
          className="ml-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          onClick={handleUpload}
          disabled={uploading || !file}
        >
          {uploading ? "Uploading..." : "Upload & Process"}
        </button>
        {uploadStatus && (
          <div className={`mt-2 p-3 rounded ${
            uploadStatus.includes("failed") || uploadStatus.includes("error") 
              ? "bg-red-50 text-red-700 border border-red-200" 
              : "bg-green-50 text-green-700 border border-green-200"
          }`}>
            {uploadStatus}
          </div>
        )}
      </section>

      {/* Uploaded Documents List */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-2">Uploaded Documents</h2>
        {documentsLoading ? (
          <div>Loading documents...</div>
        ) : documents.length === 0 ? (
          <div className="text-gray-500">No documents uploaded yet.</div>
        ) : (
          <div className="bg-white rounded shadow overflow-hidden">
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
                    <tr key={doc.id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="font-medium text-gray-900">{doc.title}</div>
                        <div className="text-sm text-gray-500 break-all">{doc.storage_path}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                          {doc.document_type}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            doc.embedding_count > 0 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {doc.embedding_count} chunks
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        <button
          className="mt-4 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          onClick={() => setRefresh((r) => r + 1)}
        >
          Refresh Documents
        </button>
      </section>

      {/* Data Source List */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-2">Data Sources</h2>
        {loading ? (
          <div>Loading...</div>
        ) : dataSources.length === 0 ? (
          <div className="text-gray-500">No data sources found.</div>
        ) : (
          <ul className="divide-y divide-gray-200 bg-white rounded shadow">
            {dataSources.map((ds) => (
              <li key={ds.id} className="p-4 flex flex-col md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="font-medium">{ds.name}</div>
                  <div className="text-sm text-blue-600 break-all">{ds.url}</div>
                  <div className="text-xs text-gray-500">Frequency: {ds.scraping_frequency}</div>
                </div>
                <div className="mt-2 md:mt-0">
                  <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${ds.status === "active" ? "bg-green-100 text-green-700" : "bg-gray-200 text-gray-600"}`}>{ds.status}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
        <button
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          onClick={() => setRefresh((r) => r + 1)}
        >
          Refresh List
        </button>
      </section>
    </div>
  );
} 
'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { SystemHealth, DashboardStats } from '@/types';

// Additional metrics interfaces
interface EmbeddingMetrics {
  total_embeddings: number;
  processing_rate: number;
  success_rate: number;
  average_processing_time: number;
  model_performance: {
    model_name: string;
    accuracy: number;
    response_time: number;
  };
}

interface IndexingStats {
  total_chunks: number;
  processing_status: {
    pending: number;
    processing: number;
    completed: number;
    failed: number;
  };
  file_type_distribution: {
    pdf: number;
    word: number;
    text: number;
    html: number;
    markdown: number;
  };
  average_chunk_size: number;
  storage_usage: string;
}

interface SystemMetrics {
  query_performance: {
    average_response_time: number;
    queries_per_hour: number;
    success_rate: number;
  };
  user_satisfaction: {
    average_rating: number;
    total_ratings: number;
    thumbs_up_percentage: number;
  };
  conflict_detection: {
    conflicts_detected: number;
    resolution_rate: number;
  };
}

export default function Home() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [embeddingMetrics, setEmbeddingMetrics] = useState<EmbeddingMetrics | null>(null);
  const [indexingStats, setIndexingStats] = useState<IndexingStats | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        // Try to fetch health data
        const healthResponse = await apiClient.getHealth();
        if (healthResponse.success && healthResponse.data) {
          setSystemHealth(healthResponse.data);
        }
        
        // Try to fetch service info as a fallback
        const serviceResponse = await apiClient.getServiceInfo();
        if (serviceResponse.success) {
          console.log('Service info:', serviceResponse.data);
        }

        // Fetch comprehensive metrics (simulate with mock data for now)
        await fetchComprehensiveMetrics();
        
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to connect to backend service');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const fetchComprehensiveMetrics = async () => {
    try {
      // In a real implementation, these would be separate API calls
      // For now, we'll use mock data based on the database schema
      
      // Mock embedding metrics
      setEmbeddingMetrics({
        total_embeddings: 1547,
        processing_rate: 45.2,
        success_rate: 97.8,
        average_processing_time: 1.2,
        model_performance: {
          model_name: 'text-embedding-3-small',
          accuracy: 94.5,
          response_time: 0.85
        }
      });

      // Mock indexing stats
      setIndexingStats({
        total_chunks: 892,
        processing_status: {
          pending: 12,
          processing: 3,
          completed: 867,
          failed: 10
        },
        file_type_distribution: {
          pdf: 234,
          word: 156,
          text: 89,
          html: 67,
          markdown: 45
        },
        average_chunk_size: 1247,
        storage_usage: '45.2 MB'
      });

      // Mock system metrics
      setSystemMetrics({
        query_performance: {
          average_response_time: 1.45,
          queries_per_hour: 127,
          success_rate: 98.9
        },
        user_satisfaction: {
          average_rating: 4.2,
          total_ratings: 186,
          thumbs_up_percentage: 87.3
        },
        conflict_detection: {
          conflicts_detected: 8,
          resolution_rate: 75.0
        }
      });

    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  const formatPercentage = (value: number) => `${value.toFixed(1)}%`;
  const formatTime = (seconds: number) => `${seconds.toFixed(2)}s`;
  const formatNumber = (num: number) => num.toLocaleString();

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-lg shadow-lg p-8 text-white">
        <div className="max-w-3xl">
          <h1 className="text-4xl font-bold mb-4">
            Welcome to RAG-Tobi 🚀
          </h1>
          <p className="text-xl text-primary-100 mb-6">
            Your AI-powered salesperson copilot is ready to help you succeed. 
            Upload documents, configure data sources, and get intelligent assistance for your sales activities.
          </p>
          <div className="flex flex-wrap gap-4">
            <button className="bg-white text-primary-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              Get Started
            </button>
            <button className="border border-white text-white px-6 py-3 rounded-lg font-semibold hover:bg-white hover:text-primary-600 transition-colors">
              Learn More
            </button>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                <div className="w-4 h-4 bg-green-500 rounded-full"></div>
              </div>
            </div>
            <div className="ml-4">
              <h3 className="text-sm font-medium text-gray-500">Frontend</h3>
              <p className="text-2xl font-semibold text-green-600">Online</p>
            </div>
          </div>
          <p className="mt-2 text-sm text-gray-600">Running on port 3000</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <div className={`w-4 h-4 rounded-full ${
                  error ? 'bg-red-500' : 'bg-blue-500'
                }`}></div>
              </div>
            </div>
            <div className="ml-4">
              <h3 className="text-sm font-medium text-gray-500">Backend API</h3>
              <p className={`text-2xl font-semibold ${
                error ? 'text-red-600' : 'text-blue-600'
              }`}>
                {error ? 'Offline' : 'Online'}
              </p>
            </div>
          </div>
          <p className="mt-2 text-sm text-gray-600">
            {error ? 'Connection failed' : 'Running on port 8000'}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                <div className={`w-4 h-4 rounded-full ${
                  systemHealth?.database === 'healthy' ? 'bg-purple-500' : 'bg-yellow-500'
                }`}></div>
              </div>
            </div>
            <div className="ml-4">
              <h3 className="text-sm font-medium text-gray-500">Database</h3>
              <p className={`text-2xl font-semibold ${
                systemHealth?.database === 'healthy' ? 'text-purple-600' : 'text-yellow-600'
              }`}>
                {systemHealth?.database === 'healthy' ? 'Connected' : 'Pending'}
              </p>
            </div>
          </div>
          <p className="mt-2 text-sm text-gray-600">Supabase PostgreSQL</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                <div className="w-4 h-4 bg-indigo-500 rounded-full"></div>
              </div>
            </div>
            <div className="ml-4">
              <h3 className="text-sm font-medium text-gray-500">RAG System</h3>
              <p className="text-2xl font-semibold text-indigo-600">Ready</p>
            </div>
          </div>
          <p className="mt-2 text-sm text-gray-600">OpenAI Embeddings</p>
        </div>
      </div>

      {/* Embedding Quality Metrics */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">
            📊 Embedding Quality Metrics
          </h2>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600">Real-time</span>
          </div>
        </div>
        
        {embeddingMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary-600">
                {formatNumber(embeddingMetrics.total_embeddings)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Total Embeddings</div>
              <div className="text-xs text-green-600 mt-1">
                +{embeddingMetrics.processing_rate}/min
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">
                {formatPercentage(embeddingMetrics.success_rate)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Success Rate</div>
              <div className="text-xs text-gray-500 mt-1">Last 24h</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">
                {formatTime(embeddingMetrics.average_processing_time)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Avg Processing Time</div>
              <div className="text-xs text-gray-500 mt-1">Per document</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600">
                {formatPercentage(embeddingMetrics.model_performance.accuracy)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Model Accuracy</div>
              <div className="text-xs text-gray-500 mt-1">
                {embeddingMetrics.model_performance.model_name}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Indexing Statistics */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">
          🔍 Indexing Statistics
        </h2>
        
        {indexingStats && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Processing Status */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">Processing Status</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Completed</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{
                        width: `${(indexingStats.processing_status.completed / indexingStats.total_chunks) * 100}%`
                      }}></div>
                    </div>
                    <span className="text-sm font-medium text-green-600">
                      {indexingStats.processing_status.completed}
                    </span>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Pending</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div className="bg-yellow-500 h-2 rounded-full" style={{
                        width: `${(indexingStats.processing_status.pending / indexingStats.total_chunks) * 100}%`
                      }}></div>
                    </div>
                    <span className="text-sm font-medium text-yellow-600">
                      {indexingStats.processing_status.pending}
                    </span>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Processing</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-500 h-2 rounded-full" style={{
                        width: `${(indexingStats.processing_status.processing / indexingStats.total_chunks) * 100}%`
                      }}></div>
                    </div>
                    <span className="text-sm font-medium text-blue-600">
                      {indexingStats.processing_status.processing}
                    </span>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Failed</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div className="bg-red-500 h-2 rounded-full" style={{
                        width: `${(indexingStats.processing_status.failed / indexingStats.total_chunks) * 100}%`
                      }}></div>
                    </div>
                    <span className="text-sm font-medium text-red-600">
                      {indexingStats.processing_status.failed}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* File Type Distribution */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">File Type Distribution</h3>
              <div className="space-y-2">
                {Object.entries(indexingStats.file_type_distribution).map(([type, count]) => (
                  <div key={type} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 capitalize">{type}</span>
                    <span className="text-sm font-medium text-gray-900">{count} files</span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Storage & Performance */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">Storage & Performance</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Chunks</span>
                  <span className="text-sm font-medium text-gray-900">
                    {formatNumber(indexingStats.total_chunks)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg Chunk Size</span>
                  <span className="text-sm font-medium text-gray-900">
                    {formatNumber(indexingStats.average_chunk_size)} chars
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Storage Usage</span>
                  <span className="text-sm font-medium text-gray-900">
                    {indexingStats.storage_usage}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* System Performance Metrics */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">
          ⚡ System Performance Metrics
        </h2>
        
        {systemMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Query Performance */}
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600 mb-2">
                {formatTime(systemMetrics.query_performance.average_response_time)}
              </div>
              <div className="text-sm text-blue-700 mb-1">Average Response Time</div>
              <div className="text-xs text-blue-600">
                {systemMetrics.query_performance.queries_per_hour} queries/hour
              </div>
              <div className="text-xs text-blue-600">
                {formatPercentage(systemMetrics.query_performance.success_rate)} success rate
              </div>
            </div>
            
            {/* User Satisfaction */}
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-2">
                {systemMetrics.user_satisfaction.average_rating}/5
              </div>
              <div className="text-sm text-green-700 mb-1">User Satisfaction</div>
              <div className="text-xs text-green-600">
                {systemMetrics.user_satisfaction.total_ratings} ratings
              </div>
              <div className="text-xs text-green-600">
                {formatPercentage(systemMetrics.user_satisfaction.thumbs_up_percentage)} positive
              </div>
            </div>
            
            {/* Conflict Detection */}
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 mb-2">
                {systemMetrics.conflict_detection.conflicts_detected}
              </div>
              <div className="text-sm text-purple-700 mb-1">Conflicts Detected</div>
              <div className="text-xs text-purple-600">
                {formatPercentage(systemMetrics.conflict_detection.resolution_rate)} resolved
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Features Overview */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Available Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">📄</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">Document Processing</h3>
              <p className="text-sm text-gray-600">
                Upload and process PDFs, Word docs, and web content
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">🔍</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">Semantic Search</h3>
              <p className="text-sm text-gray-600">
                Find relevant information with AI-powered search
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">📊</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">Analytics & Monitoring</h3>
              <p className="text-sm text-gray-600">
                Track performance and system health
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">💬</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">Conversation Management</h3>
              <p className="text-sm text-gray-600">
                Smart chat with context-aware responses
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">🤖</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">LangGraph Agent</h3>
              <p className="text-sm text-gray-600">
                Advanced AI agent with state management
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">⚡</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">Real-time Metrics</h3>
              <p className="text-sm text-gray-600">
                Live monitoring of system performance
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Development Status */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Development Status
        </h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Infrastructure Setup</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Complete
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Backend API Structure</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Complete
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Frontend Framework</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Complete
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Document Upload & Processing</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Complete
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Metrics Dashboard</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Complete
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">RAG System</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
              In Progress
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Agent Architecture</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
              Pending
            </span>
          </div>
        </div>
      </div>

      {/* Next Steps */}
      <div className="bg-blue-50 rounded-lg border border-blue-200 p-6">
        <h2 className="text-lg font-semibold text-blue-900 mb-3">
          🚀 Next Steps
        </h2>
        <ul className="space-y-2 text-sm text-blue-800">
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Implement LangGraph agent architecture</span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Add conversational AI capabilities</span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Build temporary chat interface for testing</span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Integrate monitoring and testing infrastructure</span>
          </li>
        </ul>
      </div>
    </div>
  );
} 
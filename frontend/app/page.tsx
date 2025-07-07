'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { SystemHealth, DashboardStats } from '@/types';

export default function Home() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
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
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to connect to backend service');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

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
                <span className="text-primary-600 font-semibold">🌐</span>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900">Web Scraping</h3>
              <p className="text-sm text-gray-600">
                Automatically scrape and index website content
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
            <span>Complete RAG system with document loaders and embeddings</span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Implement LangGraph agent architecture</span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Build web management dashboard components</span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-600">•</span>
            <span>Add monitoring and testing infrastructure</span>
          </li>
        </ul>
      </div>
    </div>
  );
} 
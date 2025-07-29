'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { SystemHealth } from '@/types';

export default function DevHomePage() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const healthResponse = await apiClient.getHealth();
        if (healthResponse.success && healthResponse.data) {
          setSystemHealth(healthResponse.data);
        }
      } catch (err) {
        console.error('Error fetching system health:', err);
        setError('Failed to connect to backend service');
      } finally {
        setLoading(false);
      }
    };

    fetchSystemStatus();
  }, []);

  const devTools = [
    {
      title: 'Dual Agent Debug',
      path: '/dev/dualagentdebug',
      description: 'Test and debug the dual agent system with comprehensive chat interface and state inspection',
      status: 'Active',
      features: ['Live Chat Testing', 'Agent State Inspection', 'Message Flow Debug', 'User Simulation']
    },
    {
      title: 'Data Management',
      path: '/dev/manage',
      description: 'Upload documents, manage data sources, and monitor document processing pipeline',
      status: 'Active', 
      features: ['Document Upload', 'Data Source Config', 'Processing Status', 'Vector Store Management']
    },
    {
      title: 'Memory Check',
      path: '/dev/memorycheck',
      description: 'Debug memory systems, conversation summaries, and long-term memory storage',
      status: 'Active',
      features: ['Memory Inspector', 'Conversation Summaries', 'User Context Debug', 'Memory Consolidation']
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Developer Tools</h1>
              <p className="mt-2 text-gray-600">
                Internal testing, debugging, and management interfaces
              </p>
            </div>
            <Link
              href="/"
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              <span>Back to Client Home</span>
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Status */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">System Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full mr-3 ${
                  error ? 'bg-red-500' : 'bg-green-500'
                }`}></div>
                <div>
                  <p className="text-sm font-medium text-gray-900">Backend API</p>
                  <p className="text-sm text-gray-600">
                    {error ? 'Offline' : 'Connected'}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full mr-3 ${
                  systemHealth?.database === 'healthy' ? 'bg-green-500' : 'bg-yellow-500'
                }`}></div>
                <div>
                  <p className="text-sm font-medium text-gray-900">Database</p>
                  <p className="text-sm text-gray-600">
                    {systemHealth?.database === 'healthy' ? 'Connected' : 'Checking...'}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full mr-3 bg-blue-500"></div>
                <div>
                  <p className="text-sm font-medium text-gray-900">RAG System</p>
                  <p className="text-sm text-gray-600">Ready</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Dev Tools Grid */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Available Tools</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {devTools.map((tool, index) => (
              <Link
                key={index}
                href={tool.path}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow group"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 group-hover:text-primary-600">
                      {tool.title}
                    </h3>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      tool.status === 'Active' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {tool.status}
                    </span>
                  </div>
                </div>
                
                <p className="text-gray-600 mb-4 text-sm">
                  {tool.description}
                </p>
                
                <div className="space-y-1">
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Key Features
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {tool.features.map((feature, featureIndex) => (
                      <span
                        key={featureIndex}
                        className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-700"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
              Restart Services
            </button>
            <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
              Clear Cache
            </button>
            <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
              View Logs
            </button>
            <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
              System Config
            </button>
          </div>
        </div>

        {/* Documentation */}
        <div className="mt-8 bg-blue-50 rounded-lg border border-blue-200 p-6">
          <h2 className="text-lg font-semibold text-blue-900 mb-3">
            📚 Developer Documentation
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h3 className="font-medium text-blue-800 mb-2">Development Guides</h3>
              <ul className="space-y-1 text-blue-700">
                <li>• Agent Architecture Overview</li>
                <li>• Memory System Documentation</li>
                <li>• RAG Pipeline Configuration</li>
                <li>• Testing Best Practices</li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium text-blue-800 mb-2">API References</h3>
              <ul className="space-y-1 text-blue-700">
                <li>• Chat API Endpoints</li>
                <li>• Document Processing API</li>
                <li>• Memory Management API</li>
                <li>• System Health Endpoints</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 
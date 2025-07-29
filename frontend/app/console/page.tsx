'use client';

import { useState } from 'react';
import Link from 'next/link';
import DataEntry from './components/DataEntry';
import AgentTools from './components/AgentTools';
import Settings from './components/Settings';

type ActiveSection = 'data-entry' | 'agent-tools' | 'settings';

export default function ConsolePage() {
  const [activeSection, setActiveSection] = useState<ActiveSection>('data-entry');

  const sidebarItems = [
    {
      id: 'data-entry' as ActiveSection,
      name: 'Data Entry',
      description: 'Manage sales data and content'
    },
    {
      id: 'agent-tools' as ActiveSection,
      name: 'Agent Tools',
      description: 'Configure AI agent behavior'
    },
    {
      id: 'settings' as ActiveSection,
      name: 'Settings',
      description: 'System and user preferences'
    }
  ];

  const renderActiveComponent = () => {
    switch (activeSection) {
      case 'data-entry':
        return <DataEntry />;
      case 'agent-tools':
        return <AgentTools />;
      case 'settings':
        return <Settings />;
      default:
        return <DataEntry />;
    }
  };

  const getActiveItem = () => sidebarItems.find(item => item.id === activeSection);

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-64 bg-white shadow-lg border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Admin Console</h1>
            <p className="text-sm text-gray-500">Dashboard</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {sidebarItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveSection(item.id)}
              className={`w-full text-left p-4 rounded-lg transition-all duration-200 group ${
                activeSection === item.id
                  ? 'bg-primary-50 border border-primary-200 shadow-sm'
                  : 'hover:bg-gray-50 border border-transparent'
              }`}
            >
              <div className="flex items-center space-x-3">
                <div className="flex-1">
                  <div className={`font-semibold ${
                    activeSection === item.id ? 'text-primary-700' : 'text-gray-700'
                  }`}>
                    {item.name}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {item.description}
                  </div>
                </div>
              </div>
              {activeSection === item.id && (
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 w-1 h-8 bg-primary-500 rounded-full"></div>
              )}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200">
          <Link
            href="/"
            className="flex items-center space-x-2 text-sm text-gray-600 hover:text-primary-600 transition-colors duration-200"
          >
            <span>←</span>
            <span>Back to Homepage</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="bg-white shadow-sm border-b border-gray-200 px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {getActiveItem()?.name}
              </h2>
              <p className="text-gray-600 mt-1">{getActiveItem()?.description}</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-600">System Online</span>
              </div>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-8 overflow-auto">
          <div className="max-w-7xl mx-auto">
            {renderActiveComponent()}
          </div>
        </div>
      </div>
    </div>
  );
} 
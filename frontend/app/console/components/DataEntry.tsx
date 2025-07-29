'use client';

import React, { useState } from 'react';

const DataEntry: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'upload' | 'manage' | 'analyze'>('upload');

  const tabs = [
    { id: 'upload', name: 'Upload Content' },
    { id: 'manage', name: 'Manage Data' },
    { id: 'analyze', name: 'Data Analysis' }
  ];

  const renderUploadTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Sales Content</h3>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors duration-200">
          <div className="space-y-4">
            <div>
              <p className="text-lg font-medium text-gray-900">Drop files here or click to browse</p>
              <p className="text-sm text-gray-500 mt-1">
                Supported formats: PDF, DOCX, TXT, CSV, Excel
              </p>
            </div>
            <button className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors duration-200">
              Choose Files
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="font-semibold text-gray-900 mb-3">Quick Upload Categories</h4>
          <div className="space-y-2">
            {['Sales Playbooks', 'Product Catalogs', 'Training Materials', 'Case Studies', 'Pricing Sheets'].map((category) => (
              <button
                key={category}
                className="w-full text-left p-3 rounded-lg bg-gray-50 hover:bg-primary-50 hover:border-primary-200 border border-transparent transition-all duration-200"
              >
                <span className="font-medium text-gray-700">{category}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="font-semibold text-gray-900 mb-3">Recent Uploads</h4>
          <div className="space-y-3">
            {[
              { name: 'Sales_Playbook_2024.pdf', size: '2.3 MB', status: 'Processing' },
              { name: 'Product_Catalog.xlsx', size: '1.8 MB', status: 'Complete' },
              { name: 'Training_Module_1.docx', size: '956 KB', status: 'Complete' }
            ].map((file, index) => (
              <div key={index} className="flex items-center justify-between p-2 rounded bg-gray-50">
                <div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{file.size}</p>
                  </div>
                </div>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  file.status === 'Complete' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {file.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderManageTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Content Library</h3>
          <div className="flex space-x-2">
            <input
              type="text"
              placeholder="Search content..."
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <button className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors duration-200">
              Filter
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-medium text-gray-700">Name</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Type</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Size</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Upload Date</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Status</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: 'Sales Process Guide', type: 'PDF', size: '2.3 MB', date: '2024-01-15', status: 'Active' },
                { name: 'Product Features List', type: 'Excel', size: '1.2 MB', date: '2024-01-14', status: 'Active' },
                { name: 'Objection Handling Scripts', type: 'Word', size: '890 KB', date: '2024-01-13', status: 'Draft' },
                { name: 'Competitor Analysis', type: 'PDF', size: '3.1 MB', date: '2024-01-12', status: 'Active' }
              ].map((item, index) => (
                <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 font-medium text-gray-900">{item.name}</td>
                  <td className="py-3 px-4 text-gray-600">{item.type}</td>
                  <td className="py-3 px-4 text-gray-600">{item.size}</td>
                  <td className="py-3 px-4 text-gray-600">{item.date}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      item.status === 'Active' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {item.status}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 text-sm">Edit</button>
                      <button className="text-red-600 hover:text-red-800 text-sm">Delete</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderAnalyzeTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { title: 'Total Documents', value: '247', change: '+12%' },
          { title: 'Data Sources', value: '18', change: '+2' },
          { title: 'Processing Success', value: '98.2%', change: '+0.5%' }
        ].map((stat, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                <p className="text-sm text-green-600 mt-1">{stat.change}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Content Performance</h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-2xl font-bold text-primary-600 mb-2">ANALYTICS</div>
            <p>Analytics Chart Placeholder</p>
            <p className="text-sm">Integration with analytics service required</p>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-1">
        <div className="flex space-x-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-primary-100 text-primary-700 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <span>{tab.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'upload' && renderUploadTab()}
        {activeTab === 'manage' && renderManageTab()}
        {activeTab === 'analyze' && renderAnalyzeTab()}
      </div>
    </div>
  );
};

export default DataEntry; 
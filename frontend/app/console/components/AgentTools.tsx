'use client';

import React, { useState } from 'react';

const AgentTools: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'behavior' | 'training' | 'performance'>('behavior');

  const tabs = [
    { id: 'behavior', name: 'Agent Behavior' },
    { id: 'training', name: 'Model Training' },
    { id: 'performance', name: 'Performance Metrics' }
  ];

  const renderBehaviorTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Personality Settings */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Personality</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Communication Style
              </label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500">
                <option>Professional & Friendly</option>
                <option>Formal & Direct</option>
                <option>Casual & Conversational</option>
                <option>Technical & Detailed</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Response Length
              </label>
              <div className="flex items-center space-x-4">
                <input
                  type="range"
                  min="1"
                  max="5"
                  defaultValue="3"
                  className="flex-1"
                />
                <span className="text-sm text-gray-600 w-20">Moderate</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Concise</span>
                <span>Detailed</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Proactivity Level
              </label>
              <div className="flex items-center space-x-4">
                <input
                  type="range"
                  min="1"
                  max="5"
                  defaultValue="4"
                  className="flex-1"
                />
                <span className="text-sm text-gray-600 w-20">High</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Reactive</span>
                <span>Proactive</span>
              </div>
            </div>
          </div>
        </div>

        {/* Response Rules */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Response Rules</h3>
          <div className="space-y-4">
            {[
              { rule: 'Always ask clarifying questions', enabled: true },
              { rule: 'Provide sources for claims', enabled: true },
              { rule: 'Offer next steps in responses', enabled: false },
              { rule: 'Use customer name in responses', enabled: true },
              { rule: 'Include pricing information when relevant', enabled: false }
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">{item.rule}</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    defaultChecked={item.enabled}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                </label>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { name: 'Reset to Defaults', action: 'reset' },
            { name: 'Save Configuration', action: 'save' },
            { name: 'Test Agent', action: 'test' },
            { name: 'Export Settings', action: 'export' }
          ].map((action, index) => (
            <button
              key={index}
              className="p-4 bg-gray-50 rounded-lg hover:bg-primary-50 hover:border-primary-200 border border-transparent transition-all duration-200"
            >
              <span className="text-sm font-medium text-gray-700">{action.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const renderTrainingTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Knowledge Base Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { title: 'Total Documents', value: '1,247', status: 'Processed', color: 'green' },
            { title: 'Training Examples', value: '3,891', status: 'Active', color: 'blue' },
            { title: 'Knowledge Gaps', value: '12', status: 'Needs Attention', color: 'yellow' }
          ].map((item, index) => (
            <div key={index} className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{item.value}</div>
              <div className="text-sm text-gray-600 mt-1">{item.title}</div>
              <div className={`text-xs mt-2 px-2 py-1 rounded-full inline-block ${
                item.color === 'green' ? 'bg-green-100 text-green-800' :
                item.color === 'blue' ? 'bg-blue-100 text-blue-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                {item.status}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Topics */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Training Topics</h3>
            <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
              + Add Topic
            </button>
          </div>
          <div className="space-y-3">
            {[
              { topic: 'Product Knowledge', progress: 95, examples: 145 },
              { topic: 'Objection Handling', progress: 87, examples: 98 },
              { topic: 'Closing Techniques', progress: 76, examples: 67 },
              { topic: 'Industry Insights', progress: 92, examples: 203 },
              { topic: 'Competitor Analysis', progress: 83, examples: 89 }
            ].map((item, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium text-gray-900">{item.topic}</span>
                  <span className="text-sm text-gray-600">{item.examples} examples</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full"
                    style={{ width: `${item.progress}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1">{item.progress}% complete</div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Training Activity */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Training Activity</h3>
          <div className="space-y-3">
            {[
              { action: 'Added 15 new product examples', time: '2 hours ago', type: 'success' },
              { action: 'Updated pricing information', time: '4 hours ago', type: 'info' },
              { action: 'Training model retrained', time: '1 day ago', type: 'warning' },
              { action: 'Knowledge validation completed', time: '2 days ago', type: 'success' }
            ].map((item, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  item.type === 'success' ? 'bg-green-500' :
                  item.type === 'info' ? 'bg-blue-500' :
                  'bg-yellow-500'
                }`}></div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{item.action}</p>
                  <p className="text-xs text-gray-500">{item.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderPerformanceTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { title: 'Response Time', value: '1.2s', change: '-0.3s' },
          { title: 'Accuracy Rate', value: '94.7%', change: '+2.1%' },
          { title: 'User Satisfaction', value: '4.6/5', change: '+0.2' },
          { title: 'Conversations', value: '2,847', change: '+147' }
        ].map((metric, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-600">{metric.title}</span>
            </div>
            <div className="text-2xl font-bold text-gray-900">{metric.value}</div>
            <div className="text-sm text-green-600 mt-1">{metric.change}</div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Overview</h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
                          <div className="text-2xl font-bold text-primary-600 mb-2">TRENDS</div>
            <p>Performance Dashboard</p>
            <p className="text-sm">Real-time metrics and analytics</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Performing Topics</h3>
          <div className="space-y-3">
            {[
              { topic: 'Product Features', score: 98.2, queries: 234 },
              { topic: 'Pricing Information', score: 96.7, queries: 189 },
              { topic: 'Technical Support', score: 94.1, queries: 156 },
              { topic: 'Integration Help', score: 91.8, queries: 143 }
            ].map((item, index) => (
              <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{item.topic}</div>
                  <div className="text-xs text-gray-500">{item.queries} queries</div>
                </div>
                <div className="text-right">
                  <div className="font-bold text-green-600">{item.score}%</div>
                  <div className="text-xs text-gray-500">accuracy</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Areas for Improvement</h3>
          <div className="space-y-3">
            {[
              { area: 'Complex Pricing Scenarios', score: 76.3, priority: 'High' },
              { area: 'Competitor Comparisons', score: 82.1, priority: 'Medium' },
              { area: 'Custom Integrations', score: 79.4, priority: 'Medium' },
              { area: 'Advanced Features', score: 85.2, priority: 'Low' }
            ].map((item, index) => (
              <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{item.area}</div>
                  <div className={`text-xs px-2 py-1 rounded-full inline-block mt-1 ${
                    item.priority === 'High' ? 'bg-red-100 text-red-800' :
                    item.priority === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {item.priority} Priority
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-bold text-orange-600">{item.score}%</div>
                  <div className="text-xs text-gray-500">accuracy</div>
                </div>
              </div>
            ))}
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
        {activeTab === 'behavior' && renderBehaviorTab()}
        {activeTab === 'training' && renderTrainingTab()}
        {activeTab === 'performance' && renderPerformanceTab()}
      </div>
    </div>
  );
};

export default AgentTools; 
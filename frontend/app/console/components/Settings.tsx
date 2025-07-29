'use client';

import React, { useState } from 'react';

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'general' | 'security' | 'integrations' | 'notifications'>('general');

    const tabs = [
    { id: 'general', name: 'General' },
    { id: 'security', name: 'Security' },
    { id: 'integrations', name: 'Integrations' },
    { id: 'notifications', name: 'Notifications' }
  ];

  const renderGeneralTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Configuration */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                System Language
              </label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500">
                <option>English (US)</option>
                <option>English (UK)</option>
                <option>Spanish</option>
                <option>French</option>
                <option>German</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Time Zone
              </label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500">
                <option>UTC-8 (Pacific Time)</option>
                <option>UTC-5 (Eastern Time)</option>
                <option>UTC+0 (GMT)</option>
                <option>UTC+1 (Central European Time)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Date Format
              </label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500">
                <option>MM/DD/YYYY</option>
                <option>DD/MM/YYYY</option>
                <option>YYYY-MM-DD</option>
              </select>
            </div>
          </div>
        </div>

        {/* Theme & Appearance */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Theme & Appearance</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Theme Mode
              </label>
              <div className="grid grid-cols-3 gap-3">
                {['Light', 'Dark', 'Auto'].map((theme) => (
                  <button
                    key={theme}
                    className={`p-3 rounded-lg border-2 text-center transition-all duration-200 ${
                      theme === 'Light' 
                        ? 'border-primary-500 bg-primary-50 text-primary-700' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-sm font-medium">{theme}</div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Accent Color
              </label>
              <div className="flex space-x-2">
                {['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-red-500', 'bg-yellow-500'].map((color) => (
                  <button
                    key={color}
                    className={`w-8 h-8 rounded-full ${color} ${
                      color === 'bg-blue-500' ? 'ring-2 ring-gray-400' : ''
                    }`}
                  ></button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium text-gray-900">Compact Mode</div>
                <div className="text-sm text-gray-500">Reduce spacing for more content</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Data Management */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Management</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg text-center">
            <div className="font-medium text-gray-900">Export Data</div>
            <div className="text-sm text-gray-500 mb-3">Download all your data</div>
            <button className="bg-primary-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-primary-700 transition-colors duration-200">
              Export
            </button>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-lg text-center">
            <div className="font-medium text-gray-900">Import Data</div>
            <div className="text-sm text-gray-500 mb-3">Import from external sources</div>
            <button className="bg-gray-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-gray-700 transition-colors duration-200">
              Import
            </button>
          </div>
          
          <div className="p-4 bg-red-50 rounded-lg text-center border border-red-200">
            <div className="font-medium text-red-900">Clear Data</div>
            <div className="text-sm text-red-600 mb-3">Remove all stored data</div>
            <button className="bg-red-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-red-700 transition-colors duration-200">
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSecurityTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Authentication */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Authentication</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-200">
              <div>
                <div className="font-medium text-green-900">Two-Factor Authentication</div>
                <div className="text-sm text-green-600">Enabled via authenticator app</div>
              </div>
              <button className="text-green-600 hover:text-green-700 text-sm">
                Manage
              </button>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="text-gray-400">🔑</div>
                <div>
                  <div className="font-medium text-gray-900">Single Sign-On (SSO)</div>
                  <div className="text-sm text-gray-500">Not configured</div>
                </div>
              </div>
              <button className="text-primary-600 hover:text-primary-700 text-sm">
                Setup
              </button>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="text-gray-400">📱</div>
                <div>
                  <div className="font-medium text-gray-900">Backup Codes</div>
                  <div className="text-sm text-gray-500">Generate recovery codes</div>
                </div>
              </div>
              <button className="text-primary-600 hover:text-primary-700 text-sm">
                Generate
              </button>
            </div>
          </div>
        </div>

        {/* Access Control */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Access Control</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session Timeout
              </label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500">
                <option>15 minutes</option>
                <option>30 minutes</option>
                <option>1 hour</option>
                <option>2 hours</option>
                <option>Never</option>
              </select>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium text-gray-900">Require re-authentication</div>
                <div className="text-sm text-gray-500">For sensitive operations</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" defaultChecked className="sr-only peer" />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium text-gray-900">IP Whitelisting</div>
                <div className="text-sm text-gray-500">Restrict access by IP address</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Security Logs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Security Activity</h3>
        <div className="space-y-3">
          {[
            { action: 'Successful login', location: 'San Francisco, CA', time: '2 hours ago', status: 'success' },
            { action: 'Password changed', location: 'San Francisco, CA', time: '1 day ago', status: 'info' },
            { action: 'Failed login attempt', location: 'Unknown', time: '3 days ago', status: 'warning' },
            { action: '2FA enabled', location: 'San Francisco, CA', time: '1 week ago', status: 'success' }
          ].map((log, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full ${
                  log.status === 'success' ? 'bg-green-500' :
                  log.status === 'info' ? 'bg-blue-500' :
                  'bg-yellow-500'
                }`}></div>
                <div>
                  <div className="font-medium text-gray-900">{log.action}</div>
                  <div className="text-sm text-gray-500">{log.location}</div>
                </div>
              </div>
              <div className="text-sm text-gray-500">{log.time}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderIntegrationsTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          { 
            name: 'Salesforce', 
            status: 'Connected', 
            description: 'CRM data synchronization',
            connected: true
          },
          { 
            name: 'HubSpot', 
            status: 'Not Connected', 
            description: 'Marketing automation',
            connected: false
          },
          { 
            name: 'Slack', 
            status: 'Connected', 
            description: 'Team notifications',
            connected: true
          },
          { 
            name: 'Google Drive', 
            status: 'Connected', 
            description: 'Document storage',
            connected: true
          },
          { 
            name: 'Microsoft Teams', 
            status: 'Not Connected', 
            description: 'Team collaboration',
            connected: false
          },
          { 
            name: 'Zapier', 
            status: 'Not Connected', 
            description: 'Workflow automation',
            connected: false
          }
        ].map((integration, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="font-semibold text-gray-900">{integration.name}</div>
                <div className="text-xs text-gray-500">{integration.description}</div>
              </div>
            </div>
            
            <div className={`text-xs px-2 py-1 rounded-full inline-block mb-3 ${
              integration.connected 
                ? 'bg-green-100 text-green-800' 
                : 'bg-gray-100 text-gray-800'
            }`}>
              {integration.status}
            </div>
            
            <button className={`w-full py-2 px-4 rounded-lg text-sm font-medium transition-colors duration-200 ${
              integration.connected
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-primary-600 text-white hover:bg-primary-700'
            }`}>
              {integration.connected ? 'Disconnect' : 'Connect'}
            </button>
          </div>
        ))}
      </div>

      {/* API Configuration */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">API Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Base URL
            </label>
            <input
              type="text"
              value="https://api.tobi.com/v1"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              readOnly
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Key
            </label>
            <div className="flex space-x-2">
              <input
                type="password"
                value="sk-1234567890abcdef"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                readOnly
              />
              <button className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors duration-200">
                Regenerate
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Email Notifications */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Email Notifications</h3>
          <div className="space-y-4">
            {[
              { name: 'System Alerts', description: 'Critical system notifications', enabled: true },
              { name: 'Data Processing', description: 'Document upload and processing updates', enabled: true },
              { name: 'Agent Performance', description: 'Weekly performance reports', enabled: false },
              { name: 'Security Events', description: 'Login attempts and security issues', enabled: true },
              { name: 'Marketing Updates', description: 'Product updates and features', enabled: false }
            ].map((notification, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{notification.name}</div>
                  <div className="text-sm text-gray-500">{notification.description}</div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    defaultChecked={notification.enabled}
                    className="sr-only peer" 
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Push Notifications */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Push Notifications</h3>
          <div className="space-y-4">
            {[
              { name: 'Real-time Alerts', description: 'Immediate system notifications', enabled: true },
              { name: 'Agent Responses', description: 'When agent completes tasks', enabled: true },
              { name: 'Conversation Updates', description: 'New customer conversations', enabled: false },
              { name: 'Performance Milestones', description: 'Achievement notifications', enabled: true }
            ].map((notification, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{notification.name}</div>
                  <div className="text-sm text-gray-500">{notification.description}</div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    defaultChecked={notification.enabled}
                    className="sr-only peer" 
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                </label>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Notification Schedule */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Notification Schedule</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quiet Hours Start
            </label>
            <input
              type="time"
              defaultValue="22:00"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quiet Hours End
            </label>
            <input
              type="time"
              defaultValue="08:00"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>
        
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Weekend Notifications
          </label>
          <div className="flex items-center space-x-4">
            <label className="flex items-center">
              <input type="checkbox" className="mr-2" />
              <span className="text-sm text-gray-700">Saturday</span>
            </label>
            <label className="flex items-center">
              <input type="checkbox" className="mr-2" />
              <span className="text-sm text-gray-700">Sunday</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-1">
        <div className="flex space-x-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 whitespace-nowrap ${
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
        {activeTab === 'general' && renderGeneralTab()}
        {activeTab === 'security' && renderSecurityTab()}
        {activeTab === 'integrations' && renderIntegrationsTab()}
        {activeTab === 'notifications' && renderNotificationsTab()}
      </div>
    </div>
  );
};

export default Settings; 
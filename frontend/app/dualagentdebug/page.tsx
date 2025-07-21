"use client";

import { useState, useEffect, useRef } from 'react';
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
interface Message {
  id: string;
  role: 'human' | 'ai' | 'system';
  content: string;
  timestamp: string;
  sources?: Array<{
    content: string;
    metadata: Record<string, any>;
  }>;
}

interface User {
  id: string;
  name: string;
  email: string;
  user_type: 'employee' | 'customer';
  employee_id?: string;
  customer_id?: string;
}

interface ConversationState {
  messages: Message[];
  conversationId: string | null;
  isLoading: boolean;
  error: string | null;
}

interface ChatRequest {
  message: string;
  conversation_id?: string;
  user_id?: string;
  include_sources: boolean;
}

interface ChatResponse {
  message: string;
  conversation_id: string;
  sources: Array<{
    content: string;
    metadata: Record<string, any>;
  }>;
  suggestions: string[];
  metadata: Record<string, any>;
}

interface ConfirmationRequest {
  confirmation_id: string;
  customer_id: string;
  customer_name: string;
  customer_email?: string;
  message_content: string;
  message_type: string;
  requested_by: string;
  requested_at: string;
  expires_at: string;
  status: string;
  conversation_id: string;
}

export default function DualAgentDebugPage() {
  // User selection state
  const [users, setUsers] = useState<User[]>([]);
  const [selectedEmployeeUser, setSelectedEmployeeUser] = useState<User | null>(null);
  const [selectedCustomerUser, setSelectedCustomerUser] = useState<User | null>(null);
  const [usersLoading, setUsersLoading] = useState(true);

  // Chat state for both sides
  const [employeeChat, setEmployeeChat] = useState<ConversationState>({
    messages: [],
    conversationId: null,
    isLoading: false,
    error: null
  });

  const [customerChat, setCustomerChat] = useState<ConversationState>({
    messages: [],
    conversationId: null,
    isLoading: false,
    error: null
  });

  // Input states
  const [employeeInput, setEmployeeInput] = useState('');
  const [customerInput, setCustomerInput] = useState('');

  // Confirmation state
  const [pendingConfirmations, setPendingConfirmations] = useState<ConfirmationRequest[]>([]);

  // Debug state
  const [debugMode, setDebugMode] = useState(false);
  const [showSources, setShowSources] = useState(true);

  // Refs for auto-scroll
  const employeeChatRef = useRef<HTMLDivElement>(null);
  const customerChatRef = useRef<HTMLDivElement>(null);

  // Load users on mount
  useEffect(() => {
    loadUsers();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (employeeChatRef.current) {
      employeeChatRef.current.scrollTop = employeeChatRef.current.scrollHeight;
    }
  }, [employeeChat.messages]);

  useEffect(() => {
    if (customerChatRef.current) {
      customerChatRef.current.scrollTop = customerChatRef.current.scrollHeight;
    }
  }, [customerChat.messages]);

  // Polling for pending confirmations
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (employeeChat.conversationId || customerChat.conversationId) {
      interval = setInterval(() => {
        checkPendingConfirmations();
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [employeeChat.conversationId, customerChat.conversationId]);

  async function loadUsers() {
    try {
      setUsersLoading(true);
      
      // Fetch users with their associated employee/customer IDs
      const { data, error } = await supabase
        .from('users')
        .select(`
          id,
          display_name,
          email,
          user_type,
          employee_id,
          customer_id
        `)
        .order('display_name');

      if (error) throw error;

      // Transform to our User type
      const transformedUsers = data?.map(user => ({
        id: user.id,
        name: user.display_name || user.email,
        email: user.email,
        user_type: user.user_type === 'employee' ? 'employee' as const : 'customer' as const,
        employee_id: user.employee_id,
        customer_id: user.customer_id
      })) || [];

      setUsers(transformedUsers);

      // Auto-select first employee and customer if available
      const employees = transformedUsers.filter(u => u.user_type === 'employee');
      const customers = transformedUsers.filter(u => u.user_type === 'customer');
      
      if (employees.length > 0) setSelectedEmployeeUser(employees[0]);
      if (customers.length > 0) setSelectedCustomerUser(customers[0]);

    } catch (error) {
      console.error('Error loading users:', error);
    } finally {
      setUsersLoading(false);
    }
  }

  async function sendMessage(message: string, userType: 'employee' | 'customer') {
    const isEmployee = userType === 'employee';
    const user = isEmployee ? selectedEmployeeUser : selectedCustomerUser;
    const currentChat = isEmployee ? employeeChat : customerChat;
    const setChat = isEmployee ? setEmployeeChat : setCustomerChat;

    if (!user || !message.trim()) return;

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'human',
      content: message.trim(),
      timestamp: new Date().toISOString()
    };

    setChat(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true,
      error: null
    }));

    try {
      const chatRequest: ChatRequest = {
        message: message.trim(),
        conversation_id: currentChat.conversationId || undefined,
        user_id: user.id,
        include_sources: showSources
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(chatRequest)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.message || 'API request failed');
      }

      const chatResponse: ChatResponse = result.data;

      // Add AI response to chat
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        content: chatResponse.message,
        timestamp: new Date().toISOString(),
        sources: chatResponse.sources
      };

      setChat(prev => ({
        ...prev,
        messages: [...prev.messages, aiMessage],
        conversationId: chatResponse.conversation_id,
        isLoading: false
      }));

    } catch (error) {
      console.error('Error sending message:', error);
      setChat(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      }));
    }
  }

  async function checkPendingConfirmations() {
    const conversationIds = [employeeChat.conversationId, customerChat.conversationId].filter(Boolean);
    
    for (const conversationId of conversationIds) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/chat/confirmation/pending/${conversationId}`);
        if (response.ok) {
          const result = await response.json();
          if (result.success && result.data?.length > 0) {
            setPendingConfirmations(prev => {
              const newConfirmations = result.data.filter((conf: ConfirmationRequest) => 
                !prev.find(p => p.confirmation_id === conf.confirmation_id)
              );
              return [...prev, ...newConfirmations];
            });
          }
        }
      } catch (error) {
        console.error('Error checking confirmations:', error);
      }
    }
  }

  async function respondToConfirmation(confirmationId: string, action: 'approved' | 'cancelled', modifiedMessage?: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat/confirmation/respond`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          confirmation_id: confirmationId,
          action,
          modified_message: modifiedMessage,
          responded_at: new Date().toISOString()
        })
      });

      if (response.ok) {
        setPendingConfirmations(prev => 
          prev.filter(conf => conf.confirmation_id !== confirmationId)
        );
      }
    } catch (error) {
      console.error('Error responding to confirmation:', error);
    }
  }

  function clearChat(userType: 'employee' | 'customer') {
    if (userType === 'employee') {
      setEmployeeChat({
        messages: [],
        conversationId: null,
        isLoading: false,
        error: null
      });
    } else {
      setCustomerChat({
        messages: [],
        conversationId: null,
        isLoading: false,
        error: null
      });
    }
  }

  function renderMessage(message: Message) {
    const isHuman = message.role === 'human';
    return (
      <div
        key={message.id}
        className={`flex ${isHuman ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div
          className={`max-w-[70%] rounded-lg px-4 py-2 ${
            isHuman
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-800'
          }`}
        >
          <div className="text-sm">{message.content}</div>
          <div className={`text-xs mt-1 opacity-70`}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
          
          {/* Sources */}
          {message.sources && message.sources.length > 0 && showSources && (
            <div className="mt-2 pt-2 border-t border-gray-300">
              <div className="text-xs font-semibold mb-1">Sources:</div>
              {message.sources.slice(0, 2).map((source, idx) => (
                <div key={idx} className="text-xs bg-white bg-opacity-20 rounded p-1 mb-1">
                  {source.content.substring(0, 100)}...
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderChatInterface(
    userType: 'employee' | 'customer',
    user: User | null,
    chat: ConversationState,
    input: string,
    setInput: (value: string) => void
  ) {
    const isEmployee = userType === 'employee';
    const bgColor = isEmployee ? 'bg-blue-50' : 'bg-green-50';
    const borderColor = isEmployee ? 'border-blue-200' : 'border-green-200';
    const buttonColor = isEmployee ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-600 hover:bg-green-700';
    
    return (
      <div className={`flex-1 ${bgColor} ${borderColor} border rounded-lg flex flex-col`}>
        {/* Header */}
        <div className={`p-4 border-b ${borderColor} bg-white`}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">
                {isEmployee ? '👨‍💼 Employee' : '👤 Customer'} Chat
              </h3>
              <div className="text-sm text-gray-600">
                User: {user?.name || 'None selected'} ({user?.email})
              </div>
              {chat.conversationId && (
                <div className="text-xs text-gray-500">
                  Conversation: {chat.conversationId.substring(0, 8)}...
                </div>
              )}
            </div>
            <button
              onClick={() => clearChat(userType)}
              className="text-red-600 hover:text-red-800 text-sm"
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={isEmployee ? employeeChatRef : customerChatRef}
          className="flex-1 p-4 overflow-y-auto min-h-[400px] max-h-[600px]"
        >
          {chat.messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              <p>No messages yet. Start a conversation!</p>
              <div className="mt-4 space-y-2 text-left max-w-md mx-auto">
                <p className="text-sm font-semibold">Try asking:</p>
                <ul className="text-xs space-y-1">
                  {isEmployee ? (
                    <>
                      <li>• "Show me all active customers"</li>
                      <li>• "Send a follow-up message to John Doe"</li>
                      <li>• "What vehicles do we have in stock?"</li>
                    </>
                  ) : (
                    <>
                      <li>• "What cars do you have available?"</li>
                      <li>• "Tell me about your pricing"</li>
                      <li>• "What's your financing options?"</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          ) : (
            chat.messages.map(renderMessage)
          )}

          {chat.isLoading && (
            <div className="flex justify-start mb-4">
              <div className="bg-gray-100 rounded-lg px-4 py-2">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}

          {chat.error && (
            <div className="bg-red-100 border border-red-300 rounded-lg p-3 mb-4">
              <div className="text-red-700 text-sm">
                <strong>Error:</strong> {chat.error}
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className={`p-4 border-t ${borderColor} bg-white`}>
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !chat.isLoading && user) {
                  sendMessage(input, userType);
                  setInput('');
                }
              }}
              placeholder={`Type your message as ${userType}...`}
              disabled={!user || chat.isLoading}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={() => {
                if (input && !chat.isLoading && user) {
                  sendMessage(input, userType);
                  setInput('');
                }
              }}
              disabled={!user || !input.trim() || chat.isLoading}
              className={`px-4 py-2 ${buttonColor} text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            🚀 Dual Agent Debug Interface
          </h1>
          <p className="text-gray-600">
            Test customer and employee chat experiences side-by-side
          </p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Employee User Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Employee User
              </label>
              <select
                value={selectedEmployeeUser?.id || ''}
                onChange={(e) => {
                  const user = users.find(u => u.id === e.target.value);
                  setSelectedEmployeeUser(user || null);
                }}
                disabled={usersLoading}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select Employee...</option>
                {users.filter(u => u.user_type === 'employee').map(user => (
                  <option key={user.id} value={user.id}>
                    {user.name} ({user.email})
                  </option>
                ))}
              </select>
            </div>

            {/* Customer User Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Customer User
              </label>
              <select
                value={selectedCustomerUser?.id || ''}
                onChange={(e) => {
                  const user = users.find(u => u.id === e.target.value);
                  setSelectedCustomerUser(user || null);
                }}
                disabled={usersLoading}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="">Select Customer...</option>
                {users.filter(u => u.user_type === 'customer').map(user => (
                  <option key={user.id} value={user.id}>
                    {user.name} ({user.email})
                  </option>
                ))}
              </select>
            </div>

            {/* Debug Options */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Debug Options
              </label>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={showSources}
                    onChange={(e) => setShowSources(e.target.checked)}
                    className="mr-2"
                  />
                  <span className="text-sm">Show Sources</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={debugMode}
                    onChange={(e) => setDebugMode(e.target.checked)}
                    className="mr-2"
                  />
                  <span className="text-sm">Debug Mode</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Pending Confirmations */}
        {pendingConfirmations.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <h3 className="text-lg font-semibold text-yellow-800 mb-3">
              🔔 Pending Confirmations ({pendingConfirmations.length})
            </h3>
            <div className="space-y-3">
              {pendingConfirmations.map((conf) => (
                <div key={conf.confirmation_id} className="bg-white rounded-lg p-4 border border-yellow-300">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="font-medium">Customer Message Request</h4>
                      <p className="text-sm text-gray-600">
                        To: {conf.customer_name} ({conf.customer_email})
                      </p>
                      <p className="text-sm text-gray-600">
                        Type: {conf.message_type}
                      </p>
                    </div>
                    <span className="text-xs text-gray-500">
                      {new Date(conf.expires_at).toLocaleString()}
                    </span>
                  </div>
                  <div className="bg-gray-50 rounded p-3 mb-3">
                    <p className="text-sm">{conf.message_content}</p>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => respondToConfirmation(conf.confirmation_id, 'approved')}
                      className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => respondToConfirmation(conf.confirmation_id, 'cancelled')}
                      className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chat Interfaces */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {renderChatInterface(
            'employee',
            selectedEmployeeUser,
            employeeChat,
            employeeInput,
            setEmployeeInput
          )}
          
          {renderChatInterface(
            'customer',
            selectedCustomerUser,
            customerChat,
            customerInput,
            setCustomerInput
          )}
        </div>

        {/* Debug Information */}
        {debugMode && (
          <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4">🔍 Debug Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-medium mb-2">Employee State</h4>
                <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto">
                  {JSON.stringify({
                    user: selectedEmployeeUser,
                    conversationId: employeeChat.conversationId,
                    messageCount: employeeChat.messages.length,
                    isLoading: employeeChat.isLoading,
                    error: employeeChat.error
                  }, null, 2)}
                </pre>
              </div>
              <div>
                <h4 className="font-medium mb-2">Customer State</h4>
                <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto">
                  {JSON.stringify({
                    user: selectedCustomerUser,
                    conversationId: customerChat.conversationId,
                    messageCount: customerChat.messages.length,
                    isLoading: customerChat.isLoading,
                    error: customerChat.error
                  }, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* Status */}
        <div className="mt-6 text-center text-sm text-gray-500">
          <p>
            API: {API_BASE_URL} | 
            Users: {users.length} loaded | 
            Confirmations: {pendingConfirmations.length} pending
          </p>
        </div>
      </div>
    </div>
  );
} 
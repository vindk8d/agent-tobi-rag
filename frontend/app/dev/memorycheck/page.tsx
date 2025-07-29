'use client';

import { useState, useEffect, useRef } from 'react';
import { createClient } from '@supabase/supabase-js';
import { 
  User, 
  LongTermMemory, 
  ConversationSummary, 
  CustomerData, 
  DatabaseMessage,
  MemoryType,
  ConversationSummaryType,
  ConsolidationStatus 
} from '@/types';

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

const supabase = supabaseUrl && supabaseAnonKey ? 
  createClient(supabaseUrl, supabaseAnonKey) : null;

// Chat Message Interface
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// User Summary Interface (simplified from master summary)
interface UserSummary {
  user_id: string;
  latest_summary: string;
  conversation_count: number;
  has_history: boolean;
  latest_conversation_id?: string;
}

// Chat Component
function ChatInterface({ selectedUserId }: { selectedUserId?: string }) {
  const [messages, setChatMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isChatMinimized, setIsChatMinimized] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load recent messages when user is selected
  useEffect(() => {
    if (selectedUserId) {
      loadRecentMessages();
    } else {
      // Show welcome message when no user is selected
      setChatMessages([
        {
          id: `welcome-${Date.now()}`,
          role: 'assistant',
          content: "Hello! I'm your AI assistant. I can help you with sales questions, analyze customer data, and provide insights based on our memory system. Please select a user from the memory debug interface to continue.",
          timestamp: new Date()
        }
      ]);
      setCurrentConversationId(null);
    }
  }, [selectedUserId]);

  const loadRecentMessages = async () => {
    if (!selectedUserId) return;

    try {
      // SINGLE CONVERSATION PER USER: Load messages from the user's single ongoing conversation
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/chat/recent-messages/${selectedUserId}`);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.success && result.data && result.data.length > 0) {
          // Convert backend messages to frontend format
          const loadedMessages = result.data.map((msg: any) => ({
            id: msg.id,
            role: msg.role === 'human' ? 'user' : 'assistant',
            content: msg.content,
            timestamp: new Date(msg.timestamp)
          }));
          
          setChatMessages(loadedMessages);
          
          // Set conversation ID from messages (single conversation per user)
          if (result.data.length > 0) {
            setCurrentConversationId(result.data[result.data.length - 1].conversation_id);
          }
        } else {
          // Show welcome message for user with no conversation history
          setChatMessages([
            {
              id: `welcome-user-${Date.now()}`,
              role: 'assistant',
              content: "Hello! I'm ready to assist you with sales questions, customer data analysis, and insights. How can I help you today?",
              timestamp: new Date()
            }
          ]);
        }
      } else {
        console.warn('Failed to load recent messages:', response.status);
        // Fall back to welcome message
        setChatMessages([
          {
            id: `welcome-fallback-${Date.now()}`,
            role: 'assistant',  
            content: "Hello! I'm ready to assist you with sales questions, customer data analysis, and insights. How can I help you today?",
            timestamp: new Date()
          }
        ]);
      }
    } catch (error) {
      console.error('Error loading recent messages:', error);
      // Fall back to welcome message
      setChatMessages([
        {
          id: `welcome-error-${Date.now()}`,
          role: 'assistant',
          content: "Hello! I'm ready to assist you with sales questions, customer data analysis, and insights. How can I help you today?",
          timestamp: new Date()
        }
      ]);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !selectedUserId) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Call the actual sales copilot agent
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          conversation_id: currentConversationId || undefined,
          user_id: selectedUserId || undefined,
          include_sources: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Debug logging
      console.log('Chat API response:', result);
      console.log('Response structure:', {
        hasMessage: !!result.message,
        hasConversationId: !!result.conversation_id,
        hasSuccess: !!result.success,
        hasData: !!result.data
      });
      
      // Handle both direct ChatResponse format and wrapped format
      let responseData;
      if (result.success && result.data) {
        // Wrapped format
        responseData = result.data;
      } else if (result.message && result.conversation_id) {
        // Direct ChatResponse format
        responseData = result;
      } else {
        throw new Error('Invalid response format from agent');
      }

      const assistantMessage: ChatMessage = {
        id: `${responseData.conversation_id}-${Date.now()}`, // Unique ID combining conversation ID and timestamp
        role: 'assistant',
        content: responseData.message,
        timestamp: new Date()
      };

      setChatMessages(prev => [...prev, assistantMessage]);
      
      // Track the conversation ID for deletion
      if (responseData.conversation_id && !currentConversationId) {
        setCurrentConversationId(responseData.conversation_id);
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I apologize, but I encountered an error processing your request. Please try again.",
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleDeleteConversation = async () => {
    if (!currentConversationId || isDeleting || !selectedUserId) return;

    try {
      setIsDeleting(true);
      
      // SINGLE CONVERSATION PER USER: Delete the user's single conversation
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/chat/conversations/${currentConversationId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Reset chat state - user will start fresh with a new conversation
      setCurrentConversationId(null);
      await loadRecentMessages();
      
      console.log('User conversation cleared successfully');
    } catch (error) {
      console.error('Error deleting conversation:', error);
      // You could add a toast notification here if desired
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className={`fixed right-4 top-0 bg-white border-l border-gray-200 shadow-lg transition-all duration-300 ease-in-out z-50 ${
      isChatMinimized ? 'w-16' : 'w-124'
    }`} style={{ height: 'calc(100vh - 55px)' }}>
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
        {!isChatMinimized && (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <h3 className="text-lg font-semibold text-gray-900">AI Assistant</h3>
          </div>
        )}
        <div className="flex items-center space-x-1">
          {!isChatMinimized && currentConversationId && (
            <button
              onClick={handleDeleteConversation}
              disabled={isDeleting}
              className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors disabled:opacity-50"
              title="Delete conversation"
            >
              {isDeleting ? (
                <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              )}
            </button>
          )}
          <button
            onClick={() => setIsChatMinimized(!isChatMinimized)}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
          >
            {isChatMinimized ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 12H4" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {!isChatMinimized && (
        <>
                     {/* Messages Area */}
           <div className="flex-1 overflow-y-auto p-4 space-y-4" style={{ height: 'calc(100vh - 240px)' }}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900 border border-gray-200'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  <p className={`text-xs mt-1 ${
                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 border border-gray-200 rounded-lg px-4 py-2 max-w-xs">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-xs text-gray-500">AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4">
            <div className="flex space-x-2">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={selectedUserId ? "Ask about sales data, customers, or get insights..." : "Please select a user first"}
                className="flex-1 resize-none rounded-md border border-gray-300 px-3 py-2 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                rows={2}
                disabled={isLoading || !selectedUserId}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading || !selectedUserId}
                className="self-end inline-flex items-center justify-center w-10 h-10 rounded-md bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? (
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                )}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {selectedUserId ? "Press Enter to send, Shift+Enter for new line" : "Select a user from the debug interface to start chatting"}
            </p>
          </div>
        </>
      )}
    </div>
  );
}

export default function MemoryCheckPage() {
  // State management
  const [selectedUserId, setSelectedUserId] = useState<string>('');
  const [users, setUsers] = useState<User[]>([]);
  const [userCrmData, setUserCrmData] = useState<CustomerData | null>(null);
  const [longTermMemories, setLongTermMemories] = useState<LongTermMemory[]>([]);
  const [masterSummaries, setMasterSummaries] = useState<UserSummary[]>([]);
  const [conversationSummaries, setConversationSummaries] = useState<ConversationSummary[]>([]);
  const [messages, setMessages] = useState<DatabaseMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Additional state for debugging tools
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<LongTermMemory[]>([]);
  const [consolidationStatus, setConsolidationStatus] = useState<string | null>(null);
  const [debugMode, setDebugMode] = useState(false);
  
  // Memory configuration state (loaded from backend)
  const [memoryConfig, setMemoryConfig] = useState({
    memory_summary_interval: 10, // Default fallback value
    memory_max_messages: 12,
    memory_auto_summarize: true
  });
  
  // Individual loading states for hot reloading
  const [loadingStates, setLoadingStates] = useState({
    userData: false,
    memoryData: false,
    masterSummaryData: false,
    conversationData: false,
    messages: false
  });
  
  // Last update timestamps
  const [lastUpdated, setLastUpdated] = useState({
    userData: null as Date | null,
    memoryData: null as Date | null,
    masterSummaryData: null as Date | null,
    conversationData: null as Date | null,
    messages: null as Date | null
  });

  // Load memory configuration from backend
  const loadMemoryConfig = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/memory/config`);
      const result = await response.json();
      
      if (result.success && result.data) {
        setMemoryConfig(result.data);
        console.log('✅ Memory configuration loaded:', result.data);
      } else {
        console.warn('⚠️ Failed to load memory configuration, using defaults');
      }
    } catch (error) {
      console.error('❌ Error loading memory configuration:', error);
      // Keep using default values on error
    }
  };

  // Load users and memory config on component mount
  useEffect(() => {
    if (supabase) {
      loadUsers();
      loadMemoryConfig();
    }
  }, []);

  // Load user-specific data when user is selected
  useEffect(() => {
    if (selectedUserId && supabase) {
      loadUserData(selectedUserId);
      loadMemoryData(selectedUserId);
      loadUserSummaryData(selectedUserId);
      loadConversationData(selectedUserId);
      loadMessages(selectedUserId);
    }
  }, [selectedUserId]);

  const loadUsers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Use backend API instead of direct Supabase access to bypass RLS
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/users`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setUsers(result.data || []);
      } else {
        throw new Error(result.message || 'Failed to load users');
      }
    } catch (err) {
      console.error('Error loading users:', err);
      setError(`Failed to load users: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const loadUserData = async (userId: string) => {
    try {
      setLoadingStates(prev => ({ ...prev, userData: true }));
      console.log(`🔍 Loading user CRM data for user: ${userId}`);
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/users/${userId}/crm`);
      const result = await response.json();
      
      console.log(`📊 User CRM API response:`, { 
        status: response.status, 
        success: result.success, 
        dataExists: !!result.data,
        userId: userId,
        endpoint: `${apiUrl}/api/v1/memory-debug/users/${userId}/crm`
      });
      
      if (result.success && result.data) {
        setUserCrmData(result.data);
        console.log(`✅ User CRM data loaded:`, result.data);
      } else {
        setUserCrmData(null);
        console.log(`⚠️ No CRM data found for user: ${userId}`);
      }
      
      setLastUpdated(prev => ({ ...prev, userData: new Date() }));
    } catch (err) {
      console.error('❌ Error loading user CRM data:', err);
      setUserCrmData(null);
    } finally {
      setLoadingStates(prev => ({ ...prev, userData: false }));
    }
  };

  const loadMemoryData = async (userId: string) => {
    try {
      setLoadingStates(prev => ({ ...prev, memoryData: true }));
      console.log(`🧠 Loading long-term memory data for user: ${userId}`);
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/users/${userId}/memories`);
      const result = await response.json();
      
      console.log(`📊 Memory API response:`, { 
        status: response.status, 
        success: result.success, 
        memoryCount: result.data?.length || 0,
        userId: userId,
        endpoint: `${apiUrl}/api/v1/memory-debug/users/${userId}/memories`
      });
      
      if (result.success) {
        setLongTermMemories(result.data || []);
        console.log(`✅ Long-term memories loaded: ${result.data?.length || 0} entries`);
        
        // Log memory types and namespaces for verification
        if (result.data?.length > 0) {
          const memoryTypes = Array.from(new Set(result.data.map((m: any) => m.memory_type)));
          const namespaces = Array.from(new Set(result.data.map((m: any) => m.namespace?.[0])));
          console.log(`📈 Memory breakdown:`, { memoryTypes, namespaces });
        }
      } else {
        console.error('❌ Error loading memory data:', result.message);
        setLongTermMemories([]);
      }
      
      setLastUpdated(prev => ({ ...prev, memoryData: new Date() }));
    } catch (err) {
      console.error('❌ Error loading memory data:', err);
      setLongTermMemories([]);
    } finally {
      setLoadingStates(prev => ({ ...prev, memoryData: false }));
    }
  };

  const loadUserSummaryData = async (userId: string) => {
    try {
      setLoadingStates(prev => ({ ...prev, masterSummaryData: true }));
      console.log(`📋 Loading user summary for user: ${userId}`);
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/users/${userId}/summary`);
      const result = await response.json();
      
      console.log(`📊 User summary API response:`, { 
        status: response.status, 
        success: result.success, 
        hasData: !!result.data,
        userId: userId,
        endpoint: `${apiUrl}/api/v1/memory-debug/users/${userId}/summary`
      });
      
      if (result.success && result.data) {
        setMasterSummaries([result.data]); // Store as array for compatibility
        console.log(`✅ User summary loaded`);
      } else {
        console.error('❌ Error loading user summary:', result.message);
        setMasterSummaries([]);
      }
      
      setLastUpdated(prev => ({ ...prev, masterSummaryData: new Date() }));
    } catch (err) {
      console.error('❌ Error loading master summaries:', err);
      setMasterSummaries([]);
    } finally {
      setLoadingStates(prev => ({ ...prev, masterSummaryData: false }));
    }
  };

  const loadConversationData = async (userId: string) => {
    try {
      setLoadingStates(prev => ({ ...prev, conversationData: true }));
      console.log(`📝 Loading conversation summaries for user: ${userId}`);
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/users/${userId}/conversation-summaries`);
      const result = await response.json();
      
      console.log(`📊 Conversation summaries API response:`, { 
        status: response.status, 
        success: result.success, 
        summaryCount: result.data?.length || 0,
        userId: userId,
        endpoint: `${apiUrl}/api/v1/memory-debug/users/${userId}/conversation-summaries`
      });
      
      if (result.success) {
        setConversationSummaries(result.data || []);
        console.log(`✅ Conversation summaries loaded: ${result.data?.length || 0} entries`);
        
        // Log summary types and message counts for verification
        if (result.data?.length > 0) {
          const summaryTypes = Array.from(new Set(result.data.map((s: any) => s.summary_type)));
          const totalMessages = result.data.reduce((sum: number, s: any) => sum + (s.message_count || 0), 0);
          console.log(`📈 Summary breakdown:`, { summaryTypes, totalMessages });
        }
      } else {
        console.error('❌ Error loading conversation summaries:', result.message);
        setConversationSummaries([]);
      }
      
      setLastUpdated(prev => ({ ...prev, conversationData: new Date() }));
    } catch (err) {
      console.error('❌ Error loading conversation summaries:', err);
      setConversationSummaries([]);
    } finally {
      setLoadingStates(prev => ({ ...prev, conversationData: false }));
    }
  };

  const loadMessages = async (userId: string) => {
    try {
      setLoadingStates(prev => ({ ...prev, messages: true }));
      console.log(`💬 Loading messages for user: ${userId}`);
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/users/${userId}/messages`);
      const result = await response.json();
      
      console.log(`📊 Messages API response:`, { 
        status: response.status, 
        success: result.success, 
        messageCount: result.data?.length || 0,
        userId: userId,
        endpoint: `${apiUrl}/api/v1/memory-debug/users/${userId}/messages`
      });
      
      if (result.success) {
        setMessages(result.data || []);
        console.log(`✅ Messages loaded: ${result.data?.length || 0} entries`);
        
        // Log message roles and conversation breakdown for verification
        if (result.data?.length > 0) {
          const roles = Array.from(new Set(result.data.map((m: any) => m.role)));
          const conversationIds = Array.from(new Set(result.data.map((m: any) => m.conversation_id))).length;
          console.log(`📈 Message breakdown:`, { roles, conversationCount: conversationIds });
        }
      } else {
        console.error('❌ Error loading messages:', result.message);
        setMessages([]);
      }
      
      setLastUpdated(prev => ({ ...prev, messages: new Date() }));
    } catch (err) {
      console.error('❌ Error loading messages:', err);
      setMessages([]);
    } finally {
      setLoadingStates(prev => ({ ...prev, messages: false }));
    }
  };

  // Debugging tool functions
  const searchMemories = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setLoading(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/memory/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          user_id: selectedUserId || undefined,
          similarity_threshold: 0.5,
          limit: 10
        })
      });
      
      const result = await response.json();
      if (result.success) {
        setSearchResults(result.data || []);
      } else {
        setError(`Search failed: ${result.message}`);
      }
    } catch (err) {
      console.error('Memory search error:', err);
      setError('Memory search failed');
    } finally {
      setLoading(false);
    }
  };

  const triggerMemoryConsolidation = async () => {
    if (!selectedUserId) {
      setError('Please select a user first');
      return;
    }
    
    try {
      setLoading(true);
      setConsolidationStatus('Triggering consolidation...');
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/memory-debug/memory/consolidate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: selectedUserId,
          force: debugMode
        })
      });
      
      const result = await response.json();
      if (result.success) {
        setConsolidationStatus('Consolidation triggered successfully');
        // Reload memory data to see changes
        await loadMemoryData(selectedUserId);
        await loadConversationData(selectedUserId);
      } else {
        setConsolidationStatus(`Consolidation failed: ${result.message}`);
      }
      
      // Clear status after 3 seconds
      setTimeout(() => setConsolidationStatus(null), 3000);
    } catch (err) {
      console.error('Consolidation error:', err);
      setConsolidationStatus('Consolidation request failed');
      setTimeout(() => setConsolidationStatus(null), 3000);
    } finally {
      setLoading(false);
    }
  };

  const refreshAllData = async () => {
    if (!selectedUserId) return;
    
    setLoading(true);
    await Promise.all([
      loadUserData(selectedUserId),
      loadMemoryData(selectedUserId),
      loadUserSummaryData(selectedUserId),
      loadConversationData(selectedUserId),
      loadMessages(selectedUserId)
    ]);
    setLoading(false);
  };

  const clearSearchResults = () => {
    setSearchQuery('');
    setSearchResults([]);
  };

  // Calculate countdown to next summarization
  const getMessagesUntilSummarization = (): number => {
    if (messages.length === 0) return memoryConfig.memory_summary_interval;
    
    // Get the current conversation message count for the selected user
    const currentConversationMessages = messages.length;
    
    // Calculate how many more messages needed until next summarization
    const messagesInCurrentCycle = currentConversationMessages % memoryConfig.memory_summary_interval;
    const messagesUntilNext = messagesInCurrentCycle === 0 ? 0 : memoryConfig.memory_summary_interval - messagesInCurrentCycle;
    
    return messagesUntilNext;
  };

  // Set up real-time subscriptions and periodic refresh
  useEffect(() => {
    if (!selectedUserId || !supabase) return;

    console.log(`🔄 Setting up real-time subscriptions for user: ${selectedUserId}`);

    // Real-time subscriptions for all relevant tables
    const memorySubscription = supabase
      .channel('long-term-memories-realtime')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'long_term_memories' },
        (payload) => {
          console.log('🧠 Memory change received:', payload);
          // Check if the change affects the selected user
          if (payload.new && (payload.new as any).namespace && 
              JSON.stringify((payload.new as any).namespace).includes(selectedUserId)) {
            console.log('🔄 Reloading memory data...');
            loadMemoryData(selectedUserId);
          } else if (payload.old && (payload.old as any).namespace && 
                     JSON.stringify((payload.old as any).namespace).includes(selectedUserId)) {
            console.log('🔄 Reloading memory data (deletion)...');
            loadMemoryData(selectedUserId);
          }
        }
      )
      .subscribe();

    const messageSubscription = supabase
      .channel('messages-realtime')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'messages' },
        (payload) => {
          console.log('💬 Message change received:', payload);
          // Check if the change affects the selected user
          if ((payload.new && (payload.new as any).user_id === selectedUserId) ||
              (payload.old && (payload.old as any).user_id === selectedUserId)) {
            console.log('🔄 Reloading messages...');
            loadMessages(selectedUserId);
          }
        }
      )
      .subscribe();

    const conversationSummariesSubscription = supabase
      .channel('conversation-summaries-realtime')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'conversation_summaries' },
        (payload) => {
          console.log('📝 Conversation summary change received:', payload);
          // Check if the change affects the selected user
          if ((payload.new && (payload.new as any).user_id === selectedUserId) ||
              (payload.old && (payload.old as any).user_id === selectedUserId)) {
            console.log('🔄 Reloading conversation summaries...');
            loadConversationData(selectedUserId);
          }
        }
      )
      .subscribe();

    const conversationsSubscription = supabase
      .channel('conversations-realtime')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'conversations' },
        (payload) => {
          console.log('🗨️ Conversation change received:', payload);
          // Check if the change affects the selected user
          if ((payload.new && (payload.new as any).user_id === selectedUserId) ||
              (payload.old && (payload.old as any).user_id === selectedUserId)) {
            console.log('🔄 Reloading conversation data...');
            loadConversationData(selectedUserId);
            loadMessages(selectedUserId);
          }
        }
      )
      .subscribe();

    const customersSubscription = supabase
      .channel('customers-realtime')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'customers' },
        (payload) => {
          console.log('👤 Customer change received:', payload);
          // Check if the change affects the selected user
          if ((payload.new && (payload.new as any).user_id === selectedUserId) ||
              (payload.old && (payload.old as any).user_id === selectedUserId)) {
            console.log('🔄 Reloading user CRM data...');
            loadUserData(selectedUserId);
          }
        }
      )
      .subscribe();

    const masterSummariesSubscription = supabase
      .channel('master-summaries-realtime')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'user_master_summaries' },
        (payload) => {
          console.log('📋 Master summary change received:', payload);
          // Check if the change affects the selected user
          if ((payload.new && (payload.new as any).user_id === selectedUserId) ||
              (payload.old && (payload.old as any).user_id === selectedUserId)) {
            console.log('🔄 Reloading user summary...');
            loadUserSummaryData(selectedUserId);
          }
        }
      )
      .subscribe();

    // Periodic refresh as backup (every 30 seconds)
    const refreshInterval = setInterval(() => {
      console.log('🔄 Periodic refresh triggered...');
      refreshAllData();
    }, 30000);

    // Initial load with a small delay to ensure subscriptions are ready
    setTimeout(() => {
      console.log('🎯 Initial data load after subscriptions setup...');
      refreshAllData();
    }, 1000);

    return () => {
      console.log('🛑 Cleaning up subscriptions...');
      memorySubscription.unsubscribe();
      messageSubscription.unsubscribe(); 
      conversationSummariesSubscription.unsubscribe();
      conversationsSubscription.unsubscribe();
      customersSubscription.unsubscribe();
      masterSummariesSubscription.unsubscribe();
      clearInterval(refreshInterval);
    };
  }, [selectedUserId]);

  if (!supabase) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600">Configuration Error</h1>
          <p className="text-gray-600 mt-2">
            Supabase environment variables are not configured. Please check your .env.local file.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Chat Interface - Fixed positioned */}
      <ChatInterface selectedUserId={selectedUserId} />

      {/* Main Content - Adjusted for chat interface */}
      <div style={{ marginRight: '400px' }}> {/* Adjust for chat width + right padding */}
        {/* Header */}
        <div className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <h1 className="text-2xl font-bold text-gray-900">Memory Debug Interface</h1>
                <div className="hidden md:block">
                  <span className="text-sm text-gray-500">
                    Long-term Memory & Conversation Analysis
                  </span>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-gray-600">Live</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span className="text-xs text-gray-500">Hot Reload Active</span>
                </div>
              </div>
            </div>
        </div>
      </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* User Selection */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">👤 User Selection</h2>
            <div className="flex items-center space-x-4">
              <select
                value={selectedUserId}
                onChange={(e) => setSelectedUserId(e.target.value)}
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                disabled={loading}
              >
                <option value="">Select a user for memory analysis</option>
                {users.map((user) => (
                  <option key={user.id} value={user.id}>
                    {user.name || user.email} ({user.role || 'user'}) - {user.id.slice(0, 8)}
                  </option>
                ))}
              </select>
              <button
                onClick={loadUsers}
                disabled={loading}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
              >
                {loading ? (
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-700" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <svg className="-ml-1 mr-2 h-4 w-4 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                )}
                Refresh
              </button>
            </div>
          </div>

          {selectedUserId && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* User Data Display */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">👥 User Information</h2>
                  <div className="flex items-center space-x-2">
                    {loadingStates.userData && (
                      <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    {lastUpdated.userData && (
                      <span className="text-xs text-gray-500">
                        {lastUpdated.userData.toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
                {userCrmData ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Name</label>
                        <p className="mt-1 text-sm text-gray-900">{userCrmData.first_name} {userCrmData.last_name}</p>
                      </div>
                      <div>
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Status</label>
                        <span className={`mt-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          userCrmData.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {userCrmData.status}
                        </span>
                      </div>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Contact</label>
                      <p className="mt-1 text-sm text-gray-900">{userCrmData.email}</p>
                      {userCrmData.phone && <p className="text-sm text-gray-900">{userCrmData.phone}</p>}
                    </div>
                    {userCrmData.warmth_score && (
                      <div>
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Warmth Score</label>
                        <div className="mt-1 flex items-center">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full" 
                              style={{width: `${Math.min(userCrmData.warmth_score * 10, 100)}%`}}
                            ></div>
                          </div>
                          <span className="ml-2 text-sm text-gray-600">{userCrmData.warmth_score.toFixed(1)}/10</span>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-4">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    <p className="mt-2 text-sm text-gray-600">No CRM data found for this user</p>
                  </div>
                )}
              </div>

              {/* Memory Statistics */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">Memory Statistics</h2>
                  <div className="flex items-center space-x-2">
                    {(loadingStates.memoryData || loadingStates.masterSummaryData || loadingStates.conversationData || loadingStates.messages) && (
                      <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    {lastUpdated.memoryData && (
                      <span className="text-xs text-gray-500">
                        {lastUpdated.memoryData.toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{longTermMemories.length}</div>
                    <div className="text-xs text-gray-500">Long-term Memories</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{getMessagesUntilSummarization()}</div>
                    <div className="text-xs text-gray-500">Messages Until Summary</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{conversationSummaries.length}</div>
                    <div className="text-xs text-gray-500">Conversation Summaries</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {longTermMemories.reduce((sum, mem) => sum + mem.access_count, 0)}
                    </div>
                    <div className="text-xs text-gray-500">Total Access Count</div>
                  </div>
                </div>
              </div>

              {/* Long-term Memory Visualization */}
              <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">Long-term Memory Entries</h2>
                  <div className="flex items-center space-x-2">
                    {loadingStates.memoryData && (
                      <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    <span className="text-sm text-gray-600">
                      {longTermMemories.length} entries
                    </span>
                    {lastUpdated.memoryData && (
                      <span className="text-xs text-gray-500">
                        Last updated: {lastUpdated.memoryData.toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
                {longTermMemories.length > 0 ? (
                  <div className="overflow-hidden">
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {longTermMemories.map((memory) => (
                        <div key={memory.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                          <div className="flex justify-between items-start mb-2">
                            <div>
                              <h3 className="text-sm font-medium text-gray-900">{memory.key}</h3>
                              <p className="text-xs text-gray-500">
                                Namespace: {memory.namespace.join(' → ')}
                              </p>
                            </div>
                            <div className="text-right">
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                                memory.memory_type === 'semantic' ? 'bg-blue-100 text-blue-800' :
                                memory.memory_type === 'episodic' ? 'bg-green-100 text-green-800' :
                                'bg-purple-100 text-purple-800'
                              }`}>
                                {memory.memory_type}
                              </span>
                              <p className="text-xs text-gray-500 mt-1">
                                Accessed {memory.access_count} times
                              </p>
                            </div>
                          </div>
                          <div className="text-sm text-gray-700 mb-2">
                            <pre className="whitespace-pre-wrap text-xs bg-gray-50 p-2 rounded">
                              {JSON.stringify(memory.value, null, 2)}
                            </pre>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500">
                            <span>Created: {new Date(memory.created_at).toLocaleString()}</span>
                            <span>Last accessed: {new Date(memory.accessed_at).toLocaleString()}</span>
                          </div>
                          {memory.expiry_at && (
                            <div className="mt-1 text-xs text-orange-600">
                              Expires: {new Date(memory.expiry_at).toLocaleString()}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <p className="mt-2 text-sm text-gray-600">No long-term memories found for this user</p>
                  </div>
                )}
              </div>

              {/* Latest Conversation Summary */}
              <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">📋 Latest Conversation Summary</h2>
                  <div className="flex items-center space-x-2">
                    {loadingStates.masterSummaryData && (
                      <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    <span className="text-sm text-gray-600">
                      {masterSummaries.length > 0 ? 'Available' : 'None available'}
                    </span>
                    {lastUpdated.masterSummaryData && (
                      <span className="text-xs text-gray-500">
                        Last updated: {lastUpdated.masterSummaryData.toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
                {masterSummaries.length > 0 ? (
                  <div className="space-y-4">
                    {masterSummaries.map((summary, index) => (
                      <div key={`${summary.user_id}-${index}`} className="border border-gray-200 rounded-lg p-6 bg-gradient-to-br from-purple-50 to-blue-50">
                        <div className="flex justify-between items-start mb-4">
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                            <h3 className="text-lg font-medium text-gray-900">Latest Conversation Summary</h3>
                          </div>
                          <div className="text-right">
                            <div className="flex items-center space-x-2 text-sm text-gray-600 mb-1">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                              </svg>
                              <span>{summary.conversation_count} conversations</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              <span>{summary.has_history ? 'Has History' : 'No History'}</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
                          <p className="text-gray-800 leading-relaxed whitespace-pre-wrap text-sm">
                            {summary.latest_summary || 'No conversation history available yet. Start chatting to generate summaries.'}
                          </p>
                        </div>
                        
                        <div className="flex justify-between items-center mt-4 pt-4 border-t border-gray-200">
                          <div className="flex items-center space-x-2 text-xs text-gray-500">
                            {summary.latest_conversation_id && (
                              <div className="flex items-center space-x-1">
                                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span>From conversation: {summary.latest_conversation_id.slice(0, 8)}...</span>
                              </div>
                            )}
                          </div>
                          <div className="flex items-center space-x-1 text-xs">
                            <div className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full font-medium">
                              Latest Conversation
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="mt-2 text-sm text-gray-600">No conversation summary available yet</p>
                    <p className="text-xs text-gray-500 mt-1">Conversation summaries are generated automatically after every {memoryConfig.memory_summary_interval} messages in a conversation</p>
                  </div>
                )}
              </div>

              {/* Conversation Summaries */}
            <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">📝 Conversation Summaries</h2>
                  <div className="flex items-center space-x-2">
                    {loadingStates.conversationData && (
                      <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    <span className="text-sm text-gray-600">
                      {conversationSummaries.length} summaries
                    </span>
                    {lastUpdated.conversationData && (
                      <span className="text-xs text-gray-500">
                        Last generated: {lastUpdated.conversationData.toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
                {conversationSummaries.length > 0 ? (
                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {conversationSummaries.map((summary) => (
                      <div key={summary.id} className="border border-gray-200 rounded-lg p-3">
                        <div className="flex justify-between items-start mb-2">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            summary.summary_type === 'periodic' ? 'bg-blue-100 text-blue-800' :
                            summary.summary_type === 'final' ? 'bg-green-100 text-green-800' :
                            'bg-purple-100 text-purple-800'
                          }`}>
                            {summary.summary_type}
                          </span>
                          <span className="text-xs text-gray-500">
                            {summary.message_count} messages
                          </span>
                        </div>
                        <p className="text-sm text-gray-700 mb-2">{summary.summary_text}</p>
                        <div className="text-xs text-gray-500">
                          {new Date(summary.created_at).toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6">
                    <svg className="mx-auto h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <p className="mt-2 text-sm text-gray-600">No conversation summaries found</p>
                  </div>
                )}
              </div>

              

              {/* Memory Debugging Tools */}
              <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6 mt-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Memory Debugging Tools</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Memory Search Tool */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-700">Memory Search</h3>
                    <div className="flex space-x-2">
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search memories by key..."
                        className="flex-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                        onKeyPress={(e) => e.key === 'Enter' && searchMemories()}
                      />
                      <button
                        onClick={searchMemories}
                        disabled={loading || !searchQuery.trim()}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
                      >
                        Search
                      </button>
                    </div>
                    {searchResults.length > 0 && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-gray-500">
                            Found {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
                          </span>
                          <button
                            onClick={clearSearchResults}
                            className="text-xs text-primary-600 hover:text-primary-800"
                          >
                            Clear
                          </button>
                        </div>
                        <div className="max-h-32 overflow-y-auto space-y-1">
                          {searchResults.map((result) => (
                            <div key={result.id} className="text-xs bg-gray-50 p-2 rounded">
                              <div className="font-medium">{result.key}</div>
                              <div className="text-gray-600">
                                {result.namespace.join(' → ')} | {result.memory_type}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Memory Consolidation Tool */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-700">Memory Consolidation</h3>
                    <div className="space-y-2">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="debug-mode"
                          checked={debugMode}
                          onChange={(e) => setDebugMode(e.target.checked)}
                          className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label htmlFor="debug-mode" className="ml-2 block text-sm text-gray-700">
                          Force consolidation (Debug Mode)
                        </label>
                      </div>
                      <button
                        onClick={triggerMemoryConsolidation}
                        disabled={loading || !selectedUserId}
                        className="w-full inline-flex justify-center items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
                      >
                        Trigger Consolidation
                      </button>
                      {consolidationStatus && (
                        <div className={`text-xs p-2 rounded ${
                          consolidationStatus.includes('success') ? 'bg-green-50 text-green-700' : 
                          consolidationStatus.includes('fail') ? 'bg-red-50 text-red-700' : 
                          'bg-blue-50 text-blue-700'
                        }`}>
                          {consolidationStatus}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Data Refresh Tool */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-700">Data Management</h3>
                    <button
                      onClick={refreshAllData}
                      disabled={loading || !selectedUserId}
                      className="w-full inline-flex justify-center items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
                    >
                      {loading ? (
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-700" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : (
                        <svg className="-ml-1 mr-2 h-4 w-4 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                      )}
                      Refresh All Data
                    </button>
                  </div>

                  {/* System Stats */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-700">Quick Stats</h3>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-gray-50 p-2 rounded text-center">
                        <div className="font-bold text-blue-600">{longTermMemories.length}</div>
                        <div className="text-gray-500">Memories</div>
                      </div>
                      <div className="bg-gray-50 p-2 rounded text-center">
                        <div className="font-bold text-purple-600">{getMessagesUntilSummarization()}</div>
                        <div className="text-gray-500">Until Sum.</div>
                      </div>
                      <div className="bg-gray-50 p-2 rounded text-center">
                        <div className="font-bold text-green-600">{conversationSummaries.length}</div>
                        <div className="text-gray-500">Conv. Sum.</div>
                      </div>
                      <div className="bg-gray-50 p-2 rounded text-center">
                        <div className="font-bold text-orange-600">{messages.length}</div>
                        <div className="text-gray-500">Messages</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex items-center text-xs text-gray-500">
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Debug tools allow manual testing of memory system operations. Use debug mode to force operations regardless of normal triggers.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 
'use client';

import { useState, useEffect } from 'react';
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

export default function MemoryCheckPage() {
  // State management
  const [selectedUserId, setSelectedUserId] = useState<string>('');
  const [users, setUsers] = useState<User[]>([]);
  const [userCrmData, setUserCrmData] = useState<CustomerData | null>(null);
  const [longTermMemories, setLongTermMemories] = useState<LongTermMemory[]>([]);
  const [conversationSummaries, setConversationSummaries] = useState<ConversationSummary[]>([]);
  const [messages, setMessages] = useState<DatabaseMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Additional state for debugging tools
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<LongTermMemory[]>([]);
  const [consolidationStatus, setConsolidationStatus] = useState<string | null>(null);
  const [debugMode, setDebugMode] = useState(false);

  // Load users on component mount
  useEffect(() => {
    if (supabase) {
      loadUsers();
    }
  }, []);

  // Load user-specific data when user is selected
  useEffect(() => {
    if (selectedUserId && supabase) {
      loadUserData(selectedUserId);
      loadMemoryData(selectedUserId);
      loadConversationData(selectedUserId);
      loadMessages(selectedUserId);
    }
  }, [selectedUserId]);

  const loadUsers = async () => {
    try {
      setLoading(true);
      // Load from users table if it exists, otherwise fallback to customers
      const { data: usersData, error: usersError } = await supabase!
        .from('users')
        .select('*')
        .order('created_at', { ascending: false });

      if (usersError && usersError.message.includes('relation "users" does not exist')) {
        // Fallback to customers table from CRM
        const { data: customersData, error: customersError } = await supabase!
          .from('customers')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(20);

        if (customersError) throw customersError;
        
        // Convert customers to user format
        const customerUsers = customersData?.map(customer => ({
          id: customer.id,
          email: customer.email || `customer_${customer.id}`,
          name: `${customer.first_name} ${customer.last_name}`,
          role: 'customer',
          created_at: customer.created_at
        })) || [];

        setUsers(customerUsers);
      } else {
        if (usersError) throw usersError;
        setUsers(usersData || []);
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
      // Try to load CRM data for the user
      const { data: crmData, error: crmError } = await supabase!
        .from('customers')
        .select('*, branches(*)')
        .eq('id', userId)
        .single();

      if (!crmError && crmData) {
        setUserCrmData(crmData);
      } else {
        setUserCrmData(null);
      }
    } catch (err) {
      console.error('Error loading user CRM data:', err);
    }
  };

  const loadMemoryData = async (userId: string) => {
    try {
      // Query long-term memories using a text search on namespace array
      // Since namespace is stored as array, we use textual contains for user ID
      const { data: memoryData, error: memoryError } = await supabase!
        .from('long_term_memories')
        .select('*')
        .or(`namespace.cs.{"${userId}"},namespace.cs.{user,${userId}}`)
        .order('accessed_at', { ascending: false });

      if (memoryError) throw memoryError;
      setLongTermMemories(memoryData || []);
    } catch (err) {
      console.error('Error loading memory data:', err);
      setLongTermMemories([]);
    }
  };

  const loadConversationData = async (userId: string) => {
    try {
      const { data: summaryData, error: summaryError } = await supabase!
        .from('conversation_summaries')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (summaryError) throw summaryError;
      setConversationSummaries(summaryData || []);
    } catch (err) {
      console.error('Error loading conversation summaries:', err);
      setConversationSummaries([]);
    }
  };

  const loadMessages = async (userId: string) => {
    try {
      // Get conversations for this user first
      const { data: conversations, error: convError } = await supabase!
        .from('conversations')
        .select('id')
        .eq('user_id', userId)
        .limit(5)
        .order('created_at', { ascending: false });

      if (convError) throw convError;

      if (conversations && conversations.length > 0) {
        const conversationIds = conversations.map(c => c.id);
        
        const { data: messagesData, error: messagesError } = await supabase!
          .from('messages')
          .select('*')
          .in('conversation_id', conversationIds)
          .order('created_at', { ascending: false })
          .limit(50);

        if (messagesError) throw messagesError;
        setMessages(messagesData || []);
      } else {
        setMessages([]);
      }
    } catch (err) {
      console.error('Error loading messages:', err);
      setMessages([]);
    }
  };

  // Debugging tool functions
  const searchMemories = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/memory-debug/memory/search`, {
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
      
      const response = await fetch(`/api/v1/memory-debug/memory/consolidate`, {
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
      loadConversationData(selectedUserId),
      loadMessages(selectedUserId)
    ]);
    setLoading(false);
  };

  const clearSearchResults = () => {
    setSearchQuery('');
    setSearchResults([]);
  };

  // Set up real-time subscriptions
  useEffect(() => {
    if (!selectedUserId || !supabase) return;

    const memorySubscription = supabase
      .channel('long-term-memories')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'long_term_memories' },
        (payload) => {
          console.log('Memory change received:', payload);
          if (payload.new && (payload.new as any).namespace && 
              JSON.stringify((payload.new as any).namespace).includes(selectedUserId)) {
            loadMemoryData(selectedUserId);
          }
        }
      )
      .subscribe();

    const messageSubscription = supabase
      .channel('messages')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'messages' },
        (payload) => {
          console.log('Message change received:', payload);
          loadMessages(selectedUserId);
        }
      )
      .subscribe();

    return () => {
      memorySubscription.unsubscribe();
      messageSubscription.unsubscribe();
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
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">🧠 Memory Debug Interface</h1>
              <div className="hidden md:block">
                <span className="text-sm text-gray-500">
                  Long-term Memory & Conversation Analysis
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">Live</span>
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
              <h2 className="text-lg font-semibold text-gray-900 mb-4">👥 User Information</h2>
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
              <h2 className="text-lg font-semibold text-gray-900 mb-4">📊 Memory Statistics</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{longTermMemories.length}</div>
                  <div className="text-xs text-gray-500">Long-term Memories</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{conversationSummaries.length}</div>
                  <div className="text-xs text-gray-500">Conversation Summaries</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">{messages.length}</div>
                  <div className="text-xs text-gray-500">Recent Messages</div>
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
              <h2 className="text-lg font-semibold text-gray-900 mb-4">🧠 Long-term Memory Entries</h2>
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

            {/* Conversation Summaries */}
            <div className="lg:col-span-1 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">📝 Conversation Summaries</h2>
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

            {/* Recent Messages */}
            <div className="lg:col-span-1 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">💬 Recent Messages</h2>
              {messages.length > 0 ? (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {messages.map((message) => (
                    <div key={message.id} className={`p-3 rounded-lg ${
                      message.role === 'user' ? 'bg-blue-50 ml-4' : 
                      message.role === 'assistant' ? 'bg-gray-50 mr-4' : 
                      'bg-yellow-50'
                    }`}>
                      <div className="flex justify-between items-start mb-1">
                        <span className={`text-xs font-medium ${
                          message.role === 'user' ? 'text-blue-700' : 
                          message.role === 'assistant' ? 'text-gray-700' : 
                          'text-yellow-700'
                        }`}>
                          {message.role}
                        </span>
                        <span className="text-xs text-gray-500">
                          {new Date(message.created_at).toLocaleString()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-800 break-words">
                        {message.content.length > 150 
                          ? `${message.content.substring(0, 150)}...` 
                          : message.content
                        }
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6">
                  <svg className="mx-auto h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2M4 13h2m0 0V9a2 2 0 012-2h2m0 0V6a2 2 0 012-2h2.586a1 1 0 01.707.293l2.414 2.414A1 1 0 0017 7v2a2 2 0 012 2v2M4 13h2m0 0v3a2 2 0 002 2h2" />
                  </svg>
                  <p className="mt-2 text-sm text-gray-600">No recent messages found</p>
                </div>
              )}
            </div>

            {/* Memory Debugging Tools */}
            <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6 mt-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">🛠️ Memory Debugging Tools</h2>
              
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
                      <div className="font-bold text-green-600">{conversationSummaries.length}</div>
                      <div className="text-gray-500">Summaries</div>
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
  );
} 
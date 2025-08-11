"use client";

import { useState, useEffect, useRef, useCallback } from 'react';
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'placeholder-key';
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
  is_interrupted?: boolean;
  confirmation_id?: string;
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
  
  // Filtered customers based on selected employee
  const [filteredCustomerUsers, setFilteredCustomerUsers] = useState<User[]>([]);
  const [customersLoading, setCustomersLoading] = useState(false);

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
  const [shouldPollConfirmations, setShouldPollConfirmations] = useState(true);

  // Debug state
  const [debugMode, setDebugMode] = useState(false);
  const [showSources, setShowSources] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  
  // Refs for auto-scroll
  const employeeChatRef = useRef<HTMLDivElement>(null);
  const customerChatRef = useRef<HTMLDivElement>(null);

  // Helper function to normalize role names for consistent matching
  const normalizeRole = useCallback((role: string): string => {
    if (role === 'user') return 'human';
    if (role === 'assistant') return 'ai';
    return role;
  }, []);

  // Stable message loading functions
  const loadEmployeeMessages = useCallback(async () => {
    if (!selectedEmployeeUser) return;

    try {
      setEmployeeChat(prev => ({ ...prev, isLoading: true, error: null }));

      const url = `${API_BASE_URL}/api/v1/memory-debug/users/${selectedEmployeeUser.id}/messages`;
      const response = await fetch(url);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.success && result.data) {
          const existingMessages: Message[] = result.data.map((msg: any) => ({
            id: msg.id,
            role: normalizeRole(msg.role) as 'human' | 'ai' | 'system',
            content: msg.content,
            timestamp: msg.created_at,
            sources: []
          })).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

          const conversationId = result.data.length > 0 ? result.data[0].conversation_id : null;

          setEmployeeChat({
            messages: existingMessages,
            conversationId: conversationId,
            isLoading: false,
            error: null
          });
        } else {
          setEmployeeChat({
            messages: [],
            conversationId: null,
            isLoading: false,
            error: null
          });
        }
      } else {
        setEmployeeChat(prev => ({
          ...prev,
          messages: [],
          conversationId: null,
          isLoading: false
        }));
      }
    } catch (error) {
      console.error('Error loading employee messages:', error);
      setEmployeeChat(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load messages'
      }));
    }
  }, [selectedEmployeeUser, normalizeRole]);

  const loadCustomerMessages = useCallback(async () => {
    if (!selectedCustomerUser) return;

    try {
      setCustomerChat(prev => ({ ...prev, isLoading: true, error: null }));

      const url = `${API_BASE_URL}/api/v1/memory-debug/users/${selectedCustomerUser.id}/messages`;
      const response = await fetch(url);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.success && result.data) {
          const existingMessages: Message[] = result.data.map((msg: any) => ({
            id: msg.id,
            role: normalizeRole(msg.role) as 'human' | 'ai' | 'system',
            content: msg.content,
            timestamp: msg.created_at,
            sources: []
          })).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

          const conversationId = result.data.length > 0 ? result.data[0].conversation_id : null;

          setCustomerChat({
            messages: existingMessages,
            conversationId: conversationId,
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
      } else {
        setCustomerChat(prev => ({
          ...prev,
          messages: [],
          conversationId: null,
          isLoading: false
        }));
      }
    } catch (error) {
      console.error('Error loading customer messages:', error);
      setCustomerChat(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load messages'
      }));
    }
  }, [selectedCustomerUser, normalizeRole]);

  // Load users on mount
  useEffect(() => {
    loadUsers();
  }, []);

  // Load existing messages when customer is selected
  useEffect(() => {
    if (selectedCustomerUser) {
      loadCustomerMessages();
    }
  }, [selectedCustomerUser, loadCustomerMessages]);

  // Load existing messages when employee is selected
  useEffect(() => {
    if (selectedEmployeeUser) {
      loadEmployeeMessages();
    }
  }, [selectedEmployeeUser, loadEmployeeMessages]);

  // Fetch customers for selected employee
  useEffect(() => {
    if (selectedEmployeeUser && users.length > 0) {
      fetchCustomersForEmployee();
    } else {
      setFilteredCustomerUsers([]);
      setSelectedCustomerUser(null);
    }
  }, [selectedEmployeeUser, users]);

  // Refresh messages after sending (no constant polling)
  const refreshMessages = useCallback(async (userType?: 'employee' | 'customer') => {
    if (userType === 'employee' && selectedEmployeeUser) {
      await loadEmployeeMessages();
    } else if (userType === 'customer' && selectedCustomerUser) {
      await loadCustomerMessages();
    } else {
      // Refresh both if no specific type
      if (selectedEmployeeUser) await loadEmployeeMessages();
      if (selectedCustomerUser) await loadCustomerMessages();
    }
    setLastUpdate(new Date().toLocaleTimeString());
  }, [selectedEmployeeUser, selectedCustomerUser, loadEmployeeMessages, loadCustomerMessages]);

  // Track if we're currently refreshing to prevent multiple simultaneous refreshes
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Simple realtime subscription to detect new messages
  useEffect(() => {
    if (!employeeChat.conversationId && !customerChat.conversationId) {
      return;
    }

    console.log('🔔 Setting up simple realtime subscription for new messages');
    
    const channel = supabase
      .channel('new-messages')
      .on('postgres_changes', 
        { 
          event: 'INSERT',
          schema: 'public', 
          table: 'messages'
        }, 
        (payload) => {
          console.log('📨 New message detected:', payload);
          const newMessage = payload.new as any;
          
          // Check if this message belongs to either current conversation
          const isEmployeeConversation = employeeChat.conversationId && 
            newMessage?.conversation_id === employeeChat.conversationId;
          const isCustomerConversation = customerChat.conversationId && 
            newMessage?.conversation_id === customerChat.conversationId;
          
          if ((isEmployeeConversation || isCustomerConversation) && !isRefreshing) {
            console.log('✅ Message belongs to current conversation - refreshing');
            
            // Prevent multiple simultaneous refreshes
            setIsRefreshing(true);
            
            // Small delay to ensure message is fully persisted, then refresh
            setTimeout(async () => {
              try {
                if (isEmployeeConversation) {
                  await loadEmployeeMessages();
                }
                if (isCustomerConversation) {
                  await loadCustomerMessages();
                }
                setLastUpdate(new Date().toLocaleTimeString());
              } finally {
                setIsRefreshing(false);
              }
            }, 800); // Slightly longer delay to ensure message is fully persisted
          }
        }
      )
      .subscribe((status) => {
        console.log('🔔 Realtime subscription status:', status);
      });

    return () => {
      console.log('🧹 Cleaning up realtime subscription');
      supabase.removeChannel(channel);
    };
  }, [employeeChat.conversationId, customerChat.conversationId, loadEmployeeMessages, loadCustomerMessages, isRefreshing]);

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
    if ((employeeChat.conversationId || customerChat.conversationId) && shouldPollConfirmations) {
      interval = setInterval(() => {
        checkPendingConfirmations();
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [employeeChat.conversationId, customerChat.conversationId, shouldPollConfirmations]);

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
        user_type: user.user_type || (user.customer_id ? 'customer' as const : user.employee_id ? 'employee' as const : 'customer' as const),
        employee_id: user.employee_id,
        customer_id: user.customer_id
      })) || [];

      setUsers(transformedUsers);

      // Auto-select first employee (customer will be auto-selected after filtering)
      const employees = transformedUsers.filter(u => u.user_type === 'employee');
      
      if (employees.length > 0) setSelectedEmployeeUser(employees[0]);

    } catch (error) {
      console.error('Error loading users:', error);
    } finally {
      setUsersLoading(false);
    }
  }

  async function fetchCustomersForEmployee() {
    if (!selectedEmployeeUser?.employee_id) {
      setFilteredCustomerUsers([]);
      setSelectedCustomerUser(null);
      return;
    }

    setCustomersLoading(true);
    try {
      // First, get unique customer IDs for this employee from opportunities
      const customerIdsResult = await supabase
        .from('opportunities')
        .select('customer_id')
        .eq('opportunity_salesperson_ae_id', selectedEmployeeUser.employee_id);

      if (customerIdsResult.error) {
        console.error('Error fetching customer IDs:', customerIdsResult.error);
        setFilteredCustomerUsers([]);
        return;
      }

      const uniqueCustomerIds = Array.from(new Set(customerIdsResult.data?.map(opp => opp.customer_id) || []));
      
      if (uniqueCustomerIds.length === 0) {
        setFilteredCustomerUsers([]);
        setSelectedCustomerUser(null);
        return;
      }

      // Then get users who are customers and have customer_ids in our list
      const customerUsers = users.filter(user => 
        user.user_type === 'customer' && 
        user.customer_id && 
        uniqueCustomerIds.includes(user.customer_id)
      );

      setFilteredCustomerUsers(customerUsers);
      
      // If currently selected customer is not in the filtered list, clear selection
      if (selectedCustomerUser && !customerUsers.find(u => u.id === selectedCustomerUser.id)) {
        setSelectedCustomerUser(null);
      }
      
      // Auto-select first customer if none is selected and we have customers available
      if (!selectedCustomerUser && customerUsers.length > 0) {
        setSelectedCustomerUser(customerUsers[0]);
      }
      
    } catch (error) {
      console.error('Error fetching customers for employee:', error);
      setFilteredCustomerUsers([]);
    } finally {
      setCustomersLoading(false);
    }
  }

  async function clearChat(userType: 'employee' | 'customer') {
    const user = userType === 'employee' ? selectedEmployeeUser : selectedCustomerUser;
    
    if (!user) {
      console.warn(`No ${userType} user selected`);
      return;
    }

    // Clear local state immediately for better UX
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

    // Clear the user's conversation via API
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat/user/${user.id}/all-conversations`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const result = await response.json();
        console.log(`Successfully cleared ${userType} conversation:`, result.message);
      } else {
        const errorData = await response.json();
        console.error(`Failed to clear ${userType} conversation:`, errorData.detail || 'Unknown error');
      }
    } catch (apiError) {
      console.error(`API error clearing ${userType} conversation:`, apiError);
    }
  }

  async function sendMessage(message: string, userType: 'employee' | 'customer') {
    // Restart confirmation polling when sending a new message
    setShouldPollConfirmations(true);
    
    const isEmployee = userType === 'employee';
    const user = isEmployee ? selectedEmployeeUser : selectedCustomerUser;
    const currentChat = isEmployee ? employeeChat : customerChat;
    const setChat = isEmployee ? setEmployeeChat : setCustomerChat;

    if (!user || !message.trim()) {
      return;
    }

    // Add user message to chat immediately for better UX
    const userMessage: Message = {
      id: `temp_user_${Date.now()}`,
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
      
      // Handle both direct ChatResponse and wrapped APIResponse formats
      let chatResponse: ChatResponse;
      if (result.data) {
        if (!result.success) {
          throw new Error(result.message || 'API request failed');
        }
        chatResponse = result.data;
      } else {
        chatResponse = result;
      }

      // Update conversation ID if it changed
      if (chatResponse.conversation_id && chatResponse.conversation_id !== currentChat.conversationId) {
        setChat(prev => ({
          ...prev,
          conversationId: chatResponse.conversation_id
        }));
      }

      // Check if this is an interrupted response (confirmation request)
      if (chatResponse.is_interrupted && chatResponse.confirmation_id) {
        const confirmationMessage: Message = {
          id: `temp_confirmation_${Date.now()}`,
          role: 'ai',
          content: chatResponse.message,
          timestamp: new Date().toISOString(),
          sources: []
        };

        setChat(prev => ({
          ...prev,
          messages: [...prev.messages, confirmationMessage],
          isLoading: false
        }));

        // Check for pending confirmations (realtime subscription will handle message refresh)
        setTimeout(() => {
          checkPendingConfirmations();
        }, 1000);
        
        return;
      }

      // Add AI response to chat
      const aiMessage: Message = {
        id: `temp_ai_${Date.now()}`,
        role: 'ai',
        content: chatResponse.message,
        timestamp: new Date().toISOString(),
        sources: chatResponse.sources
      };

      setChat(prev => ({
        ...prev,
        messages: [...prev.messages, aiMessage],
        isLoading: false
      }));

      // Note: Realtime subscription will automatically refresh messages when they're persisted to database

    } catch (error) {
      console.error('Error sending message:', error);
      setChat(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      }));
    }
  }

  // Helper function to parse confirmation message and extract details
  function parseConfirmationMessage(rawMessage: string, confirmationId: string, conversationId: string): ConfirmationRequest {
    const customerMatch = rawMessage.match(/\*\*To:\*\*\s*([^(]+)\s*\(([^)]+)\)/);
    const typeMatch = rawMessage.match(/\*\*Type:\*\*\s*([^\n]+)/);
    const messageMatch = rawMessage.match(/\*\*Message:\*\*\s*([^*]+?)(?=\n\n\*\*Instructions)/);
    
    return {
      confirmation_id: confirmationId,
      customer_id: 'parsed_customer',
      customer_name: customerMatch ? customerMatch[1].trim() : 'Unknown Customer',
      customer_email: customerMatch ? customerMatch[2].trim() : '',
      message_content: messageMatch ? messageMatch[1].trim() : 'Unable to parse message',
      message_type: typeMatch ? typeMatch[1].trim() : 'Unknown',
      requested_by: 'employee',
      requested_at: new Date().toISOString(),
      expires_at: new Date(Date.now() + 3600000).toISOString(),
      status: 'pending',
      conversation_id: conversationId
    };
  }

  async function checkPendingConfirmations() {
    const conversationIds = [employeeChat.conversationId, customerChat.conversationId].filter(Boolean);
    
    let foundAnyConfirmations = false;
    
    for (const conversationId of conversationIds) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/chat/confirmation/pending/${conversationId}`);
        if (response.ok) {
          const result = await response.json();
          const rawConfirmations = result.confirmations || result.data || [];
          
          if (rawConfirmations.length > 0) {
            foundAnyConfirmations = true;
            const parsedConfirmations = rawConfirmations.map((conf: any) => 
              parseConfirmationMessage(conf.message, conf.confirmation_id, conversationId)
            );
            
            setPendingConfirmations(prev => {
              const newConfirmations = parsedConfirmations.filter((conf: ConfirmationRequest) => 
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
    
    // Clear stale frontend state if no confirmations found
    if (!foundAnyConfirmations && conversationIds.length > 0) {
      setPendingConfirmations(prev => {
        if (prev.length === 0) {
          setTimeout(() => setShouldPollConfirmations(false), 100);
        }
        return [];
      });
    }
  }

  async function respondToConfirmation(confirmationId: string, action: 'approved' | 'cancelled', modifiedMessage?: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat/confirmation/${confirmationId}/respond`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          action: action === 'approved' ? 'approve' : 'deny',
          modified_message: modifiedMessage
        })
      });

      if (response.ok) {
        setPendingConfirmations(prev => 
          prev.filter(conf => conf.confirmation_id !== confirmationId)
        );
        
        console.log(`✅ Confirmation ${confirmationId} processed successfully`);
      } else {
        console.error('Failed to respond to confirmation:', await response.text());
      }
    } catch (error) {
      console.error('Error responding to confirmation:', error);
    }
  }

  function renderMessage(message: Message) {
    const isHuman = message.role === 'human';
    const isAI = message.role === 'ai';
    const isSystem = message.role === 'system';
    
    let messageStyle = '';
    let containerStyle = '';
    
    if (isHuman) {
      containerStyle = 'justify-end';
      messageStyle = 'bg-blue-600 text-white';
    } else if (isAI) {
      containerStyle = 'justify-start';
      messageStyle = 'bg-green-600 text-white';
    } else if (isSystem) {
      containerStyle = 'justify-start';
      messageStyle = 'bg-gray-500 text-white';
    } else {
      containerStyle = 'justify-start';
      messageStyle = 'bg-gray-100 text-gray-800';
    }
    
    return (
      <div
        key={message.id}
        className={`flex ${containerStyle} mb-4`}
      >
        <div
          className={`max-w-[70%] rounded-lg px-4 py-2 ${messageStyle}`}
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
            <div className="flex space-x-2">
              <button
                onClick={() => clearChat(userType)}
                className="text-red-600 hover:text-red-800 text-sm"
                title="Clear this user's conversation history"
              >
                Clear
              </button>
            </div>
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
              <div className="bg-green-600 text-white rounded-lg px-4 py-2">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
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
            Dual Agent Debug Interface
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
                  const user = filteredCustomerUsers.find(u => u.id === e.target.value);
                  setSelectedCustomerUser(user || null);
                }}
                disabled={usersLoading || customersLoading || !selectedEmployeeUser}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="">
                  {!selectedEmployeeUser 
                    ? "Select Employee First..." 
                    : customersLoading 
                    ? "Loading Customers..." 
                    : filteredCustomerUsers.length === 0 
                    ? "No Customers for Employee" 
                    : "Select Customer..."
                  }
                </option>
                {filteredCustomerUsers.map(user => (
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
                
                {/* Realtime Status & Manual Refresh */}
                <div className="pt-2 border-t border-gray-200 space-y-2">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-xs font-medium">
                      Realtime: {(employeeChat.conversationId || customerChat.conversationId) ? 'Active' : 'Waiting'}
                    </span>
                  </div>
                  <button
                    onClick={async () => {
                      if (!isRefreshing) {
                        setIsRefreshing(true);
                        try {
                          await refreshMessages();
                        } finally {
                          setIsRefreshing(false);
                        }
                      }
                    }}
                    disabled={isRefreshing}
                    className="flex items-center space-x-2 px-3 py-1 bg-blue-100 hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed rounded text-xs font-medium text-blue-800"
                    title="Manually refresh messages for both chats"
                  >
                    <span>{isRefreshing ? 'Refreshing...' : 'Manual Refresh'}</span>
                  </button>
                  {lastUpdate && (
                    <div className="text-xs text-gray-500">
                      Last update: {lastUpdate}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Pending Confirmations */}
        {pendingConfirmations.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <h3 className="text-lg font-semibold text-yellow-800 mb-3">
              Pending Confirmations ({pendingConfirmations.length})
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
            Filtered Customers: {filteredCustomerUsers.length} | 
            Selected: {selectedEmployeeUser ? `Employee(${selectedEmployeeUser.name})` : 'No Employee'} & {selectedCustomerUser ? `Customer(${selectedCustomerUser.name})` : 'No Customer'} |
            Confirmations: {pendingConfirmations.length} pending |
            Mode: <span className="font-medium text-green-600">Realtime + On-Demand</span>
            {isRefreshing && (
              <> | <span className="text-blue-600 font-medium">Refreshing...</span></>
            )}
            {lastUpdate && (
              <> | Last refresh: {lastUpdate}</>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
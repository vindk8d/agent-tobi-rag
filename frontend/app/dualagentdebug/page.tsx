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

  // Hot reload state
  const [realtimeStatus, setRealtimeStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [lastMessageUpdate, setLastMessageUpdate] = useState<string | null>(null);
  const [hotReloadNotification, setHotReloadNotification] = useState<string | null>(null);

  // Refs for auto-scroll
  const employeeChatRef = useRef<HTMLDivElement>(null);
  const customerChatRef = useRef<HTMLDivElement>(null);
  
  // Refs to avoid stale closures in realtime subscriptions
  const employeeConversationRef = useRef<string | null>(null);
  const customerConversationRef = useRef<string | null>(null);
  const selectedEmployeeRef = useRef<User | null>(null);
  const selectedCustomerRef = useRef<User | null>(null);

  // Load users on mount
  useEffect(() => {
    loadUsers();
  }, []);

  // Update refs when values change
  useEffect(() => {
    employeeConversationRef.current = employeeChat.conversationId;
    customerConversationRef.current = customerChat.conversationId;
    selectedEmployeeRef.current = selectedEmployeeUser;
    selectedCustomerRef.current = selectedCustomerUser;
  }, [employeeChat.conversationId, customerChat.conversationId, selectedEmployeeUser, selectedCustomerUser]);

  // Load existing messages when customer is selected
  useEffect(() => {
    if (selectedCustomerUser) {
      loadCustomerMessages();
    }
  }, [selectedCustomerUser]);

  // Load existing messages when employee is selected
  useEffect(() => {
    if (selectedEmployeeUser) {
      loadEmployeeMessages();
    }
  }, [selectedEmployeeUser]);

  // Set up realtime subscriptions for hot reload
  useEffect(() => {
    // Only set up subscriptions if we have users selected
    if (!selectedEmployeeUser && !selectedCustomerUser) {
      return;
    }

    console.log('🔄 Setting up realtime subscriptions for hot reload');
    console.log('👨‍💼 Employee user:', selectedEmployeeUser?.id);
    console.log('👤 Customer user:', selectedCustomerUser?.id);

    // Set status to connecting
    setRealtimeStatus('connecting');

    // Create separate channels for employee and customer to avoid conflicts
    const channels: any[] = [];
    let connectedChannels = 0;
    const totalChannels = (selectedEmployeeUser ? 1 : 0) + (selectedCustomerUser ? 1 : 0) + 1; // +1 for generic

    const checkAllConnected = () => {
      connectedChannels++;
      if (connectedChannels >= totalChannels) {
        setRealtimeStatus('connected');
      }
    };

    // Employee messages subscription
    if (selectedEmployeeUser) {
      const employeeChannel = supabase
        .channel(`employee-messages-${selectedEmployeeUser.id}`)
        .on('postgres_changes', 
          { 
            event: '*', // Listen to INSERT, UPDATE, DELETE
            schema: 'public', 
            table: 'messages'
          }, 
          (payload) => {
            console.log('🔔 Employee message change detected:', payload);
            setLastMessageUpdate(`Employee: ${new Date().toLocaleTimeString()}`);
            
            // Check if this message belongs to the current employee's conversation
            if (payload.new && selectedEmployeeRef.current && employeeConversationRef.current) {
              const newMessage = payload.new as any;
              console.log(`👨‍💼 Employee: Checking message conversation_id ${newMessage?.conversation_id} vs current ${employeeConversationRef.current}`);
              
              if (newMessage?.conversation_id === employeeConversationRef.current) {
                console.log('✅ Employee: Message belongs to current conversation - reloading messages');
                // Immediate reload without delay for better UX
                loadEmployeeMessages();
                
                // Show hot reload notification
                setHotReloadNotification('Employee message updated via hot reload');
                setTimeout(() => setHotReloadNotification(null), 3000);
              } else {
                console.log('❌ Employee: Message from different conversation, ignoring');
              }
            } else if (payload.old && selectedEmployeeRef.current && employeeConversationRef.current) {
              // Handle DELETE events
              const deletedMessage = payload.old as any;
              if (deletedMessage?.conversation_id === employeeConversationRef.current) {
                console.log('🗑️ Employee: Message deleted from current conversation - reloading messages');
                loadEmployeeMessages();
                
                // Show hot reload notification
                setHotReloadNotification('Employee message deleted via hot reload');
                setTimeout(() => setHotReloadNotification(null), 3000);
              }
            }
          }
        )
        .subscribe((status) => {
          console.log('👨‍💼 Employee realtime subscription status:', status);
          if (status === 'SUBSCRIBED') {
            checkAllConnected();
          } else if (status === 'CLOSED' || status === 'CHANNEL_ERROR') {
            setRealtimeStatus('disconnected');
          }
        });
      
      channels.push(employeeChannel);
    }

    // Customer messages subscription  
    if (selectedCustomerUser) {
      const customerChannel = supabase
        .channel(`customer-messages-${selectedCustomerUser.id}`)
        .on('postgres_changes', 
          { 
            event: '*', // Listen to INSERT, UPDATE, DELETE
            schema: 'public', 
            table: 'messages'
          }, 
          (payload) => {
            console.log('🔔 Customer message change detected:', payload);
            setLastMessageUpdate(`Customer: ${new Date().toLocaleTimeString()}`);
            
            // Check if this message belongs to the current customer's conversation
            if (payload.new && selectedCustomerRef.current && customerConversationRef.current) {
              const newMessage = payload.new as any;
              console.log(`👤 Customer: Checking message conversation_id ${newMessage?.conversation_id} vs current ${customerConversationRef.current}`);
              
              if (newMessage?.conversation_id === customerConversationRef.current) {
                console.log('✅ Customer: Message belongs to current conversation - reloading messages');
                // Immediate reload without delay for better UX
                loadCustomerMessages();
                
                // Show hot reload notification
                setHotReloadNotification('Customer message updated via hot reload');
                setTimeout(() => setHotReloadNotification(null), 3000);
              } else {
                console.log('❌ Customer: Message from different conversation, ignoring');
              }
            } else if (payload.old && selectedCustomerRef.current && customerConversationRef.current) {
              // Handle DELETE events
              const deletedMessage = payload.old as any;
              if (deletedMessage?.conversation_id === customerConversationRef.current) {
                console.log('🗑️ Customer: Message deleted from current conversation - reloading messages');
                loadCustomerMessages();
                
                // Show hot reload notification
                setHotReloadNotification('Customer message deleted via hot reload');
                setTimeout(() => setHotReloadNotification(null), 3000);
              }
            }
          }
        )
        .subscribe((status) => {
          console.log('👤 Customer realtime subscription status:', status);
          if (status === 'SUBSCRIBED') {
            checkAllConnected();
          } else if (status === 'CLOSED' || status === 'CHANNEL_ERROR') {
            setRealtimeStatus('disconnected');
          }
        });
      
      channels.push(customerChannel);
    }

    // Generic message subscription for any conversation changes (fallback)
    const genericChannel = supabase
      .channel('all-messages-fallback')
      .on('postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'messages'
        },
        (payload) => {
          console.log('🔔 Generic message change detected:', payload);
          setLastMessageUpdate(`Generic: ${new Date().toLocaleTimeString()}`);
          
          // Check if this message belongs to either current conversation
          if (payload.new) {
            const newMessage = payload.new as any;
            const isEmployeeMessage = employeeConversationRef.current && 
              newMessage?.conversation_id === employeeConversationRef.current;
            const isCustomerMessage = customerConversationRef.current && 
              newMessage?.conversation_id === customerConversationRef.current;
            
            if (isEmployeeMessage) {
              console.log('🔄 Generic: Employee message detected, reloading');
              loadEmployeeMessages();
              setHotReloadNotification('Employee message updated (generic fallback)');
              setTimeout(() => setHotReloadNotification(null), 3000);
            }
            
            if (isCustomerMessage) {
              console.log('🔄 Generic: Customer message detected, reloading');
              loadCustomerMessages();
              setHotReloadNotification('Customer message updated (generic fallback)');
              setTimeout(() => setHotReloadNotification(null), 3000);
            }
          }
        }
      )
      .subscribe((status) => {
        console.log('🌐 Generic realtime subscription status:', status);
        if (status === 'SUBSCRIBED') {
          checkAllConnected();
        } else if (status === 'CLOSED' || status === 'CHANNEL_ERROR') {
          setRealtimeStatus('disconnected');
        }
      });
    
    channels.push(genericChannel);

    // Cleanup function
    return () => {
      console.log('🧹 Cleaning up realtime subscriptions');
      setRealtimeStatus('disconnected');
      setLastMessageUpdate(null);
      channels.forEach(channel => {
        supabase.removeChannel(channel);
      });
    };
  }, [selectedEmployeeUser?.id, selectedCustomerUser?.id]); // Re-run when users change

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

  async function loadCustomerMessages() {
    if (!selectedCustomerUser) return;

    try {
      setCustomerChat(prev => ({ ...prev, isLoading: true, error: null }));

      // SINGLE CONVERSATION PER USER: Always load the user's single conversation
      // No need to track conversation IDs since each user has exactly one conversation
      const url = `${API_BASE_URL}/api/v1/memory-debug/users/${selectedCustomerUser.id}/messages`;
      
      const response = await fetch(url);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.success && result.data) {
          // Transform database messages to our Message format
          const existingMessages: Message[] = result.data.map((msg: any) => {
            // Map database roles to frontend roles
            let role: 'human' | 'ai' | 'system';
            if (msg.role === 'user') {
              role = 'human';
            } else if (msg.role === 'assistant') {
              role = 'ai';
            } else if (msg.role === 'system') {
              role = 'system';
            } else {
              role = 'ai'; // Default fallback
            }
            return {
              id: msg.id,
              role: role,
              content: msg.content,
              timestamp: msg.created_at,
              sources: []
            };
          }).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

          // Get conversation ID from messages (single conversation per user)
          const conversationId = result.data.length > 0 ? result.data[0].conversation_id : null;

          setCustomerChat(prev => ({
            ...prev,
            messages: existingMessages,
            conversationId: conversationId,
            isLoading: false
          }));
        } else {
          setCustomerChat(prev => ({
            ...prev,
            messages: [],
            conversationId: null,
            isLoading: false
          }));
        }
      } else {
        console.warn('Failed to load customer messages:', response.status);
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
  }

  async function loadEmployeeMessages() {
    if (!selectedEmployeeUser) return;

    try {
      setEmployeeChat(prev => ({ ...prev, isLoading: true, error: null }));

      // SINGLE CONVERSATION PER USER: Always load the user's single conversation
      // No need to track conversation IDs since each user has exactly one conversation
      const url = `${API_BASE_URL}/api/v1/memory-debug/users/${selectedEmployeeUser.id}/messages`;
      
      const response = await fetch(url);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.success && result.data) {
          // Transform database messages to our Message format
          const existingMessages: Message[] = result.data.map((msg: any) => {
            // Map database roles to frontend roles
            let role: 'human' | 'ai' | 'system';
            if (msg.role === 'user') {
              role = 'human';
            } else if (msg.role === 'assistant') {
              role = 'ai';
            } else if (msg.role === 'system') {
              role = 'system';
            } else {
              role = 'ai'; // Default fallback
            }
            return {
              id: msg.id,
              role: role,
              content: msg.content,
              timestamp: msg.created_at,
              sources: []
            };
          }).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

          // Get conversation ID from messages (single conversation per user)
          const conversationId = result.data.length > 0 ? result.data[0].conversation_id : null;

          setEmployeeChat(prev => ({
            ...prev,
            messages: existingMessages,
            conversationId: conversationId,
            isLoading: false
          }));
        } else {
          setEmployeeChat(prev => ({
            ...prev,
            messages: [],
            conversationId: null,
            isLoading: false
          }));
        }
      } else {
        console.warn('Failed to load employee messages:', response.status);
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
  }

  async function clearChat(userType: 'employee' | 'customer') {
    const chat = userType === 'employee' ? employeeChat : customerChat;
    const user = userType === 'employee' ? selectedEmployeeUser : selectedCustomerUser;
    
    if (!user) {
      console.warn(`No ${userType} user selected`);
      return;
    }

    try {
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

      // SINGLE CONVERSATION PER USER: Clear the user's single conversation
      console.log(`Clearing conversation for ${userType} user ${user.id} via API`);
      
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
          // Don't throw - local state is already cleared
        }
      } catch (apiError) {
        console.error(`API error clearing ${userType} conversation:`, apiError);
        // Don't throw - local state is already cleared
      }

    } catch (error) {
      console.error(`Error clearing ${userType} chat:`, error);
      // Even if database deletion fails, local state is already cleared
      // User can continue with a fresh conversation
    }
  }

  async function sendMessage(message: string, userType: 'employee' | 'customer') {
    console.log('🚀 sendMessage called with:', { message, userType });
    
    // Restart polling when sending a new message (may create confirmations)
    setShouldPollConfirmations(true);
    
    const isEmployee = userType === 'employee';
    const user = isEmployee ? selectedEmployeeUser : selectedCustomerUser;
    const currentChat = isEmployee ? employeeChat : customerChat;
    const setChat = isEmployee ? setEmployeeChat : setCustomerChat;

    console.log('👤 User selection check:', { 
      isEmployee, 
      selectedEmployeeUser: selectedEmployeeUser?.id, 
      selectedCustomerUser: selectedCustomerUser?.id,
      user: user?.id,
      hasMessage: !!message.trim()
    });

    if (!user || !message.trim()) {
      console.error('❌ sendMessage aborted:', { user: user?.id, hasMessage: !!message.trim() });
      return;
    }

    // Add user message to chat immediately for better UX
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
      console.log('Raw API response:', result);
      
      // Handle both direct ChatResponse and wrapped APIResponse formats
      let chatResponse: ChatResponse;
      if (result.data) {
        // Wrapped format: {success: true, data: ChatResponse}
        if (!result.success) {
          throw new Error(result.message || 'API request failed');
        }
        chatResponse = result.data;
      } else {
        // Direct ChatResponse format
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
        // Add the confirmation message to chat AS A MESSAGE (not just in panel)
        const confirmationMessage: Message = {
          id: (Date.now() + 1).toString(),
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

        // Also trigger a check for pending confirmations (for the panel if needed)
        setTimeout(() => {
          checkPendingConfirmations();
        }, 500);
        
        return; // Done processing the confirmation message
      }

      // Add AI response to chat (normal response)
      console.log('Processing AI message. Content:', chatResponse.message);
      console.log('Full chatResponse:', chatResponse);
      
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
        isLoading: false
      }));

      // Note: Messages are automatically persisted by the backend agent
      // The realtime subscription will handle hot reloading

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
      expires_at: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
      status: 'pending',
      conversation_id: conversationId
    };
  }

  async function checkPendingConfirmations() {
    const conversationIds = [employeeChat.conversationId, customerChat.conversationId].filter(Boolean);
    
    // Track if we found any confirmations across all conversations
    let foundAnyConfirmations = false;
    
    for (const conversationId of conversationIds) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/chat/confirmation/pending/${conversationId}`);
        if (response.ok) {
          const result = await response.json();
          // Handle both formats: backend returns {confirmations: [...]} or {success: true, data: [...]}
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
    
    // CRITICAL FIX: If no confirmations found anywhere, clear stale frontend state
    if (!foundAnyConfirmations && conversationIds.length > 0) {
      setPendingConfirmations(prev => {
        if (prev.length > 0) {
          console.log('🧹 Clearing stale confirmation state - backend has no pending confirmations');
        }
        
        // Stop polling if we had no confirmations and are clearing empty state
        // This means the conversation is complete
        if (prev.length === 0) {
          console.log('⏹️ Stopping confirmation polling - no confirmations found, conversation complete');
          setTimeout(() => setShouldPollConfirmations(false), 100);
        }
        
        return [];
      });
    }
  }

  async function respondToConfirmation(confirmationId: string, action: 'approved' | 'cancelled', modifiedMessage?: string) {
    try {
      // CRITICAL FIX: Find the confirmation BEFORE removing it from state
      const confirmation = pendingConfirmations.find(c => c.confirmation_id === confirmationId);
      
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
        
        console.log(`✅ Confirmation ${confirmationId} processed successfully - backend will resume conversation automatically`);
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
    
    // console.log(`Rendering message ${message.id}: role="${message.role}", isHuman=${isHuman}, isAI=${isAI}`);
    
    // Determine message styling based on role
    let messageStyle = '';
    let containerStyle = '';
    
    if (isHuman) {
      // User messages: right-aligned, blue background
      containerStyle = 'justify-end';
      messageStyle = 'bg-blue-600 text-white';
    } else if (isAI) {
      // AI messages: left-aligned, green background
      containerStyle = 'justify-start';
      messageStyle = 'bg-green-600 text-white';
    } else if (isSystem) {
      // System messages: left-aligned, gray background
      containerStyle = 'justify-start';
      messageStyle = 'bg-gray-500 text-white';
    } else {
      // Default: left-aligned, gray background
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
                  Ongoing Conversation: {chat.conversationId.substring(0, 8)}...
                </div>
              )}
              {!chat.conversationId && chat.messages.length === 0 && (
                <div className="text-xs text-green-600">
                  Ready to start conversation
                </div>
              )}
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => clearChat(userType)}
                className="text-red-600 hover:text-red-800 text-sm"
                title="Clear this user's conversation history"
              >
                Clear Conversation
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
            🚀 Dual Agent Debug Interface
          </h1>
          <p className="text-gray-600">
            Test customer and employee chat experiences side-by-side
          </p>
        </div>

        {/* Hot Reload Notification */}
        {hotReloadNotification && (
          <div className="mb-4 bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            <div className="text-sm text-blue-800 font-medium">
              🔄 {hotReloadNotification}
            </div>
          </div>
        )}

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
                
                {/* Hot Reload Status Indicator */}
                <div className="pt-2 border-t border-gray-200">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      realtimeStatus === 'connected' ? 'bg-green-500' :
                      realtimeStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                      'bg-red-500'
                    }`}></div>
                    <span className="text-xs font-medium">
                      Hot Reload: {realtimeStatus === 'connected' ? 'Connected' :
                      realtimeStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
                    </span>
                  </div>
                  {lastMessageUpdate && (
                    <div className="text-xs text-gray-500 mt-1">
                      Last update: {lastMessageUpdate}
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
            Confirmations: {pendingConfirmations.length} pending |
            Hot Reload: <span className={`font-medium ${
              realtimeStatus === 'connected' ? 'text-green-600' :
              realtimeStatus === 'connecting' ? 'text-yellow-600' :
              'text-red-600'
            }`}>
              {realtimeStatus}
            </span>
            {lastMessageUpdate && (
              <> | Last sync: {lastMessageUpdate}</>
            )}
          </p>
        </div>
      </div>
    </div>
  );
} 
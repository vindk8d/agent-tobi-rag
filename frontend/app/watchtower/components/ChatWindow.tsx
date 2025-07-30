'use client';

import { useState, useEffect, useRef } from 'react';
import { SupabaseClient } from '@supabase/supabase-js';
import { CustomerWithWarmth, Conversation, Message } from '../page';

interface ChatWindowProps {
  customer: CustomerWithWarmth;
  supabase: SupabaseClient;
}

export default function ChatWindow({ customer, supabase }: ChatWindowProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(true); // Default to live mode
  const [hotReloadStatus, setHotReloadStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

  useEffect(() => {
    if (customer) {
      fetchConversations();
    }
  }, [customer]);

  useEffect(() => {
    if (selectedConversation) {
      setMessages(sortMessages(selectedConversation.messages || []));
      setShouldAutoScroll(true); // Reset auto-scroll when switching conversations
      setTimeout(() => scrollToBottom(), 100); // Delay to ensure DOM is ready
    }
  }, [selectedConversation]);

  // Smart auto-scroll - only scroll if user is near bottom or if it's a new message
  useEffect(() => {
    if (shouldAutoScroll && messages.length > 0) {
      scrollToBottom();
    }
  }, [messages, shouldAutoScroll]);

  // Track scroll position to determine if we should auto-scroll
  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    if (!chatContainer) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = chatContainer;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 50; // Within 50px of bottom
      if (shouldAutoScroll !== isNearBottom) {
        console.log('Auto-scroll changed:', isNearBottom ? 'enabled' : 'disabled');
        setShouldAutoScroll(isNearBottom);
      }
    };

    // Initial check
    handleScroll();
    
    chatContainer.addEventListener('scroll', handleScroll, { passive: true });
    return () => chatContainer.removeEventListener('scroll', handleScroll);
  }, [shouldAutoScroll]);

  useEffect(() => {
    if (selectedConversation && isLive) {
      setHotReloadStatus('connecting');
      console.log('Setting up hot reload for conversation:', selectedConversation.id);
      
      // Set up real-time subscription for new messages in this specific conversation
      const messagesSubscription = supabase
        .channel(`messages_${selectedConversation.id}`)
        .on(
          'postgres_changes',
          {
            event: '*',
            schema: 'public',
            table: 'messages',
            filter: `conversation_id=eq.${selectedConversation.id}`,
          },
          (payload) => {
            console.log('Hot reload: message change detected', payload);
            const message = payload.new as Message;
            
            if (payload.eventType === 'INSERT') {
              console.log('Adding new message via hot reload:', message);
              setMessages(prevMessages => {
                // STATE RECONCILIATION: Use robust merge instead of simple duplicate check
                const reconciledMessages = reconcileMessages(prevMessages, [message]);
                return reconciledMessages;
              });
              
              // Enable auto-scroll for new messages (likely from the current conversation)
              setShouldAutoScroll(true);
            } else if (payload.eventType === 'UPDATE') {
              console.log('Updating message:', message);
              setMessages(prevMessages => 
                prevMessages.map(m => m.id === message.id ? message : m)
              );
            } else if (payload.eventType === 'DELETE') {
              console.log('Deleting message:', payload.old);
              setMessages(prevMessages => 
                prevMessages.filter(m => m.id !== payload.old.id)
              );
            }
          }
        )
        .subscribe((status, err) => {
          console.log('Messages subscription status:', status, err);
          if (status === 'SUBSCRIBED') {
            setHotReloadStatus('connected');
            console.log('✅ Hot reload connected for conversation:', selectedConversation.id);
          } else if (status === 'CHANNEL_ERROR') {
            setHotReloadStatus('disconnected');
            console.error('❌ Hot reload error:', err);
          } else {
            setHotReloadStatus('connecting');
            console.log('🔄 Hot reload connecting...', status);
          }
        });

      return () => {
        console.log('Cleaning up hot reload subscription for conversation:', selectedConversation.id);
        messagesSubscription.unsubscribe();
        setHotReloadStatus('disconnected');
      };
    } else {
      console.log('Hot reload disabled or no conversation selected');
      setHotReloadStatus('disconnected');
    }
  }, [selectedConversation, isLive, supabase]);

  // Hot reload setup for conversations
  useEffect(() => {
    if (!customer.user_account || !isLive) {
      return;
    }
    
    const conversationsSubscription = supabase
      .channel(`customer_conversations_${customer.user_account.id}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'conversations',
          filter: `user_id=eq.${customer.user_account.id}`,
        },
        (payload) => {
          console.log('Hot reload: conversation change detected', payload);
          fetchConversations();
        }
      )
      .subscribe();

    return () => {
      conversationsSubscription.unsubscribe();
    };
  }, [customer.user_account, isLive, supabase]);

  const fetchConversations = async () => {
    setLoading(true);
    try {
      // Skip if customer doesn't have a user account
      if (!customer.user_account) {
        setConversations([]);
        setSelectedConversation(null);
        setLoading(false);
        return;
      }

      const { data, error } = await supabase
        .from('conversations')
        .select(`
          id,
          user_id,
          title,
          created_at,
          updated_at,
          messages:messages(
            id,
            conversation_id,
            role,
            content,
            created_at,
            metadata
          )
        `)
        .eq('user_id', customer.user_account.id)
        .order('updated_at', { ascending: false });

      if (error) {
        console.error('Error fetching conversations:', error);
        return;
      }

      const conversationsData = data || [];
      setConversations(conversationsData);

      // Auto-select the most recent conversation
      if (conversationsData.length > 0) {
        setSelectedConversation(conversationsData[0]);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
    } finally {
      setLoading(false);
    }
  };

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
  };

  // Ensure messages are always sorted chronologically
  const sortMessages = (messages: Message[]) => {
    return [...messages].sort((a, b) => 
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );
  };

  // STATE RECONCILIATION: Helper function to merge messages (WhatsApp/Telegram approach)
  const reconcileMessages = (currentMessages: Message[], serverMessages: Message[]): Message[] => {
    console.log('🔄 ChatWindow state reconciliation: merging', currentMessages.length, 'current with', serverMessages.length, 'server messages');
    
    const result: Message[] = [];
    const processedServerIds = new Set<string>();
    
    // Step 1: Add all server messages (they are the source of truth)
    serverMessages.forEach(serverMsg => {
      if (!processedServerIds.has(serverMsg.id)) {
        result.push(serverMsg);
        processedServerIds.add(serverMsg.id);
      }
    });
    
    // Step 2: Handle temp messages - either match with server or keep as orphaned optimistic updates
    currentMessages.forEach(currentMsg => {
      const isTemp = currentMsg.id.startsWith('temp_');
      
      if (!isTemp) {
        // Non-temp message - add if not already present
        if (!result.find(msg => msg.id === currentMsg.id)) {
          result.push(currentMsg);
        }
        return;
      }
      
      // This is a temp message - try to match with server message
      const matchingServerMsg = serverMessages.find(serverMsg => {
        // Match by content, role, and timestamp (within 10 seconds tolerance)
        const contentMatch = serverMsg.content.trim() === currentMsg.content.trim();
        const roleMatch = (
          (serverMsg.role === 'user' && currentMsg.role === 'human') ||
          (serverMsg.role === 'assistant' && currentMsg.role === 'ai') ||
          (serverMsg.role === currentMsg.role)
        );
        const timeDiff = Math.abs(
          new Date(serverMsg.created_at).getTime() - 
          new Date(currentMsg.created_at).getTime()
        );
        const timeMatch = timeDiff < 10000; // 10 second tolerance
        
        return contentMatch && roleMatch && timeMatch;
      });
      
      if (matchingServerMsg) {
        console.log(`✅ ChatWindow: Temp message ${currentMsg.id} matched with server message ${matchingServerMsg.id}`);
        // Server message already added in step 1, temp message is now reconciled
      } else {
        console.log(`⚠️ ChatWindow: Orphaned temp message ${currentMsg.id} - keeping as optimistic update`);
        // Keep orphaned temp message (better than losing user's message)
        result.push(currentMsg);
      }
    });
    
    // Step 3: Sort chronologically and return
    const sortedResult = result.sort((a, b) => 
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );
    
    console.log('🎯 ChatWindow state reconciliation complete:', sortedResult.length, 'final messages');
    return sortedResult;
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== today.getFullYear() ? 'numeric' : undefined,
      });
    }
  };

  const groupMessagesByDate = (messages: Message[]) => {
    const groups: { [date: string]: Message[] } = {};
    
    messages.forEach(message => {
      const date = new Date(message.created_at).toDateString();
      if (!groups[date]) {
        groups[date] = [];
      }
      groups[date].push(message);
    });

    return Object.entries(groups).sort(([a], [b]) => 
      new Date(a).getTime() - new Date(b).getTime()
    );
  };

  const getMessageBubbleStyle = (role: string) => {
    switch (role) {
      case 'user':
      case 'human':
        return 'bg-blue-600 text-white max-w-xs lg:max-w-md rounded-tl-lg rounded-tr-lg rounded-br-lg';
      case 'assistant':
      case 'ai':
        return 'bg-white text-gray-900 max-w-xs lg:max-w-md border border-gray-200 rounded-tl-lg rounded-tr-lg rounded-bl-lg';
      case 'system':
        return 'bg-yellow-50 text-yellow-800 max-w-md border border-yellow-200 rounded-lg';
      default:
        return 'bg-gray-50 text-gray-900 max-w-xs lg:max-w-md border border-gray-300 rounded-tl-lg rounded-tr-lg rounded-br-lg';
    }
  };

  const getMessageLabel = (role: string) => {
    switch (role) {
      case 'user':
      case 'human':
        return customer.name;
      case 'assistant':
      case 'ai':
        return 'AI Assistant';
      case 'system':
        return 'System';
      default:
        return `Unknown (${role})`;
    }
  };

  const getLabelStyle = (role: string) => {
    switch (role) {
      case 'user':
      case 'human':
        return 'text-blue-200 text-xs font-medium mb-1';
      case 'assistant':
      case 'ai':
        return 'text-gray-600 text-xs font-medium mb-1';
      case 'system':
        return 'text-yellow-600 text-xs font-medium mb-1 text-center';
      default:
        return 'text-gray-400 text-xs font-medium mb-1';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h2 className="text-lg font-semibold text-gray-900">Customer Chat with AI Agent</h2>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsLive(!isLive)}
              className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium transition-colors duration-200 ${
                isLive && hotReloadStatus === 'connected'
                  ? 'bg-green-100 text-green-800' 
                  : isLive && hotReloadStatus === 'connecting'
                  ? 'bg-yellow-100 text-yellow-800'
                  : isLive
                  ? 'bg-red-100 text-red-800'
                  : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
              }`}
            >
              <div className={`w-2 h-2 rounded-full ${
                isLive && hotReloadStatus === 'connected'
                  ? 'bg-green-500 animate-pulse' 
                  : isLive && hotReloadStatus === 'connecting'
                  ? 'bg-yellow-500 animate-pulse'
                  : isLive
                  ? 'bg-red-500'
                  : 'bg-gray-400'
              }`}></div>
              <span>
                {!isLive 
                  ? 'Offline' 
                  : hotReloadStatus === 'connected' 
                  ? 'Live' 
                  : hotReloadStatus === 'connecting'
                  ? 'Connecting...'
                  : 'Disconnected'
                }
              </span>
            </button>
            
            {loading && (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
            )}
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 relative" ref={chatContainerRef}>
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
            <p className="text-sm text-gray-500">Loading conversations...</p>
          </div>
        ) : !selectedConversation ? (
          <div className="text-center py-8 text-gray-500">
            <svg className="w-12 h-12 mx-auto text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            {!customer.user_account ? (
              <>
                <p className="text-sm font-medium text-red-600">No User Account</p>
                <p className="text-xs text-gray-400 mt-1">Customer doesn't have a user account for chat conversations</p>
              </>
            ) : (
              <>
                <p className="text-sm">No conversations found</p>
                <p className="text-xs text-gray-400 mt-1">
                  Conversations will appear here when the customer chats with the bot
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {groupMessagesByDate(messages).map(([date, dateMessages]) => (
              <div key={date}>
                {/* Date Separator */}
                <div className="flex items-center justify-center py-2">
                  <div className="bg-gray-100 text-gray-600 text-xs px-3 py-1 rounded-full">
                    {formatDate(date)}
                  </div>
                </div>

                {/* Messages for this date */}
                <div className="space-y-4">
                  {dateMessages.map((message) => (
                    <div key={message.id} className={`flex ${
                      message.role === 'user' || message.role === 'human' ? 'justify-start' : message.role === 'system' ? 'justify-center' : 'justify-end'
                    }`}>
                      {/* Message Bubble */}
                      <div className={`px-4 py-3 rounded-lg shadow-sm ${getMessageBubbleStyle(message.role)}`}>
                        {/* Sender Label */}
                        <div className={getLabelStyle(message.role)}>
                          {getMessageLabel(message.role)}
                        </div>
                        
                        {/* Message Content */}
                        <div className={`text-sm break-words leading-relaxed ${
                          message.role === 'system' ? 'text-center' : ''
                        }`}>
                          {message.content}
                        </div>
                        
                        {/* Timestamp */}
                        <div className={`text-xs mt-2 ${
                          message.role === 'user' || message.role === 'human' ? 'text-blue-200' : 
                          message.role === 'system' ? 'text-yellow-600 text-center' : 'text-gray-500'
                        }`}>
                          {formatTime(message.created_at)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
        
        {/* Scroll to Bottom Button */}
        {!shouldAutoScroll && messages.length > 0 && (
          <button
            onClick={() => {
              setShouldAutoScroll(true);
              scrollToBottom();
            }}
            className="absolute bottom-4 right-4 bg-primary-600 hover:bg-primary-700 text-white rounded-full p-3 shadow-lg transition-all duration-200 animate-bounce"
            title="Scroll to bottom"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
} 
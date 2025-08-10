'use client'

// Removed useChat import - using direct API calls instead
import { useState, useRef, useEffect, useCallback } from 'react'
import { clsx } from 'clsx'
import { createClient } from '@supabase/supabase-js'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import rehypeRaw from 'rehype-raw'

// Supabase client setup (optional)
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
const supabase = supabaseUrl && supabaseAnonKey ? createClient(supabaseUrl, supabaseAnonKey) : null

interface Source {
  title?: string
  url?: string
  content?: string
  relevance_score?: number
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at?: string
  sources?: Source[]
  is_interrupted?: boolean
  hitl_phase?: string
  hitl_prompt?: string
  hitl_context?: any
  error?: boolean
  error_type?: string
  file?: string  // Optional file URL for download
  metadata?: {
    type: string
    phase?: string
    error_type?: string
  }
}

interface HitlState {
  isActive: boolean
  phase?: string
  prompt?: string
  context?: any
  conversationId?: string
}

// Types from watchtower
interface Employee {
  id: string
  name: string
  position: string
  email?: string
  branch_id: string
  is_active: boolean
}

interface Customer {
  id: string
  name: string
  phone?: string
  mobile_number?: string
  email?: string
  company?: string
  is_for_business: boolean
  address?: string
  notes?: string
  created_at: string
}

interface User {
  id: string
  display_name: string
  email: string
  user_type: 'employee' | 'customer'
  is_active: boolean
  created_at: string
}

interface StoredMessage {
  id: string
  conversation_id: string
  role: string
  content: string
  created_at: string
  metadata?: any
}

interface StoredConversation {
  id: string
  user_id: string
  title?: string
  created_at: string
  updated_at: string
}

interface MediaAttachment {
  type: 'image' | 'video' | 'audio' | 'document'
  url: string
  name: string
  size?: number
  mimeType?: string
}

export default function TobiChatPage() {
  const [conversationId, setConversationId] = useState<string | null>(null)
  
  const [hitlState, setHitlState] = useState<HitlState>({ isActive: false })
  const [lastSources, setLastSources] = useState<Source[]>([])
  
  // Authentication and login state
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loginPassword, setLoginPassword] = useState('Admin123')
  const [loginError, setLoginError] = useState('')
  
  // User selection and conversation history
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [users, setUsers] = useState<User[]>([])
  const [conversations, setConversations] = useState<StoredConversation[]>([])
  const [selectedConversation, setSelectedConversation] = useState<StoredConversation | null>(null)
  const [storedMessages, setStoredMessages] = useState<StoredMessage[]>([])
  const [showUserSelector, setShowUserSelector] = useState(false)
  const [showConversationHistory, setShowConversationHistory] = useState(false)
  const [loadingUsers, setLoadingUsers] = useState(false)
  const [loadingConversations, setLoadingConversations] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)

  // Chat state management - following dual agent debug pattern
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Save conversation title when conversation is created or significantly updated
  const updateConversationTitle = async (conversationId: string, firstMessage: string) => {
    if (!supabase || !conversationId) return
    
    try {
      // Generate a title from the first message (truncate to 50 chars)
      const title = firstMessage.length > 50 
        ? firstMessage.substring(0, 47) + '...' 
        : firstMessage
      
      const { error } = await supabase
        .from('conversations')
        .update({ title, updated_at: new Date().toISOString() })
        .eq('id', conversationId)
      
      if (error) {
        console.error('Error updating conversation title:', error)
      } else {
        console.log('✅ Updated conversation title')
      }
    } catch (error) {
      console.error('Error updating conversation title:', error)
    }
  }

  // Direct API call function - following dual agent debug pattern
  const sendMessage = async (message: string) => {
    if (!selectedUser || !message.trim() || loadingHistory) {
      console.log('🔍 [CHAT] Cannot send message - selectedUser:', !!selectedUser, 'message:', !!message.trim(), 'loadingHistory:', loadingHistory)
      return
    }

    // Generate conversation ID only if none exists
    const currentConversationId = conversationId || crypto.randomUUID()
    const isNewConversation = !conversationId
    if (isNewConversation) {
      console.log('🔍 [CHAT] No existing conversation ID, generating new one:', currentConversationId)
      setConversationId(currentConversationId)
    }

    // Add user message immediately
    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: message.trim(),
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    setError(null)

    try {
      const requestBody = {
        message: message.trim(),
        conversation_id: currentConversationId,
        user_id: selectedUser.id,
        include_sources: true
      }

      console.log('🔍 [CHAT] Sending message with conversation_id:', currentConversationId)
      console.log('🔍 [CHAT] Full request body:', requestBody)
      console.log('🔍 [CHAT] User ID being sent to backend:', selectedUser.id)

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      console.log('🔍 [CHAT] Response:', result)

      // Handle both direct ChatResponse and wrapped APIResponse formats
      let chatResponse: any
      if (result.data) {
        if (!result.success) {
          throw new Error(result.message || 'API request failed')
        }
        chatResponse = result.data
      } else {
        chatResponse = result
      }

      // Update conversation ID if changed
      if (chatResponse.conversation_id && chatResponse.conversation_id !== currentConversationId) {
        console.log('🔍 [CHAT] Conversation ID updated by server:', chatResponse.conversation_id)
        setConversationId(chatResponse.conversation_id)
      }

      // Handle HITL interrupt
      if (chatResponse.is_interrupted) {
        console.log('🔍 [CHAT] HITL interrupt detected')
        setHitlState({
          isActive: true,
          phase: chatResponse.hitl_phase,
          prompt: chatResponse.hitl_prompt,
          context: chatResponse.hitl_context,
          conversationId: chatResponse.conversation_id
        })
      } else {
        // Reset HITL state for normal responses
        setHitlState({ isActive: false })
      }

      // Add AI response
      const aiMessage: ChatMessage = {
        id: `ai_${Date.now()}`,
        role: 'assistant',
        content: chatResponse.message,
        sources: chatResponse.sources || [],
        is_interrupted: chatResponse.is_interrupted,
        hitl_phase: chatResponse.hitl_phase,
        hitl_prompt: chatResponse.hitl_prompt,
        hitl_context: chatResponse.hitl_context,
        error: chatResponse.error,
        error_type: chatResponse.error_type
      }

      setMessages(prev => [...prev, aiMessage])

      // FRONTEND FIX: Store both user and AI messages directly to database
      try {
        // Store user message
        await supabase?.from('messages').insert({
          conversation_id: chatResponse.conversation_id,
          user_id: selectedUser.id,
          role: 'human',
          content: message.trim(),
          created_at: new Date().toISOString(),
          metadata: {}
        });

        // Store AI response  
        await supabase?.from('messages').insert({
          conversation_id: chatResponse.conversation_id,
          user_id: selectedUser.id,
          role: 'ai', 
          content: chatResponse.message,
          created_at: new Date().toISOString(),
          metadata: {
            sources: chatResponse.sources,
            is_interrupted: chatResponse.is_interrupted,
            hitl_phase: chatResponse.hitl_phase
          }
        });

        console.log('✅ Messages stored directly to database');
      } catch (storeError) {
        console.warn('❌ Failed to store messages directly:', storeError);
      }

      // Update conversation title if this is the first message
      if (isNewConversation && chatResponse.conversation_id) {
        await updateConversationTitle(chatResponse.conversation_id, message.trim())
      }

    } catch (err) {
      console.error('🔍 [CHAT] Error:', err)
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      sendMessage(input)
      setInput('')
    }
  }

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }, [])

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Load users on component mount
  useEffect(() => {
    fetchUsers()
  }, [])

  // Load conversations when user changes
  useEffect(() => {
    if (selectedUser) {
      fetchConversations()
    } else {
      setConversations([])
      setSelectedConversation(null)
      setStoredMessages([])
    }
  }, [selectedUser])

  // Helper function to normalize role names for consistent matching (from dualagentdebug)
  const normalizeRole = useCallback((role: string): string => {
    if (role === 'user') return 'human';
    if (role === 'assistant') return 'ai';
    return role;
  }, []);

  // Load messages function - filter messages by user_id only
  const loadUserMessages = useCallback(async () => {
    if (!selectedUser || !supabase) return;

    try {
      setLoadingHistory(true);
      setError(null);

      // Direct query to messages table filtering by user_id
      const { data: messagesData, error: messagesError } = await supabase
        .from('messages')
        .select('*')
        .eq('user_id', selectedUser.id)
        .order('created_at', { ascending: false })
        .limit(50);

      if (messagesError) {
        console.error('Messages query error:', messagesError);
        setMessages([]);
        setConversationId(null);
        return;
      }

      console.log(`Found ${messagesData?.length || 0} messages for user ${selectedUser.id}`);
      if (messagesData?.length > 0) {
        console.log('Latest message:', messagesData[0]);
        console.log('Oldest message:', messagesData[messagesData.length - 1]);
      }

      if (messagesData && messagesData.length > 0) {
        // Convert to chat messages format and reverse to get chronological order
        const existingMessages: ChatMessage[] = messagesData
          .reverse()
          .map((msg: any) => ({
            id: msg.id,
            role: normalizeRole(msg.role) === 'human' ? 'user' : 'assistant',
            content: msg.content,
            created_at: msg.created_at,
            sources: msg.metadata?.sources || [],
            is_interrupted: msg.metadata?.is_interrupted,
            hitl_phase: msg.metadata?.hitl_phase,
            error: msg.metadata?.error,
            error_type: msg.metadata?.error_type
          }));

        // Get conversation ID from first message
        const conversationIdFromMessages = messagesData[0]?.conversation_id || null;

        setMessages(existingMessages);
        setConversationId(conversationIdFromMessages);
      } else {
        // No messages found - start fresh
        console.log('No messages found for user:', selectedUser.id);
        setMessages([]);
        setConversationId(null);
      }
    } catch (error) {
      console.error('Error loading messages:', error);
      setMessages([]);
      setConversationId(null);
      setError(error instanceof Error ? error.message : 'Failed to load messages');
    } finally {
      setLoadingHistory(false);
    }
  }, [selectedUser, normalizeRole, supabase]);

  // Load messages when user changes - exactly like dualagentdebug
  useEffect(() => {
    if (selectedUser) {
      loadUserMessages();
    } else {
      setMessages([]);
      setConversationId(null);
    }
  }, [selectedUser, loadUserMessages])

  const fetchUsers = async () => {
    setLoadingUsers(true)
    try {
      // Use the backend API to fetch users - it handles the users/customers tables correctly
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_BASE_URL}/api/v1/memory-debug/users`)
      
      if (!response.ok) {
        console.error('Failed to fetch users:', response.statusText)
        return
      }

      const result = await response.json()
      
      if (result.success && result.data) {
        // Check which users have messages in the database
        const { data: messageUserIds } = await supabase!
          .from('messages')
          .select('user_id')
          .not('user_id', 'is', null);

        const usersWithMessages = new Set(messageUserIds?.map(m => m.user_id) || []);
        console.log('📊 Users with existing messages:', Array.from(usersWithMessages));

        // Transform to our User type and mark users with existing messages
        const transformedUsers = result.data.map((user: any) => ({
          id: user.id,
          display_name: user.name || user.email,
          email: user.email,
          user_type: user.role === 'employee' ? 'employee' : 'customer' as 'employee' | 'customer',
          is_active: true,
          created_at: user.created_at,
          hasMessages: usersWithMessages.has(user.id) // Add flag for users with existing messages
        }))

        console.log('📊 All users:', transformedUsers.map(u => `${u.display_name} (${u.id}) - Messages: ${u.hasMessages ? '✅' : '❌'}`));

        setUsers(transformedUsers)
      }
    } catch (error) {
      console.error('Error fetching users:', error)
    } finally {
      setLoadingUsers(false)
    }
  }

  const fetchConversations = async () => {
    if (!selectedUser || !supabase) return

    setLoadingConversations(true)
    try {
      const { data, error } = await supabase
        .from('conversations')
        .select('*')
        .eq('user_id', selectedUser.id)
        .order('updated_at', { ascending: false })
        .limit(20)

      if (error) {
        console.error('Error fetching conversations:', error)
        return
      }

      setConversations(data || [])
    } catch (error) {
      console.error('Error fetching conversations:', error)
    } finally {
      setLoadingConversations(false)
    }
  }


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Refresh messages function - exactly like dualagentdebug
  const refreshMessages = useCallback(async () => {
    if (selectedUser) {
      await loadUserMessages();
    }
  }, [selectedUser, loadUserMessages]);

  // Track if we're currently refreshing to prevent multiple simultaneous refreshes
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Real-time subscription for new messages - exactly like dualagentdebug
  useEffect(() => {
    if (!conversationId || !supabase) {
      return;
    }

    console.log('🔔 Setting up simple realtime subscription for new messages');
    
    const channel = supabase
      .channel('tobi-chat-messages')
      .on('postgres_changes', 
        { 
          event: 'INSERT',
          schema: 'public', 
          table: 'messages'
        }, 
        (payload) => {
          console.log('📨 New message detected:', payload);
          const newMessage = payload.new as any;
          
          // Check if this message belongs to current conversation
          const isCurrentConversation = conversationId && 
            newMessage?.conversation_id === conversationId;
          
          if (isCurrentConversation && !isRefreshing) {
            console.log('✅ Message belongs to current conversation - refreshing');
            
            // Prevent multiple simultaneous refreshes
            setIsRefreshing(true);
            
            // Small delay to ensure message is fully persisted, then refresh
            setTimeout(async () => {
              try {
                await loadUserMessages();
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
  }, [conversationId, loadUserMessages, isRefreshing])

  // Auto-resize textarea with optimization
  const resizeTextarea = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`
    }
  }, [])

  useEffect(() => {
    // Debounce textarea resize to improve performance
    const timeoutId = setTimeout(resizeTextarea, 0)
    return () => clearTimeout(timeoutId)
  }, [input, resizeTextarea])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  // HITL approval handlers
  const handleHitlApproval = async (approved: boolean) => {
    if (!hitlState.isActive) return
    
    const approvalMessage = approved ? 'approve' : 'deny'
    console.log(`🔍 [HITL] Sending ${approvalMessage} for phase: ${hitlState.phase}`)
    
    // Reset HITL state
    setHitlState({ isActive: false })
    
    // Send approval/denial as a regular message
    await sendMessage(approvalMessage)
  }

  const clearConversation = async () => {
    const user = selectedUser;
    
    if (!user) {
      console.warn('No user selected');
      return;
    }

    console.log('🔍 [CHAT] Clearing conversation for user:', user.display_name);

    // Clear local state immediately for better UX
    setMessages([]);
    setConversationId(null);
    setHitlState({ isActive: false });
    setLastSources([]);
    setSelectedConversation(null);
    setStoredMessages([]);
    setError(null);

    // Clear the user's conversation via API - exactly like dualagentdebug
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/chat/user/${user.id}/all-conversations`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Successfully cleared user conversation:', result.message);
      } else {
        const errorData = await response.json();
        console.error('Failed to clear user conversation:', errorData.detail || 'Unknown error');
      }
    } catch (apiError) {
      console.error('API error clearing user conversation:', apiError);
    }
  }

  // Login functions
  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    if (loginPassword === 'Admin123' && selectedUser) {
      setIsLoggedIn(true)
      setLoginError('')
    } else if (!selectedUser) {
      setLoginError('Please select a user first')
    } else {
      setLoginError('Invalid password')
    }
  }

  const handleLogout = () => {
    setIsLoggedIn(false)
    setSelectedUser(null)
    setLoginPassword('')
    setLoginError('')
    clearConversation()
  }

  // File download handler
  const handleFileDownload = (fileUrl: string, fileName?: string) => {
    const link = document.createElement('a')
    link.href = fileUrl
    link.download = fileName || fileUrl.split('/').pop() || 'download'
    link.target = '_blank'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Media detection and rendering
  const isImageUrl = (url: string) => {
    return /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(url)
  }

  const isVideoUrl = (url: string) => {
    return /\.(mp4|webm|ogg|mov|avi)$/i.test(url)
  }

  const isAudioUrl = (url: string) => {
    return /\.(mp3|wav|ogg|m4a|aac)$/i.test(url)
  }

  const extractMediaUrls = (text: string) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g
    const urls = text.match(urlRegex) || []
    return urls.filter(url => isImageUrl(url) || isVideoUrl(url) || isAudioUrl(url))
  }

  // Enhanced markdown renderer with media support
  const MarkdownRenderer = ({ content }: { content: string }) => {
    const mediaUrls = extractMediaUrls(content)
    
    return (
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight, rehypeRaw]}
          components={{
            // Custom code rendering
            code(props: any) {
              const { node, inline, className, children, ...rest } = props
              const match = /language-(\w+)/.exec(className || '')
              return !inline && match ? (
                <pre className="bg-gray-100 rounded p-3 overflow-x-auto">
                  <code className={className} {...rest}>
                    {children}
                  </code>
                </pre>
              ) : (
                <code className="bg-gray-100 px-1 py-0.5 rounded text-sm" {...rest}>
                  {children}
                </code>
              )
            },
            // Custom image rendering
            img(props: any) {
              const { src, alt, ...rest } = props
              return (
                <img
                  src={src}
                  alt={alt}
                  className="max-w-full h-auto rounded-lg shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => window.open(src, '_blank')}
                  {...rest}
                />
              )
            },
            // Custom link rendering
            a(props: any) {
              const { href, children, ...rest } = props
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-600 hover:text-primary-700 underline"
                  {...rest}
                >
                  {children}
                </a>
              )
            }
          }}
        >
          {content}
        </ReactMarkdown>
        
        {/* Render media attachments */}
        {mediaUrls.length > 0 && (
          <div className="mt-3 space-y-2">
            {mediaUrls.map((url, idx) => (
              <div key={idx} className="border rounded-lg p-2 bg-gray-50">
                {isImageUrl(url) && (
                  <div>
                    <img
                      src={url}
                      alt="Attached image"
                      className="max-w-full h-auto rounded cursor-pointer hover:opacity-90 transition-opacity"
                      onClick={() => window.open(url, '_blank')}
                    />
                  </div>
                )}
                {isVideoUrl(url) && (
                  <video
                    src={url}
                    controls
                    className="max-w-full h-auto rounded"
                    preload="metadata"
                  />
                )}
                {isAudioUrl(url) && (
                  <audio
                    src={url}
                    controls
                    className="w-full"
                    preload="metadata"
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  // Login screen - show when not logged in
  if (!isLoggedIn) {
    return (
      <div className="flex flex-col h-screen bg-gray-50">
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
            <div className="text-center mb-8">
              <div className="w-16 h-16 rounded-full overflow-hidden flex items-center justify-center mx-auto mb-4">
                <img src="/images/tobi-avatar.png" alt="Tobi" className="w-full h-full object-cover" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">Tobi Chat</h1>
              <p className="text-gray-600 mt-2">AI Sales Copilot Login</p>
            </div>

            <form onSubmit={handleLogin} className="space-y-6">
              {/* User Selection */}
              <div>
                <label className="block text-sm font-normal text-gray-700 mb-2">
                  Select User
                </label>
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowUserSelector(!showUserSelector)}
                    className="w-full p-3 text-left border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    {selectedUser ? selectedUser.display_name : 'Choose a user...'}
                    <svg className="w-5 h-5 absolute right-3 top-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {showUserSelector && (
                    <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto">
                      {loadingUsers ? (
                        <div className="px-3 py-2 text-sm text-gray-500">Loading users...</div>
                      ) : (
                        users.map((user) => (
                          <button
                            key={user.id}
                            type="button"
                            onClick={() => {
                              setSelectedUser(user)
                              setShowUserSelector(false)
                            }}
                            className={`w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none ${
                              selectedUser?.id === user.id ? 'bg-primary-100 border-l-4 border-primary-500' : ''
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-sm font-normal text-gray-900">{user.display_name}</div>
                                <div className="text-xs text-gray-500">{user.email}</div>
                              </div>
                              <span className={`px-2 py-1 text-xs font-normal rounded-full ${
                                user.user_type === 'employee' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                              }`}>
                                {user.user_type}
                              </span>
                            </div>
                          </button>
                        ))
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Password Input */}
              <div>
                <label className="block text-sm font-normal text-gray-700 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="Enter password..."
                  required
                />
              </div>

              {/* Error Message */}
              {loginError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-600">{loginError}</p>
                </div>
              )}

              {/* Login Button */}
              <button
                type="submit"
                className="w-full bg-primary-600 text-white py-3 px-4 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors font-normal"
              >
                Login to Chat
              </button>
            </form>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-full overflow-hidden flex items-center justify-center">
              <img src="/images/tobi-avatar.png" alt="Tobi" className="w-full h-full object-cover" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Tobi Chat</h1>
              <p className="text-sm text-gray-500">
                {selectedUser ? `Chatting with ${selectedUser.display_name}` : 'AI Sales Copilot'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* New Conversation */}
            <button
              onClick={clearConversation}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
              title="New conversation"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </button>

            {/* Logout Button */}
            <button
              onClick={handleLogout}
              className="p-2 text-gray-500 hover:text-red-700 hover:bg-red-50 rounded-full transition-colors"
              title="Logout"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center py-12 px-4">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
              <h2 className="text-lg font-bold text-gray-900 mb-2">Welcome to Tobi Chat</h2>
              <p className="text-gray-600 text-sm max-w-sm mx-auto">
                I'm your AI sales copilot. Ask me anything about sales, customers, or let me help you with your workflow.
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => {
          const chatMessage = message as ChatMessage
          const hasError = chatMessage.error || chatMessage.metadata?.type === 'error'
          
          // Format timestamp
          const formatTimestamp = (timestamp?: string) => {
            if (!timestamp) return ''
            try {
              const date = new Date(timestamp)
              return date.toLocaleString('en-US', {
                month: 'short',
                day: 'numeric', 
                hour: '2-digit',
                minute: '2-digit',
                hour12: true
              })
            } catch {
              return ''
            }
          }
          
          return (
            <div key={message.id} className="w-full bg-white">
              <div className="max-w-3xl mx-auto px-4 py-4">
                {message.role === 'user' ? (
                  // User message - blue box flush right with left margin on desktop
                  <div className="flex flex-col items-end ml-0 md:ml-20">
                    <div className="bg-primary-600 text-white rounded-2xl px-4 py-3 max-w-[70%]">
                      <div className="prose prose-sm prose-invert">
                        <MarkdownRenderer content={message.content} />
                      </div>
                    </div>
                    {message.created_at && (
                      <div className="text-xs text-gray-500 mt-1 px-2">
                        {formatTimestamp(message.created_at)}
                      </div>
                    )}
                  </div>
                ) : (
                  // AI message - no avatar, with right margin on desktop
                  <div className="mr-0 md:mr-20">
                    <div className={clsx(
                      'prose prose-sm max-w-none',
                      hasError && 'text-red-900'
                    )}>
                      <MarkdownRenderer content={message.content} />
                        
                        {/* File download button */}
                        {chatMessage.file && (
                          <div className="mt-3 pt-3 border-t border-gray-100">
                            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                              <div className="flex items-center space-x-3">
                                <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                                          d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                  </svg>
                                </div>
                                <div className="flex-1">
                                  <p className="text-sm font-normal text-blue-900">File Available</p>
                                  <p className="text-xs text-blue-700">Click to download attached file</p>
                                </div>
                                <button
                                  onClick={() => handleFileDownload(chatMessage.file!)}
                                  className="bg-blue-600 text-white text-sm py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors font-normal flex items-center space-x-2"
                                >
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                                          d="M12 10v6m0 0l-3-3m3 3l3-3M3 17V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                                    </svg>
                                    <span>Download</span>
                                  </button>
                              </div>
                            </div>
                          </div>
                        )}
                        
                        {/* Error indicator */}
                        {hasError && (
                          <div className="mt-2 pt-2 border-t border-red-200">
                            <div className="flex items-center space-x-2">
                              <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              <span className="text-xs text-red-600 font-normal">
                                Error: {chatMessage.error_type || chatMessage.metadata?.error_type || 'Unknown'}
                              </span>
                            </div>
                          </div>
                        )}
                        
                        {/* Sources for assistant messages */}
                        {chatMessage.sources && chatMessage.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-gray-100">
                            <p className="text-xs text-gray-500 mb-2 font-normal">📚 Sources ({chatMessage.sources.length}):</p>
                            <div className="space-y-1">
                              {chatMessage.sources.map((source, idx) => (
                                <div key={idx} className="text-xs bg-gray-50 rounded p-2">
                                  {source.url ? (
                                    <a 
                                      href={source.url} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                      className="text-primary-600 hover:text-primary-700 underline font-normal"
                                    >
                                      {source.title || source.url}
                                    </a>
                                  ) : (
                                    <span className="text-gray-700 font-normal">{source.title || 'Document'}</span>
                                  )}
                                  {source.relevance_score && (
                                    <span className="ml-2 text-gray-500 text-xs">
                                      ({Math.round(source.relevance_score * 100)}% relevant)
                                    </span>
                                  )}
                                  {source.content && (
                                    <p className="text-gray-600 mt-1 text-xs">
                                      {source.content.substring(0, 100)}...
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* HITL (Human-in-the-loop) indicator */}
                        {(chatMessage.is_interrupted || chatMessage.metadata?.type === 'hitl_interrupt') && (
                          <div className="mt-3 pt-3 border-t border-amber-100">
                            <div className="bg-amber-50 px-3 py-2 rounded-lg">
                              <div className="flex items-center space-x-2 mb-2">
                                <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></div>
                                <span className="text-xs text-amber-700 font-bold">Human approval required</span>
                              </div>
                              {chatMessage.hitl_phase && (
                                <p className="text-xs text-amber-600 mb-2">
                                  Phase: <span className="font-normal">{chatMessage.hitl_phase}</span>
                                </p>
                              )}
                              {hitlState.isActive && hitlState.conversationId === conversationId && conversationId && (
                                <div className="flex space-x-2 mt-3">
                                  <button
                                    onClick={() => handleHitlApproval(true)}
                                    className="flex-1 bg-green-600 text-white text-xs py-2 px-3 rounded-lg hover:bg-green-700 transition-colors font-normal"
                                  >
                                    ✓ Approve
                                  </button>
                                  <button
                                    onClick={() => handleHitlApproval(false)}
                                    className="flex-1 bg-red-600 text-white text-xs py-2 px-3 rounded-lg hover:bg-red-700 transition-colors font-normal"
                                  >
                                    ✗ Deny
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                    </div>
                    {message.created_at && (
                      <div className="text-xs text-gray-500 mt-1 px-0">
                        {formatTimestamp(message.created_at)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })}

        {/* Loading indicator */}
        {isLoading && (
          <div className="w-full bg-white">
            <div className="max-w-3xl mx-auto px-4 py-4">
              <div className="mr-0 md:mr-20">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-gray-500">Tobi is thinking...</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">
            Something went wrong. Please try again.
          </p>
        </div>
      )}

      {/* HITL Status Bar */}
      {hitlState.isActive && (
        <div className="bg-amber-50 border-t border-amber-200 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-amber-500 rounded-full animate-pulse"></div>
              <div>
                <p className="text-sm font-normal text-amber-800">Waiting for approval</p>
                <p className="text-xs text-amber-600">Phase: {hitlState.phase}</p>
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handleHitlApproval(true)}
                className="bg-green-600 text-white text-sm py-1 px-3 rounded-lg hover:bg-green-700 transition-colors"
              >
                Approve
              </button>
              <button
                onClick={() => handleHitlApproval(false)}
                className="bg-red-600 text-white text-sm py-1 px-3 rounded-lg hover:bg-red-700 transition-colors"
              >
                Deny
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={hitlState.isActive 
                ? "HITL mode active - use approval buttons above or type 'approve'/'deny'" 
                : loadingHistory
                ? "Loading conversation history..."
                : selectedUser
                ? "Type a message..."
                : "Please select a user first..."
              }
              disabled={isLoading || !selectedUser || loadingHistory}
              className={clsx(
                "w-full resize-none border rounded-2xl px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed",
                hitlState.isActive
                  ? "border-amber-300 focus:ring-amber-500 bg-amber-50"
                  : "border-gray-300 focus:ring-primary-500"
              )}
              rows={1}
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading || !selectedUser || loadingHistory}
            className={clsx(
              'flex-shrink-0 w-11 h-11 rounded-full flex items-center justify-center transition-all',
              input.trim() && !isLoading
                ? hitlState.isActive
                  ? 'bg-amber-600 text-white hover:bg-amber-700 shadow-md'
                  : 'bg-primary-600 text-white hover:bg-primary-700 shadow-md'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </form>
          
          {/* Status and ID display */}
          <div className="mt-2 text-xs text-gray-400 text-center flex items-center justify-center space-x-4">
            <span>ID: {conversationId ? conversationId.substring(0, 8) + '...' : 'New conversation'}</span>
            {hitlState.isActive && (
              <span className="text-amber-600 font-normal">● HITL Active</span>
            )}
            {lastSources.length > 0 && (
              <span className="text-blue-600">📚 {lastSources.length} sources</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
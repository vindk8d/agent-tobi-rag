'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { designTokens, figmaClasses } from '@/lib/design-tokens'

// Message types matching Figma design and existing API
interface ChatMessage {
  id: string
  type: 'human' | 'bot'
  content: string
  timestamp: Date
  sources?: Array<{
    title?: string
    url?: string
    content?: string
    relevance_score?: number
  }>
  is_interrupted?: boolean
  hitl_phase?: string
  error?: boolean
  error_type?: string
}

// Menu icon component (same as start page)
const MenuIcon = ({ size = "40" }: { size?: "20" | "24" | "32" | "40" | "48" | "16" }) => (
  <div className="relative size-full">
    <div className="absolute bottom-1/4 left-[12.5%] right-[12.5%] top-1/4">
      <div className="absolute inset-[-8.75%_-5.83%]">
        <svg className="block max-w-none size-full" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <line x1="3" y1="6" x2="21" y2="6"></line>
          <line x1="3" y1="12" x2="21" y2="12"></line>
          <line x1="3" y1="18" x2="21" y2="18"></line>
        </svg>
      </div>
    </div>
  </div>
)

// Individual message bubble component
const MessageBubble = ({ message }: { message: ChatMessage }) => {
  const isHuman = message.type === 'human'
  const hasError = message.error
  const hasSources = message.sources && message.sources.length > 0
  
  return (
    <div className={`flex flex-col gap-2.5 relative shrink-0 w-full ${
      isHuman ? 'items-end' : 'items-start'
    }`}>
      {isHuman ? (
        // Human message - right aligned, dark background, hugs content
        <div className="flex justify-end w-full p-[10px]">
          <div className="bg-[#6c6c6c] inline-flex items-start pl-[15px] pr-[19px] py-3 rounded-[18px] max-w-[85%]">
            <div className="font-medium leading-[1.2] text-[#ffffff] text-[14px] tracking-[-0.56px]">
              <p className="leading-[1.2] whitespace-pre-wrap break-words text-left">{message.content}</p>
            </div>
          </div>
        </div>
      ) : (
        // Bot message - left aligned, light background, hugs content
        <div className="flex justify-start w-full px-2.5 py-2">
          <div className={`inline-flex flex-col pl-[15px] pr-[19px] py-3 rounded-[18px] max-w-[85%] ${
            hasError ? 'bg-red-100 border border-red-200' : 'bg-[#e9e9e9]'
          }`}>
            <div className="font-medium leading-[1.2] text-[#343434] text-[14px] tracking-[-0.56px]">
              <p className="leading-[1.2] whitespace-pre-wrap break-words text-left">{message.content}</p>
              
              {/* Error indicator */}
              {hasError && (
                <div className="mt-2 pt-2 border-t border-red-300">
                  <div className="flex items-center space-x-1">
                    <svg className="w-3 h-3 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-xs text-red-600">
                      {message.error_type || 'Error'}
                    </span>
                  </div>
                </div>
              )}
              
              {/* Sources */}
              {hasSources && (
                <div className="mt-2 pt-2 border-t border-gray-300">
                  <p className="text-xs text-gray-500 mb-1">📚 Sources ({message.sources!.length}):</p>
                  <div className="space-y-1">
                    {message.sources!.slice(0, 3).map((source, idx) => (
                      <div key={idx} className="text-xs bg-gray-100 rounded p-1">
                        {source.url ? (
                          <a 
                            href={source.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-700 underline"
                          >
                            {source.title || source.url}
                          </a>
                        ) : (
                          <span className="text-gray-700">{source.title || 'Document'}</span>
                        )}
                        {source.relevance_score && (
                          <span className="ml-1 text-gray-500">
                            ({Math.round(source.relevance_score * 100)}%)
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Chat input component (matches Figma exactly)
const ChatInput = ({ onSendMessage, disabled }: { 
  onSendMessage: (message: string) => void
  disabled?: boolean 
}) => {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !disabled) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  // Auto-resize textarea
  const resizeTextarea = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const scrollHeight = textarea.scrollHeight
      const maxHeight = 120 // Max 5 lines approximately
      textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`
    }
  }, [])

  useEffect(() => {
    resizeTextarea()
  }, [input, resizeTextarea])

  return (
    <div className="box-border content-stretch flex flex-col gap-2.5 items-center justify-center min-w-[390px] overflow-clip px-[34px] py-4 relative shrink-0 w-full">
      <form onSubmit={handleSubmit} className="min-h-[48px] relative shrink-0 w-full">
        <div className="absolute bg-[#ffffff] inset-0 rounded-[36px]">
          <div aria-hidden="true" className="absolute border border-[#000000] border-solid inset-0 pointer-events-none rounded-[36px]" />
        </div>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="How can I help?"
          disabled={disabled}
          rows={1}
          className={`absolute font-medium leading-[1.2] left-[25px] right-[25px] top-[50%] translate-y-[-50%] text-[14px] tracking-[-0.56px] bg-transparent border-none outline-none resize-none overflow-hidden disabled:opacity-50 ${
            input.trim() ? 'text-[#000000]' : 'text-[#b3b3b3]'
          } placeholder:text-[#b3b3b3]`}
          style={{
            minHeight: '20px',
            maxHeight: '100px'
          }}
        />
      </form>
    </div>
  )
}

// Main chat interface component
export const ChatInterface = ({ 
  onBack,
  onMenuToggle,
  initialMessages = [],
  conversationId: propConversationId,
  userId = 'f26449e2-dce9-4b29-acd0-cb39a1f671fd' // John Smith - existing employee
}: { 
  onBack?: () => void
  onMenuToggle?: () => void
  initialMessages?: ChatMessage[]
  conversationId?: string | null
  userId?: string
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(propConversationId)
  const [error, setError] = useState<string | null>(null)
  const [isAwaitingHitl, setIsAwaitingHitl] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const initialMessageSentRef = useRef(false)
  const pendingRequestRef = useRef<string | null>(null) // Track pending requests
  const conversationLoadedRef = useRef(false)

  // Load existing conversation if conversationId is provided
  useEffect(() => {
    if (propConversationId && !conversationLoadedRef.current) {
      loadExistingConversation(propConversationId)
      conversationLoadedRef.current = true
    }
    
    // Reset when conversation changes
    if (propConversationId !== conversationId) {
      setConversationId(propConversationId)
      conversationLoadedRef.current = false
    }
  }, [propConversationId])

  // Handle initial message if provided
  useEffect(() => {
    if (initialMessages.length > 0 && !propConversationId) {
      // console.log('🔍 [CHAT-INTERFACE] Setting initial messages:', initialMessages)
      // Set initial messages first
      setMessages(initialMessages)
      
      const firstMessage = initialMessages[0]
      if (firstMessage.type === 'human' && !initialMessageSentRef.current) {
        // console.log('🔍 [CHAT-INTERFACE] Sending initial message to API:', firstMessage.content)
        initialMessageSentRef.current = true
        
        // Auto-send the initial human message to get bot response
        // Skip adding the human message since it's already in initialMessages
        setTimeout(() => {
          handleSendMessage(firstMessage.content, true)
        }, 500) // Small delay to ensure UI is ready
      }
    }
    
    // Cleanup function to reset the ref when component unmounts
    return () => {
      initialMessageSentRef.current = false
    }
  }, [initialMessages.length, propConversationId]) // Depend on length to handle prop changes

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messages.length > 0) {
      // Use a small delay to ensure DOM has updated
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    }
  }, [messages])

  // Load existing conversation messages
  const loadExistingConversation = async (conversationId: string) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/memory-debug/conversations/${conversationId}/messages`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      
      // Handle both direct response and wrapped APIResponse formats
      let messagesData: any[]
      if (result.data) {
        if (!result.success) {
          throw new Error(result.message || 'API request failed')
        }
        messagesData = result.data
      } else {
        messagesData = result
      }

      // Convert API messages to ChatMessage format
      const chatMessages: ChatMessage[] = messagesData.map((msg, index) => ({
        id: msg.id || `loaded_${index}`,
        type: msg.role === 'human' ? 'human' : 'bot',
        content: msg.content,
        timestamp: new Date(msg.created_at),
        sources: msg.metadata?.sources || []
      }))

      setMessages(chatMessages)
    } catch (err) {
      console.error('Error loading conversation:', err)
      setError(err instanceof Error ? err.message : 'Failed to load conversation')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendMessage = async (content: string, skipAddingMessage = false) => {
    if (!content.trim()) return

    // Prevent duplicate requests for the same message
    if (pendingRequestRef.current === content.trim()) {
      console.log('🔍 [CHAT-INTERFACE] Duplicate request detected, skipping:', content.trim())
      return
    }
    pendingRequestRef.current = content.trim()

    // Generate conversation ID if none exists
    const currentConversationId = conversationId || crypto.randomUUID()
    if (!conversationId) {
      setConversationId(currentConversationId)
    }

    // Add human message immediately (unless it's an initial message already in the list)
    if (!skipAddingMessage) {
      const humanMessage: ChatMessage = {
        id: `human_${Date.now()}`,
        type: 'human',
        content: content.trim(),
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, humanMessage])
    }
    
    setIsLoading(true)
    setError(null)

    try {
      const requestBody = {
        message: content.trim(),
        conversation_id: currentConversationId,
        user_id: userId,
        include_sources: true
      }

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
        setConversationId(chatResponse.conversation_id)
      }

      // Track HITL state for debugging (no UI changes)
      if (chatResponse.is_interrupted) {
        console.log('🔍 [MOBILE-CHAT] HITL interrupt detected - user can respond via chat')
        setIsAwaitingHitl(true)
      } else {
        setIsAwaitingHitl(false)
      }

      // Add bot response
      const botMessage: ChatMessage = {
        id: `bot_${Date.now()}`,
        type: 'bot',
        content: chatResponse.message,
        timestamp: new Date(),
        sources: chatResponse.sources || [],
        is_interrupted: chatResponse.is_interrupted,
        hitl_phase: chatResponse.hitl_phase,
        error: chatResponse.error,
        error_type: chatResponse.error_type
      }

      setMessages(prev => [...prev, botMessage])

    } catch (err) {
      console.error('Chat API Error:', err)
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
      
      // Add error message to chat
      const errorMessage: ChatMessage = {
        id: `error_${Date.now()}`,
        type: 'bot',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        error: true,
        error_type: 'api_error'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      pendingRequestRef.current = null // Clear pending request
    }
  }

  return (
    <div className="bg-[#ffffff] flex flex-col items-center justify-start px-8 py-10 relative w-full h-screen overflow-hidden">
      {/* Header - matches Figma exactly */}
      <div className="bg-[#ffffff] box-border content-stretch flex gap-2.5 items-center justify-end min-w-[390px] overflow-clip p-[4px] relative shrink-0 w-full">
        <div className="basis-0 flex flex-row grow items-center self-stretch shrink-0">
          <div className="basis-0 content-stretch flex gap-2.5 grow h-full items-center justify-start min-h-px min-w-px overflow-clip relative shrink-0">
            <button 
              onClick={onBack}
              className="flex flex-col font-semibold h-7 justify-center leading-tight not-italic relative shrink-0 text-[#000000] text-[32px] tracking-[-1.28px] w-[312px] cursor-pointer bg-transparent border-none p-0 hover:opacity-70 transition-opacity"
            >
              <p className="leading-tight text-left">Tobi</p>
            </button>
          </div>
        </div>
        <button 
          onClick={onMenuToggle}
          className="block cursor-pointer overflow-clip relative shrink-0 size-10 hover:opacity-70 transition-opacity"
        >
          <MenuIcon size="40" />
        </button>
      </div>

      {/* Chat Display Area - scrollable */}
      <div className="flex-1 min-h-0 w-full max-w-[390px]">
        <div className="h-full overflow-y-auto overflow-x-hidden px-4 py-2 scroll-smooth scrollbar-thin">
          <div className="min-h-full flex flex-col">
            {messages.length === 0 ? (
              // Empty state
              <div className="flex-1 flex items-center justify-center w-full">
                <div className="text-center py-12">
                  <p className="text-[#b3b3b3] text-[14px] font-medium">Start a conversation...</p>
                </div>
              </div>
            ) : (
              // Messages - add padding at top for first message visibility
              <>
                <div className="h-4 w-full shrink-0" /> {/* Top padding */}
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                
                {/* Loading indicator - positioned right after messages */}
                {isLoading && (
                  <div className="flex flex-col gap-2.5 relative shrink-0 w-full items-start">
                    <div className="flex justify-start w-full px-2.5 py-2">
                      <div className="bg-[#e9e9e9] inline-flex items-center pl-[15px] pr-[19px] py-3 rounded-[18px]">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-[#343434] rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-[#343434] rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-[#343434] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="h-4 w-full shrink-0" /> {/* Bottom padding */}
              </>
            )}
          </div>
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 p-3 bg-red-50 border border-red-200 rounded-lg shrink-0 w-full max-w-[390px]">
          <p className="text-sm text-red-600">
            {error}
          </p>
        </div>
      )}



      {/* Chat Input - matches Figma exactly */}
      <ChatInput 
        onSendMessage={handleSendMessage} 
        disabled={isLoading} 
      />
    </div>
  )
}

export default ChatInterface

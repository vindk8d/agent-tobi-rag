'use client'

import React, { useState, useEffect } from 'react'

// Conversation interface matching the API response
interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  latest_message: string | null
  latest_message_time: string
  latest_message_role: string | null
}

// Menu icon component (same as other components)
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

// Individual conversation item component
const ConversationItem = ({ 
  conversation, 
  onClick 
}: { 
  conversation: Conversation
  onClick: (conversationId: string) => void 
}) => {
  // Format the time to show relative time or date
  const formatTime = (timeString: string) => {
    const date = new Date(timeString)
    const now = new Date()
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (diffInHours < 168) { // 7 days
      return date.toLocaleDateString([], { weekday: 'short' })
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }

  // Truncate message preview
  const truncateMessage = (message: string | null, maxLength = 50) => {
    if (!message) return 'No messages yet'
    return message.length > maxLength ? message.substring(0, maxLength) + '...' : message
  }

  return (
    <button
      onClick={() => onClick(conversation.id)}
      className="bg-[#d9d9d9] box-border content-stretch flex flex-col gap-1 h-auto min-h-[48px] items-start justify-center overflow-clip pl-6 pr-4 py-3 relative shrink-0 w-full hover:bg-[#c9c9c9] transition-colors cursor-pointer"
    >
      <div className="flex justify-between items-start w-full">
        <div className="flex flex-col font-['Inter:Medium',_sans-serif] font-medium justify-center leading-tight not-italic relative shrink-0 text-[#000000] text-[14px] tracking-[-0.56px] flex-1 text-left">
          <p className="leading-tight font-semibold">{conversation.title}</p>
          <p className="leading-tight text-[12px] text-[#666666] mt-1">
            {truncateMessage(conversation.latest_message)}
          </p>
        </div>
        <div className="text-[10px] text-[#888888] ml-2 shrink-0">
          {formatTime(conversation.latest_message_time)}
        </div>
      </div>
    </button>
  )
}

// Main MenuWindow component
export const MenuWindow = ({
  isOpen,
  onClose,
  onConversationSelect,
  userId = 'f26449e2-dce9-4b29-acd0-cb39a1f671fd' // Default to John Smith
}: {
  isOpen: boolean
  onClose: () => void
  onConversationSelect: (conversationId: string) => void
  userId?: string
}) => {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch conversations when menu opens
  useEffect(() => {
    if (isOpen && userId) {
      fetchConversations()
    }
  }, [isOpen, userId])

  const fetchConversations = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/chat/users/${userId}/conversations`, {
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
      let conversationsData: Conversation[]
      if (result.data) {
        if (!result.success) {
          throw new Error(result.message || 'API request failed')
        }
        conversationsData = result.data
      } else {
        conversationsData = result
      }

      setConversations(conversationsData || [])
    } catch (err) {
      console.error('Error fetching conversations:', err)
      setError(err instanceof Error ? err.message : 'Failed to load conversations')
    } finally {
      setIsLoading(false)
    }
  }

  const handleConversationClick = (conversationId: string) => {
    onConversationSelect(conversationId)
    onClose() // Close menu after selection
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    // Only close if clicking the backdrop, not the menu itself
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <>
      {/* Backdrop overlay */}
      <div 
        className={`fixed inset-0 bg-black z-40 transition-all duration-500 ease-out ${
          isOpen ? 'bg-opacity-50 opacity-100' : 'bg-opacity-0 opacity-0 pointer-events-none'
        }`}
        onClick={handleBackdropClick}
      />
      
      {/* Menu window */}
      <div className={`fixed top-[5%] right-0 h-[90%] w-[90%] max-w-[400px] z-50 transform transition-transform duration-500 ease-out ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}>
        <div className="bg-[#d9d9d9] box-border content-stretch flex flex-col gap-2.5 items-start justify-start overflow-hidden px-[13px] py-0 relative rounded-l-[30px] size-full">
          {/* Menu header with close button */}
          <div className="flex justify-between items-center w-full pt-4 pb-2">
            <div className="overflow-clip relative shrink-0 size-10">
              <MenuIcon size="40" />
            </div>
            <button 
              onClick={onClose}
              className="text-[#000000] text-[24px] font-bold hover:opacity-70 transition-opacity"
            >
              ×
            </button>
          </div>
          
          {/* Conversation List */}
          <div className="box-border content-stretch flex flex-col gap-2.5 flex-1 items-start justify-start overflow-hidden p-[10px] relative shrink-0 w-full">
            {/* Header */}
            <div className="bg-[#d9d9d9] box-border content-stretch flex flex-col gap-2.5 h-12 items-start justify-center overflow-clip px-4 py-0 relative shrink-0 w-full">
              <div className="flex flex-col font-['Inter:Medium',_sans-serif] font-medium justify-center leading-[0] not-italic relative shrink-0 text-[#000000] text-[20px] text-nowrap tracking-[-0.8px]">
                <p className="leading-[0.95] whitespace-pre">Conversations</p>
              </div>
            </div>
            
            {/* Conversation items container - scrollable */}
            <div className="flex-1 w-full overflow-y-auto">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-[#666666] rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-[#666666] rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-[#666666] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center py-8 px-4">
                  <p className="text-[#666666] text-[14px] text-center mb-2">Failed to load conversations</p>
                  <button 
                    onClick={fetchConversations}
                    className="text-[#000000] text-[12px] underline hover:opacity-70"
                  >
                    Try again
                  </button>
                </div>
              ) : conversations.length === 0 ? (
                <div className="flex items-center justify-center py-8">
                  <p className="text-[#666666] text-[14px] text-center">No conversations yet</p>
                </div>
              ) : (
                <div className="flex flex-col gap-1">
                  {conversations.map((conversation) => (
                    <ConversationItem
                      key={conversation.id}
                      conversation={conversation}
                      onClick={handleConversationClick}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default MenuWindow

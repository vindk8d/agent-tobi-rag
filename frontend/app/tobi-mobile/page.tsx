'use client'

import React, { useState } from 'react'
import { ChatInterface } from './chat-interface'
import { MenuWindow } from './MenuWindow'

// SVG icon from Figma - you'll need to save this locally or use your icon system
const MenuIcon = ({ size = "48" }: { size?: "20" | "24" | "32" | "40" | "48" | "16" }) => {
  if (size === "40") {
    return (
      <div className="relative size-full" data-name="Size=40">
        <div className="absolute bottom-1/4 left-[12.5%] right-[12.5%] top-1/4">
          <div className="absolute inset-[-8.75%_-5.83%]">
            {/* Replace with your menu icon */}
            <svg className="block max-w-none size-full" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </div>
        </div>
      </div>
    )
  }
  return null
}

function SuggestedActionsRow({ onActionClick }: { onActionClick: (action: string) => void }) {
  const handleAction = (action: string) => {
    // Map actions to user messages that match the button text
    const actionMessages = {
      'generate-quote': 'Generate an Informal Quote',
      'update-opportunities': 'Update Opportunities', 
      'vehicle-info': 'Vehicle Information'
    }
    
    const message = actionMessages[action as keyof typeof actionMessages] || action
    onActionClick(message)
  }

  return (
    <div className="box-border content-center flex flex-wrap gap-5 items-center justify-center px-0 py-10 relative size-full">
      <button 
        onClick={() => handleAction('generate-quote')}
        className="bg-[#ffffff] box-border content-stretch cursor-pointer flex gap-2.5 items-center justify-center min-h-[52px] opacity-[0.81] overflow-visible px-[29px] py-3 relative rounded-lg shrink-0 w-60 hover:opacity-100 transition-opacity"
      >
        <div aria-hidden="true" className="absolute border border-[#000000] border-solid inset-0 pointer-events-none rounded-lg" />
        <div className="flex flex-col font-semibold justify-center leading-tight not-italic relative shrink-0 text-[#000000] text-[14px] text-center w-[182px]">
          <p className="leading-tight">Generate an Informal Quote</p>
        </div>
      </button>
      
      <button 
        onClick={() => handleAction('update-opportunities')}
        className="bg-[#ffffff] box-border content-stretch cursor-pointer flex gap-2.5 items-center justify-center min-h-[52px] opacity-[0.81] px-[29px] py-3 relative rounded-lg shrink-0 hover:opacity-100 transition-opacity"
      >
        <div aria-hidden="true" className="absolute border border-[#000000] border-solid inset-0 pointer-events-none rounded-lg" />
        <div className="flex flex-col font-semibold justify-center leading-tight not-italic relative shrink-0 text-[#000000] text-[14px] text-center w-[182px]">
          <p className="leading-tight">Update Opportunities</p>
        </div>
      </button>
      
      <button 
        onClick={() => handleAction('vehicle-info')}
        className="bg-[#ffffff] box-border content-stretch cursor-pointer flex gap-2.5 items-center justify-center min-h-[52px] opacity-[0.81] px-[29px] py-3 relative rounded-lg shrink-0 w-60 hover:opacity-100 transition-opacity"
      >
        <div aria-hidden="true" className="absolute border border-[#000000] border-solid inset-0 pointer-events-none rounded-lg" />
        <div className="flex flex-col font-semibold justify-center leading-tight not-italic relative shrink-0 text-[#000000] text-[14px] text-center w-[182px]">
          <p className="leading-tight">Vehicle Information</p>
        </div>
      </button>
    </div>
  )
}

function ChatEntryBox({ onSubmit }: { onSubmit: (message: string) => void }) {
  const [input, setInput] = React.useState('')
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) {
      onSubmit(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  return (
    <div className="box-border content-stretch flex flex-col gap-2.5 items-center justify-center px-[34px] py-4 relative size-full">
      <form onSubmit={handleSubmit} className="h-12 relative shrink-0 w-full">
        <div className="absolute bg-[#ffffff] inset-0 rounded-[36px]">
          <div aria-hidden="true" className="absolute border border-[#000000] border-solid inset-0 pointer-events-none rounded-[36px]" />
        </div>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="How can I help?"
          rows={1}
          className={`absolute font-medium left-[25px] right-[25px] top-[50%] translate-y-[-50%] text-[14px] tracking-[-0.56px] bg-transparent border-none outline-none resize-none overflow-y-auto ${
            input.trim() ? 'text-[#000000]' : 'text-[#b3b3b3]'
          } placeholder:text-[#b3b3b3]`}
          style={{
            height: '20px',
            maxHeight: '20px',
            lineHeight: '20px'
          }}
        />
      </form>
    </div>
  )
}

export default function TobiMobilePage() {
  const [showChat, setShowChat] = useState(false)
  const [initialMessage, setInitialMessage] = useState('')
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null)

  // Handle transition to chat interface
  const handleStartChat = (message?: string) => {
    if (message) {
      setInitialMessage(message)
    }
    setShowChat(true)
  }

  // Handle back to start page
  const handleBackToStart = () => {
    setShowChat(false)
    setInitialMessage('')
    setSelectedConversationId(null)
  }

  // Handle menu toggle
  const handleMenuToggle = () => {
    setIsMenuOpen(!isMenuOpen)
  }

  // Handle conversation selection from menu
  const handleConversationSelect = (conversationId: string) => {
    setSelectedConversationId(conversationId)
    setInitialMessage('') // Clear initial message when loading existing conversation
    setShowChat(true)
  }

  // Show chat interface if user has started chatting
  if (showChat) {
    return (
      <>
        <ChatInterface 
          onBack={handleBackToStart}
          onMenuToggle={handleMenuToggle}
          initialMessages={initialMessage ? [{
            id: 'initial',
            type: 'human' as const,
            content: initialMessage,
            timestamp: new Date()
          }] : []}
          conversationId={selectedConversationId}
        />
        <MenuWindow
          isOpen={isMenuOpen}
          onClose={() => setIsMenuOpen(false)}
          onConversationSelect={handleConversationSelect}
        />
      </>
    )
  }

  // Show start page
  return (
    <>
      <div className="bg-[#ffffff] box-border content-stretch flex flex-col gap-1 items-center justify-center px-8 py-10 relative size-full min-h-screen">
        {/* Menu Bar */}
        <div className="bg-[#ffffff] box-border content-stretch flex gap-2.5 items-center justify-end min-w-[390px] overflow-clip p-[4px] relative shrink-0 w-full">
          <button 
            onClick={handleMenuToggle}
            className="block cursor-pointer overflow-clip relative shrink-0 size-10 hover:opacity-70 transition-opacity"
          >
            <MenuIcon size="40" />
          </button>
        </div>
        
        {/* Welcome Message */}
        <div className="basis-0 box-border content-stretch flex gap-2.5 grow items-center justify-center min-h-px min-w-[391px] overflow-clip px-9 py-6 relative shrink-0 w-full">
          <div className="flex flex-col grow justify-center text-center">
            <p className="text-[48px] font-semibold text-black leading-[0.95] tracking-[-2.4px]">Hello, my name is Tobi, your sales assistant</p>
          </div>
        </div>
        
        {/* Suggested Actions */}
        <div className="basis-0 box-border content-center flex flex-wrap gap-5 grow items-center justify-center min-h-px min-w-[390px] px-0 py-10 relative shrink-0 w-full">
          <SuggestedActionsRow onActionClick={handleStartChat} />
        </div>
        
        {/* Chat Entry */}
        <div className="box-border content-stretch flex flex-col gap-2.5 items-center justify-center min-w-[440px] overflow-clip px-[28px] py-4 relative shrink-0 w-full">
          <ChatEntryBox onSubmit={handleStartChat} />
        </div>
      </div>
      
      {/* Menu Window - Always rendered for smooth animations */}
      <MenuWindow
        isOpen={isMenuOpen}
        onClose={() => setIsMenuOpen(false)}
        onConversationSelect={handleConversationSelect}
      />
    </>
  )
}

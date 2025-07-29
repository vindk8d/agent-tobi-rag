'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import EmployeeSelector from './components/EmployeeSelector';
import CustomerSidebar from './components/CustomerSidebar';
import GeneralInformation from './components/GeneralInformation';
import ConversationSummary from './components/ConversationSummary';
import ChatWindow from './components/ChatWindow';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types based on database schema
export interface Employee {
  id: string;
  name: string;
  position: string;
  email?: string;
  branch_id: string;
  is_active: boolean;
}

export interface Customer {
  id: string;
  name: string;
  phone?: string;
  mobile_number?: string;
  email?: string;
  company?: string;
  is_for_business: boolean;
  address?: string;
  notes?: string;
  created_at: string;
}

export interface Opportunity {
  id: string;
  customer_id: string;
  vehicle_id?: string;
  opportunity_salesperson_ae_id: string;
  stage: string;
  estimated_value?: number;
  probability?: number;
  expected_close_date?: string;
  created_at: string;
  customer: Customer;
}

export interface CustomerWarmth {
  id: string;
  customer_id: string;
  overall_warmth_score: number;
  warmth_level: 'ice_cold' | 'cold' | 'cool' | 'lukewarm' | 'warm' | 'hot' | 'scorching';
  purchase_probability: number;
  engagement_score: number;
  last_engagement_type?: string;
  last_engagement_date?: string;
  days_since_last_interaction: number;
  total_interactions: number;
  meaningful_interactions: number;
  predicted_purchase_value?: number;
  predicted_purchase_date?: string;
  confidence_level: number;
  warmth_notes?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CustomerWithWarmth extends Customer {
  customer_warmth?: CustomerWarmth;
  user_account?: {
    id: string;
    display_name: string;
    email: string;
    user_type: string;
  } | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
  metadata?: any;
}

export interface Conversation {
  id: string;
  user_id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export default function WatchtowerPage() {
  const [selectedEmployee, setSelectedEmployee] = useState<Employee | null>(null);
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-gray-900">Sales Watchtower</h1>
            <div className="text-sm text-gray-500">Monitor customer conversations</div>
          </div>
          
          {/* Employee Selector */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-600">System Online</span>
            </div>
            <EmployeeSelector 
              supabase={supabase}
              selectedEmployee={selectedEmployee}
              onEmployeeSelect={setSelectedEmployee}
            />
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Customer Sidebar - Fixed width */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
          <CustomerSidebar
            supabase={supabase}
            selectedEmployee={selectedEmployee}
            selectedCustomer={selectedCustomer}
            onCustomerSelect={setSelectedCustomer}
          />
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {selectedCustomer ? (
            <div className="flex-1 p-6 flex flex-col gap-6 overflow-hidden">
              {/* General Information - Equal height */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 flex-1 overflow-hidden">
                <GeneralInformation 
                  customer={selectedCustomer}
                  selectedEmployee={selectedEmployee}
                  supabase={supabase}
                />
              </div>

              {/* Conversation Summary and Chat Window - Equal height */}
              <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden min-h-0">
                {/* Conversation Summary */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 flex flex-col overflow-hidden">
                  <ConversationSummary
                    customer={selectedCustomer}
                    supabase={supabase}
                  />
                </div>

                {/* Chat Window */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 flex flex-col overflow-hidden">
                  <ChatWindow
                    customer={selectedCustomer}
                    supabase={supabase}
                  />
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="text-gray-400 mb-4">
                  <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Customer Selected</h3>
                <p className="text-gray-500 max-w-sm">
                  {selectedEmployee 
                    ? "Select a customer from the sidebar to view their information and conversations"
                    : "Please select an employee first to view their assigned customers"
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 
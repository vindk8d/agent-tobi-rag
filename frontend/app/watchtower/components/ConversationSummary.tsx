'use client';

import { useState, useEffect } from 'react';
import { SupabaseClient } from '@supabase/supabase-js';
import { CustomerWithWarmth } from '../page';

interface ConversationSummaryProps {
  customer: CustomerWithWarmth;
  supabase: SupabaseClient;
}

interface ConversationSummaryData {
  id: string;
  conversation_id: string;
  summary_text: string;
  summary_type: string;
  message_count: number;
  created_at: string;
  consolidation_status: string;
}

export default function ConversationSummary({ customer, supabase }: ConversationSummaryProps) {
  const [summaries, setSummaries] = useState<ConversationSummaryData[]>([]);
  const [latestSummary, setLatestSummary] = useState<ConversationSummaryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(true); // Default to live mode
  const [hotReloadStatus, setHotReloadStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    if (customer) {
      fetchConversationSummaries();
    }
  }, [customer]);

  // Hot reload setup for conversation summaries
  useEffect(() => {
    if (!customer.user_account || !isLive) {
      setHotReloadStatus('disconnected');
      return;
    }

    setHotReloadStatus('connecting');
    
    const summariesSubscription = supabase
      .channel(`customer_summaries_${customer.user_account.id}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'conversation_summaries',
          filter: `user_id=eq.${customer.user_account.id}`,
        },
        (payload) => {
          console.log('ConversationSummary: Hot reload triggered - refreshing summaries');
          // Refresh summaries when any change occurs
          fetchConversationSummaries();
        }
      )
      .subscribe((status, err) => {
        if (status === 'SUBSCRIBED') {
          setHotReloadStatus('connected');
          console.log('✅ ConversationSummary: Hot reload connected');
        } else if (status === 'CHANNEL_ERROR') {
          setHotReloadStatus('disconnected');
          console.error('❌ ConversationSummary: Hot reload error:', err);
        } else {
          setHotReloadStatus('connecting');
        }
      });

    return () => {
      summariesSubscription.unsubscribe();
      setHotReloadStatus('disconnected');
    };
  }, [customer.user_account, isLive, supabase]);

  const fetchConversationSummaries = async () => {
    setLoading(true);
    try {
      // Skip if customer doesn't have a user account
      if (!customer.user_account) {
        console.log('ConversationSummary: No user account for customer:', customer.name);
        setSummaries([]);
        setLatestSummary(null);
        setLoading(false);
        return;
      }

      // Fetch conversation summaries for this customer's user account, ordered by most recent
      const { data, error } = await supabase
        .from('conversation_summaries')
        .select(`
          id,
          conversation_id,
          summary_text,
          summary_type,
          message_count,
          created_at,
          consolidation_status
        `)
        .eq('user_id', customer.user_account.id)
        .eq('consolidation_status', 'active')
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) {
        console.error('ConversationSummary: Error fetching conversation summaries:', error);
        return;
      }
      
      const summariesData = data || [];
      setSummaries(summariesData);
      setLatestSummary(summariesData.length > 0 ? summariesData[0] : null);
      setLastUpdated(new Date());

      console.log(`ConversationSummary: Loaded ${summariesData.length} summaries for ${customer.name}`);

    } catch (error) {
      console.error('ConversationSummary: Error fetching conversation summaries:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Never';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = diffMs / (1000 * 60 * 60);
    const diffDays = diffMs / (1000 * 60 * 60 * 24);

    if (diffHours < 1) {
      const diffMinutes = Math.floor(diffMs / (1000 * 60));
      return `${diffMinutes} minutes ago`;
    } else if (diffHours < 24) {
      return `${Math.floor(diffHours)} hours ago`;
    } else if (diffDays < 7) {
      return `${Math.floor(diffDays)} days ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-gray-200 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Latest Conversation Summary</h2>
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

      <div className="flex-1 overflow-y-auto p-4">
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
            <p className="text-sm text-gray-500">Loading conversation summaries...</p>
          </div>
        ) : !latestSummary ? (
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
                <p className="text-sm">No conversation summaries available</p>
                <p className="text-xs text-gray-400 mt-1">Summaries are automatically generated as conversations progress</p>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {/* Latest Summary */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-end mb-3">
                <div className="text-sm text-gray-500">
                  {formatDate(latestSummary.created_at)}
                </div>
              </div>
              
              <div className="prose prose-sm max-w-none">
                <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                  {latestSummary.summary_text}
                </p>
              </div>
              
              <div className="mt-3 pt-3 border-t border-gray-200">
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Messages: {latestSummary.message_count}</span>
                  <span>ID: {latestSummary.conversation_id.slice(0, 8)}...</span>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{summaries.length}</div>
                <div className="text-sm text-blue-600">Total Summaries</div>
              </div>
              <div className="bg-green-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {summaries.reduce((total, summary) => total + summary.message_count, 0)}
                </div>
                <div className="text-sm text-green-600">Total Messages</div>
              </div>
            </div>

            {/* Recent Summaries List */}
            {summaries.length > 1 && (
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-3">Recent Summaries</h3>
                <div className="space-y-2">
                  {summaries.slice(1, 4).map((summary) => (
                    <div key={summary.id} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex items-center justify-end mb-2">
                        <span className="text-xs text-gray-500">
                          {formatDate(summary.created_at)}
                        </span>
                      </div>
                      <div className="text-sm text-gray-700 line-clamp-2">
                        {summary.summary_text.substring(0, 100)}...
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {summary.message_count} messages
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
} 
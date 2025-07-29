'use client';

import React, { useState, useEffect } from 'react';
import { ConfirmationRequest, ConfirmationResponse, ConfirmationStatus } from '@/types';

interface ConfirmationPanelProps {
  conversationId: string;
  onConfirmationResponse?: (response: ConfirmationResponse) => void;
  pollingIntervalMs?: number;
}

export default function ConfirmationPanel({ 
  conversationId, 
  onConfirmationResponse,
  pollingIntervalMs = 2000 
}: ConfirmationPanelProps) {
  const [pendingConfirmations, setPendingConfirmations] = useState<ConfirmationRequest[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll for pending confirmations
  useEffect(() => {
    const fetchPendingConfirmations = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/chat/confirmation/pending/${conversationId}`);
        const result = await response.json();
        
        if (result.success) {
          setPendingConfirmations(result.data || []);
          setError(null);
        } else {
          setError(result.error || 'Failed to fetch confirmations');
        }
      } catch (err) {
        setError('Error fetching confirmations');
      }
    };

    fetchPendingConfirmations();
    
    const interval = setInterval(fetchPendingConfirmations, pollingIntervalMs);
    return () => clearInterval(interval);
  }, [conversationId, pollingIntervalMs]);

  const handleConfirmationResponse = async (
    confirmationId: string, 
    action: ConfirmationStatus, 
    modifiedMessage?: string, 
    notes?: string
  ) => {
    setLoading(true);
    try {
      // Convert ConfirmationStatus to backend-expected action values
      let backendAction: string;
      switch (action) {
        case ConfirmationStatus.APPROVED:
          backendAction = "approve";
          break;
        case ConfirmationStatus.CANCELLED:
          backendAction = "deny";
          break;
        case ConfirmationStatus.MODIFIED:
          backendAction = "approve"; // Modified messages are approved with changes
          break;
        default:
          backendAction = "deny";
      }

      const response = {
        action: backendAction,
        modified_message: modifiedMessage,
        responded_at: new Date().toISOString(),
        notes
      };

      const result = await fetch(`http://localhost:8000/api/v1/chat/confirmation/${confirmationId}/respond`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(response),
      });

      const responseData = await result.json();
      
      if (responseData.status === "success") {
        // Remove the confirmation from pending list
        setPendingConfirmations(prev => 
          prev.filter(req => req.confirmation_id !== confirmationId)
        );
        
        // Notify parent component
        if (onConfirmationResponse) {
          onConfirmationResponse({
            confirmation_id: confirmationId,
            action: action,
            modified_message: modifiedMessage,
            responded_at: new Date().toISOString(),
            notes
          });
        }
        
        setError(null);
      } else {
        setError(responseData.detail || 'Failed to process confirmation');
      }
    } catch (err) {
      setError('Error processing confirmation');
    } finally {
      setLoading(false);
    }
  };

  if (pendingConfirmations.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {pendingConfirmations.map((confirmation) => (
        <ConfirmationCard
          key={confirmation.confirmation_id}
          confirmation={confirmation}
          onResponse={handleConfirmationResponse}
          loading={loading}
        />
      ))}
      
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <span className="text-red-400 font-bold">!</span>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Confirmation Error
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{error}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface ConfirmationCardProps {
  confirmation: ConfirmationRequest;
  onResponse: (
    confirmationId: string,
    action: ConfirmationStatus,
    modifiedMessage?: string,
    notes?: string
  ) => void;
  loading: boolean;
}

function ConfirmationCard({ confirmation, onResponse, loading }: ConfirmationCardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [modifiedMessage, setModifiedMessage] = useState(confirmation.message_content);
  const [notes, setNotes] = useState('');

  const timeRemaining = Math.max(0, new Date(confirmation.expires_at).getTime() - new Date().getTime());
  const minutesRemaining = Math.floor(timeRemaining / (1000 * 60));
  const secondsRemaining = Math.floor((timeRemaining % (1000 * 60)) / 1000);

  const handleApprove = () => {
    onResponse(confirmation.confirmation_id, ConfirmationStatus.APPROVED, undefined, notes);
  };

  const handleCancel = () => {
    onResponse(confirmation.confirmation_id, ConfirmationStatus.CANCELLED, undefined, notes);
  };

  const handleModify = () => {
    if (isEditing) {
      onResponse(confirmation.confirmation_id, ConfirmationStatus.MODIFIED, modifiedMessage, notes);
    } else {
      setIsEditing(true);
    }
  };

  const isExpired = timeRemaining <= 0;

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 shadow-sm">
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <span className="text-blue-600 font-semibold">📬</span>
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-blue-900">
              Customer Message Confirmation
            </h3>
            <p className="text-sm text-blue-700">
              Requested by {confirmation.requested_by}
            </p>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-sm text-blue-700">
            {isExpired ? (
              <span className="text-red-600 font-medium">Expired</span>
            ) : (
              <>Time remaining: {minutesRemaining}m {secondsRemaining}s</>
            )}
          </div>
        </div>
      </div>

      {/* Customer Information */}
      <div className="mt-4 bg-white rounded-lg p-4 border border-blue-100">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Customer Details</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600">Name</p>
            <p className="font-medium">{confirmation.customer_name}</p>
          </div>
          {confirmation.customer_email && (
            <div>
              <p className="text-sm text-gray-600">Email</p>
              <p className="font-medium">{confirmation.customer_email}</p>
            </div>
          )}
          <div>
            <p className="text-sm text-gray-600">Message Type</p>
            <p className="font-medium capitalize">{confirmation.message_type.replace('_', ' ')}</p>
          </div>
        </div>
      </div>

      {/* Message Content */}
      <div className="mt-4 bg-white rounded-lg p-4 border border-blue-100">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Message to Send</h4>
        {isEditing ? (
          <textarea
            value={modifiedMessage}
            onChange={(e) => setModifiedMessage(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows={4}
            maxLength={2000}
          />
        ) : (
          <p className="text-gray-800 whitespace-pre-wrap">{confirmation.message_content}</p>
        )}
      </div>

      {/* Notes */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Notes (optional)
        </label>
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Add any notes about this decision..."
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      {/* Action Buttons */}
      <div className="mt-6 flex flex-wrap gap-3">
        <button
          onClick={handleApprove}
          disabled={loading || isExpired}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Processing...' : 'Approve & Send'}
        </button>
        
        <button
          onClick={handleModify}
          disabled={loading || isExpired}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isEditing ? 'Save & Send Modified' : 'Modify Message'}
        </button>
        
        <button
          onClick={handleCancel}
          disabled={loading || isExpired}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Cancel
        </button>
        
        {isEditing && (
          <button
            onClick={() => {
              setIsEditing(false);
              setModifiedMessage(confirmation.message_content);
            }}
            disabled={loading}
            className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel Edit
          </button>
        )}
      </div>
    </div>
  );
} 
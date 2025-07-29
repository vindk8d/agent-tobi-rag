'use client';

import { useState, useEffect } from 'react';
import { SupabaseClient } from '@supabase/supabase-js';
import { Employee, Customer, CustomerWithWarmth } from '../page';

interface CustomerSidebarProps {
  supabase: SupabaseClient;
  selectedEmployee: Employee | null;
  selectedCustomer: Customer | null;
  onCustomerSelect: (customer: Customer | null) => void;
}

export default function CustomerSidebar({
  supabase,
  selectedEmployee,
  selectedCustomer,
  onCustomerSelect,
}: CustomerSidebarProps) {
  const [customers, setCustomers] = useState<CustomerWithWarmth[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    if (selectedEmployee) {
      fetchCustomersForEmployee();
    } else {
      setCustomers([]);
      onCustomerSelect(null);
    }
  }, [selectedEmployee]);

  const fetchCustomersForEmployee = async () => {
    if (!selectedEmployee) return;
    
    setLoading(true);
    try {
      // First, get unique customer IDs for this employee
      const customerIdsResult = await supabase
        .from('opportunities')
        .select('customer_id')
        .eq('opportunity_salesperson_ae_id', selectedEmployee.id);

      if (customerIdsResult.error) {
        console.error('Error fetching customer IDs:', customerIdsResult.error);
        return;
      }

      const uniqueCustomerIds = Array.from(new Set(customerIdsResult.data?.map(opp => opp.customer_id) || []));
      
      if (uniqueCustomerIds.length === 0) {
        setCustomers([]);
        return;
      }

      // Then get customers with their warmth data and user information
      const { data, error } = await supabase
        .from('customers')
        .select(`
          id,
          name,
          phone,
          mobile_number,
          email,
          company,
          is_for_business,
          address,
          notes,
          created_at,
          customer_warmth(
            id,
            customer_id,
            overall_warmth_score,
            warmth_level,
            purchase_probability,
            engagement_score,
            last_engagement_type,
            last_engagement_date,
            days_since_last_interaction,
            total_interactions,
            meaningful_interactions,
            predicted_purchase_value,
            predicted_purchase_date,
            confidence_level,
            warmth_notes,
            is_active,
            created_at,
            updated_at
          )
        `)
        .in('id', uniqueCustomerIds);

      if (error) {
        console.error('Error fetching customers:', error);
        return;
      }

      // Get user information separately for better reliability
      const { data: userData, error: userError } = await supabase
        .from('users')
        .select(`
          id,
          display_name,
          email,
          user_type,
          customer_id
        `)
        .eq('user_type', 'customer')
        .in('customer_id', uniqueCustomerIds);

      if (userError) {
        console.error('Error fetching user data:', userError);
      }

      // Create a map of customer_id to user data for easy lookup
      const userMap = new Map();
      if (userData) {
        userData.forEach(user => {
          userMap.set(user.customer_id, user);
        });
      }

      // Format the data and filter for customers with active warmth data
      const formattedData = (data || [])
        .map((customer: any) => ({
          ...customer,
          customer_warmth: customer.customer_warmth?.find((cw: any) => cw.is_active === true) || null,
          user_account: userMap.get(customer.id) || null
        }))
        .filter(customer => customer.customer_warmth !== null) // Only show customers with warmth data
        .sort((a, b) => (b.customer_warmth?.overall_warmth_score || 0) - (a.customer_warmth?.overall_warmth_score || 0));

      setCustomers(formattedData);
    } catch (error) {
      console.error('Error fetching customers:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredCustomers = customers.filter(customer => {
    if (!searchTerm) return true;
    return (
      customer.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      customer.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      customer.company?.toLowerCase().includes(searchTerm.toLowerCase())
    );
  });

  const getWarmthColor = (warmthLevel: string) => {
    switch (warmthLevel) {
      case 'scorching':
        return 'bg-red-200 text-red-900 border border-red-300';
      case 'hot':
        return 'bg-red-100 text-red-800';
      case 'warm':
        return 'bg-orange-100 text-orange-800';
      case 'lukewarm':
        return 'bg-yellow-100 text-yellow-800';
      case 'cool':
        return 'bg-blue-100 text-blue-800';
      case 'cold':
        return 'bg-blue-200 text-blue-900';
      case 'ice_cold':
        return 'bg-gray-200 text-gray-900';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };



  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          {selectedEmployee ? `${selectedEmployee.name}'s Customers` : 'Select Employee'}
        </h2>
        
        {selectedEmployee && (
          <div className="relative">
            <input
              type="text"
              placeholder="Search customers..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <svg className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        )}
      </div>

      {/* Customer List */}
      <div className="flex-1 overflow-y-auto">
        {!selectedEmployee ? (
          <div className="p-4 text-center text-gray-500">
            <div className="text-gray-400 mb-2">
              <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
              </svg>
            </div>
            <p className="text-sm">Please select an employee to view their customers</p>
          </div>
        ) : loading ? (
          <div className="p-4 text-center text-gray-500">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
            <p className="text-sm">Loading customers...</p>
          </div>
        ) : filteredCustomers.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <div className="text-gray-400 mb-2">
              <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <p className="text-sm">
              {searchTerm 
                ? `No customers found matching "${searchTerm}"`
                : 'No customers assigned to this employee'
              }
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {filteredCustomers.map((customer) => (
              <button
                key={customer.id}
                onClick={() => onCustomerSelect(customer)}
                className={`w-full p-4 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none transition-colors duration-200 ${
                  selectedCustomer?.id === customer.id ? 'bg-primary-50 border-r-2 border-primary-500' : ''
                }`}
              >
                <div className="space-y-3">
                  {/* Customer Name and Warmth Level */}
                  <div className="flex items-center justify-between">
                    <h3 className="font-medium text-gray-900 truncate">{customer.name}</h3>
                    {customer.customer_warmth && (
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getWarmthColor(customer.customer_warmth.warmth_level)}`}>
                        {customer.customer_warmth.warmth_level.replace('_', ' ')}
                      </span>
                    )}
                  </div>

                  {/* Company */}
                  {customer.company && (
                    <p className="text-sm text-gray-600 truncate">{customer.company}</p>
                  )}

                  {/* User Account Status */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">User Account</span>
                    <span className={`font-medium ${customer.user_account ? 'text-green-600' : 'text-red-600'}`}>
                      {customer.user_account ? `ID: ${customer.user_account.id.substring(0, 8)}...` : 'No Account'}
                    </span>
                  </div>

                  {/* Warmth Score */}
                  {customer.customer_warmth && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">Warmth Score</span>
                      <span className="font-medium text-gray-900">{Math.round(customer.customer_warmth.overall_warmth_score)}/100</span>
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {selectedEmployee && filteredCustomers.length > 0 && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="text-sm text-gray-600">
            <span className="font-medium">{filteredCustomers.length}</span> customers
            {searchTerm && (
              <span> matching "{searchTerm}"</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
} 
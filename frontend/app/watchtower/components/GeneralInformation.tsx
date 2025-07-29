'use client';

import { useState, useEffect } from 'react';
import { SupabaseClient } from '@supabase/supabase-js';
import { Employee, Customer, Opportunity, CustomerWarmth } from '../page';

interface GeneralInformationProps {
  customer: Customer;
  selectedEmployee: Employee | null;
  supabase: SupabaseClient;
}

interface Vehicle {
  id: string;
  brand: string;
  model: string;
  year: number;
  type: string;
  color?: string;
  power?: number;
  fuel_type?: string;
}

interface OpportunityWithVehicle extends Opportunity {
  vehicle?: Vehicle;
  referral_name?: string;
  notes?: string;
}

export default function GeneralInformation({ 
  customer, 
  selectedEmployee, 
  supabase 
}: GeneralInformationProps) {
  const [opportunities, setOpportunities] = useState<OpportunityWithVehicle[]>([]);
  const [customerWarmth, setCustomerWarmth] = useState<CustomerWarmth | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (customer && selectedEmployee) {
      fetchOpportunityDetails();
      fetchCustomerWarmth();
    }
  }, [customer, selectedEmployee]);

  const fetchOpportunityDetails = async () => {
    if (!customer || !selectedEmployee) return;

    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('opportunities')
        .select(`
          id,
          customer_id,
          vehicle_id,
          opportunity_salesperson_ae_id,
          stage,
          estimated_value,
          probability,
          expected_close_date,
          referral_name,
          notes,
          created_at,
          vehicle:vehicles(
            id,
            brand,
            model,
            year,
            type,
            color,
            power,
            fuel_type
          )
        `)
        .eq('customer_id', customer.id)
        .eq('opportunity_salesperson_ae_id', selectedEmployee.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching opportunity details:', error);
        return;
      }

             const formattedData = (data || []).map((opp: any) => ({
         ...opp,
         customer: customer,
         vehicle: Array.isArray(opp.vehicle) ? opp.vehicle[0] : opp.vehicle
       }));
       setOpportunities(formattedData);
    } catch (error) {
      console.error('Error fetching opportunity details:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchCustomerWarmth = async () => {
    if (!customer) return;

    try {
      const { data, error } = await supabase
        .from('customer_warmth')
        .select('*')
        .eq('customer_id', customer.id)
        .eq('is_active', true)
        .single();

      if (error) {
        console.error('Error fetching customer warmth:', error);
        return;
      }

      setCustomerWarmth(data);
    } catch (error) {
      console.error('Error fetching customer warmth:', error);
    }
  };

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

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'New':
        return 'bg-blue-100 text-blue-800';
      case 'Contacted':
        return 'bg-yellow-100 text-yellow-800';
      case 'Consideration':
        return 'bg-orange-100 text-orange-800';
      case 'Purchase Intent':
        return 'bg-purple-100 text-purple-800';
      case 'Won':
        return 'bg-green-100 text-green-800';
      case 'Lost':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };



  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-6 pb-4 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">General Information</h2>
        {loading && (
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
        )}
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 pt-4 overflow-hidden">
        {/* Customer Information */}
        <div className="flex flex-col overflow-hidden">
          <h3 className="text-lg font-medium text-gray-900 pb-2 border-b border-gray-200 mb-4">
            Customer Details
          </h3>
          <div className="flex-1 overflow-y-auto space-y-4 pr-2">
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-600">Name</label>
              <p className="text-sm text-gray-900 mt-1">{customer.name}</p>
            </div>
            
            {customer.company && (
              <div>
                <label className="text-sm font-medium text-gray-600">Company</label>
                <p className="text-sm text-gray-900 mt-1">{customer.company}</p>
              </div>
            )}
            
            {customer.email && (
              <div>
                <label className="text-sm font-medium text-gray-600">Email</label>
                <p className="text-sm text-gray-900 mt-1">{customer.email}</p>
              </div>
            )}
            
            {customer.phone && (
              <div>
                <label className="text-sm font-medium text-gray-600">Phone</label>
                <p className="text-sm text-gray-900 mt-1">{customer.phone}</p>
              </div>
            )}
            
            {customer.mobile_number && (
              <div>
                <label className="text-sm font-medium text-gray-600">Mobile</label>
                <p className="text-sm text-gray-900 mt-1">{customer.mobile_number}</p>
              </div>
            )}
            
            {customer.address && (
              <div>
                <label className="text-sm font-medium text-gray-600">Address</label>
                <p className="text-sm text-gray-900 mt-1">{customer.address}</p>
              </div>
            )}
          </div>

          {/* Last Engagement Information */}
          {customerWarmth && customerWarmth.last_engagement_type && (
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <h4 className="text-sm font-medium text-gray-900">Last Engagement</h4>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-600">Engagement Type</label>
                  <p className="text-sm text-gray-900 mt-1">
                    {customerWarmth.last_engagement_type.replace('_', ' ')}
                  </p>
                </div>
                
                <div>
                  <label className="text-sm font-medium text-gray-600">Days Since Contact</label>
                  <p className="text-sm text-gray-900 mt-1">
                    {customerWarmth.days_since_last_interaction} days ago
                  </p>
                </div>
              </div>
            </div>
          )}

          </div>
        </div>

        {/* Opportunity Information */}
        <div className="flex flex-col overflow-hidden">
          <h3 className="text-lg font-medium text-gray-900 pb-2 border-b border-gray-200 mb-4">
            Active Opportunities
          </h3>
          <div className="flex-1 overflow-y-auto pr-2">
          
          {opportunities.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <svg className="w-12 h-12 mx-auto text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-sm">No active opportunities</p>
            </div>
          ) : (
            <div className="space-y-4">
              {opportunities.map((opportunity) => (
                <div key={opportunity.id} className="border border-gray-200 rounded-lg p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStageColor(opportunity.stage)}`}>
                      {opportunity.stage}
                    </span>
                    <div className="text-xs text-gray-500">
                      {formatDate(opportunity.created_at)}
                    </div>
                  </div>

                  {opportunity.vehicle && (
                    <div>
                      <p className="text-sm text-gray-900 mt-1">
                        {opportunity.vehicle.year} {opportunity.vehicle.brand} {opportunity.vehicle.model}
                        {opportunity.vehicle.color && ` - ${opportunity.vehicle.color}`}
                      </p>
                      <p className="text-xs text-gray-500">
                        {opportunity.vehicle.type} • {opportunity.vehicle.fuel_type}
                        {opportunity.vehicle.power && ` • ${opportunity.vehicle.power} HP`}
                      </p>
                    </div>
                  )}

                  {opportunity.referral_name && (
                    <div>
                      <label className="text-sm font-medium text-gray-600">Referral</label>
                      <p className="text-sm text-gray-900 mt-1">{opportunity.referral_name}</p>
                    </div>
                  )}

                </div>
              ))}
            </div>
          )}
          </div>
        </div>
      </div>
    </div>
  );
} 
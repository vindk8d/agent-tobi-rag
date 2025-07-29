'use client';

import { useState, useEffect } from 'react';
import { SupabaseClient } from '@supabase/supabase-js';
import { Employee } from '../page';

interface EmployeeSelectorProps {
  supabase: SupabaseClient;
  selectedEmployee: Employee | null;
  onEmployeeSelect: (employee: Employee | null) => void;
}

export default function EmployeeSelector({ 
  supabase, 
  selectedEmployee, 
  onEmployeeSelect 
}: EmployeeSelectorProps) {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    fetchEmployees();
  }, []);

  const fetchEmployees = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('employees')
        .select('*')
        .eq('is_active', true)
        .order('name');

      if (error) {
        console.error('Error fetching employees:', error);
        return;
      }

      setEmployees(data || []);
    } catch (error) {
      console.error('Error fetching employees:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleEmployeeSelect = (employee: Employee) => {
    onEmployeeSelect(employee);
    setIsOpen(false);
  };

  const getPositionBadgeColor = (position: string) => {
    switch (position) {
      case 'director':
        return 'bg-purple-100 text-purple-800';
      case 'manager':
        return 'bg-blue-100 text-blue-800';
      case 'account_executive':
        return 'bg-green-100 text-green-800';
      case 'sales_agent':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatPosition = (position: string) => {
    return position.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-1.5 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 min-w-[200px]"
        disabled={loading}
      >
        {selectedEmployee ? (
          <>
            <div className="flex-1 text-left">
              <div className="text-xs font-medium text-gray-900">{selectedEmployee.name}</div>
              <div className="text-xs text-gray-500">{formatPosition(selectedEmployee.position)}</div>
            </div>
            <span className={`px-1.5 py-0.5 text-xs font-medium rounded-full ${getPositionBadgeColor(selectedEmployee.position)}`}>
              {formatPosition(selectedEmployee.position)}
            </span>
          </>
        ) : (
          <div className="flex-1 text-left text-gray-500 text-xs">
            {loading ? 'Loading employees...' : 'Select Employee'}
          </div>
        )}
        <svg className={`w-3 h-3 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto">
          {employees.length === 0 ? (
            <div className="px-3 py-2 text-xs text-gray-500">
              {loading ? 'Loading...' : 'No employees found'}
            </div>
          ) : (
            <>
              <button
                onClick={() => handleEmployeeSelect(null)}
                className="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none"
              >
                <div className="text-xs text-gray-500">All Employees</div>
              </button>
              <div className="border-t border-gray-100"></div>
              {employees.map((employee) => (
                <button
                  key={employee.id}
                  onClick={() => handleEmployeeSelect(employee)}
                  className={`w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none ${
                    selectedEmployee?.id === employee.id ? 'bg-primary-100 border-l-4 border-primary-500' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-medium text-gray-900">{employee.name}</div>
                      <div className="text-xs text-gray-500">{employee.email}</div>
                    </div>
                    <span className={`px-1.5 py-0.5 text-xs font-medium rounded-full ${getPositionBadgeColor(employee.position)}`}>
                      {formatPosition(employee.position)}
                    </span>
                  </div>
                </button>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
} 
'use client';

import { useState } from 'react';
import { clsx } from 'clsx';

interface CustomerData {
  name: string;
  company: string;
  email: string;
  phone: string;
  address: string;
}

interface VehicleSpecifications {
  engine: string;
  power: string;
  torque: string;
  fuel_type: string;
  transmission: string;
}

interface VehicleData {
  make: string;
  model: string;
  type: string;
  color: string;
  year: string;
  specifications: VehicleSpecifications;
}

interface AddOn {
  name: string;
  description: string;
  price: number;
}

interface PricingData {
  base_price: number;
  insurance: number;
  lto_fees: number;
  discounts: number;
  total_amount: number;
  add_ons: AddOn[];
  discount_description: string;
}

interface EmployeeData {
  name: string;
  position: string;
  email: string;
  phone: string;
  branch_name: string;
  branch_region: string;
}

interface QuotationData {
  quotation_number: string;
  customer: CustomerData;
  vehicle: VehicleData;
  pricing: PricingData;
  employee: EmployeeData;
}

const defaultData: QuotationData = {
  quotation_number: "Q2025-TEST-001",
  customer: {
    name: "Juan Dela Cruz",
    company: "ABC Corporation",
    email: "juan@abc.com.ph",
    phone: "09171234567",
    address: "123 Makati Avenue, Makati City, Metro Manila"
  },
  vehicle: {
    make: "Toyota",
    model: "Camry",
    type: "sedan",
    color: "Pearl White",
    year: "2025",
    specifications: {
      engine: "2.5L 4-Cylinder",
      power: "203",
      torque: "250",
      fuel_type: "gasoline",
      transmission: "automatic"
    }
  },
  pricing: {
    base_price: 1850000,
    insurance: 45000,
    lto_fees: 15000,
    discounts: 50000,
    total_amount: 1860000,
    add_ons: [
      {
        name: "Premium Audio System",
        description: "JBL Premium Sound System",
        price: 25000
      },
      {
        name: "Tint Package",
        description: "3M Ceramic Tint - Full Car",
        price: 15000
      }
    ],
    discount_description: "First Time Buyer Discount"
  },
  employee: {
    name: "Maria Santos",
    position: "sales_agent",
    email: "maria.santos@premiummotors.ph",
    phone: "09181234567",
    branch_name: "Makati Branch",
    branch_region: "central"
  }
};

export default function PDFTestPage() {
  const [formData, setFormData] = useState<QuotationData>(defaultData);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    success: boolean;
    message: string;
    pdfUrl?: string;
    htmlPreview?: string;
    storagePath?: string;
    storageUrl?: string;
    uploadStatus?: string;
  } | null>(null);
  const [activeTab, setActiveTab] = useState<'customer' | 'vehicle' | 'pricing' | 'employee'>('customer');

  const updateFormData = (section: keyof QuotationData, field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  const updateNestedFormData = (section: keyof QuotationData, subSection: string, field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [subSection]: {
          ...(prev[section] as any)[subSection],
          [field]: value
        }
      }
    }));
  };

  const addAddOn = () => {
    const newAddOn: AddOn = {
      name: "",
      description: "",
      price: 0
    };
    setFormData(prev => ({
      ...prev,
      pricing: {
        ...prev.pricing,
        add_ons: [...prev.pricing.add_ons, newAddOn]
      }
    }));
  };

  const updateAddOn = (index: number, field: keyof AddOn, value: any) => {
    setFormData(prev => ({
      ...prev,
      pricing: {
        ...prev.pricing,
        add_ons: prev.pricing.add_ons.map((addon, i) => 
          i === index ? { ...addon, [field]: value } : addon
        )
      }
    }));
  };

  const removeAddOn = (index: number) => {
    setFormData(prev => ({
      ...prev,
      pricing: {
        ...prev.pricing,
        add_ons: prev.pricing.add_ons.filter((_, i) => i !== index)
      }
    }));
  };

  const calculateTotal = () => {
    const { base_price, insurance, lto_fees, discounts, add_ons } = formData.pricing;
    const addOnTotal = add_ons.reduce((sum, addon) => sum + addon.price, 0);
    const total = base_price + insurance + lto_fees + addOnTotal - discounts;
    
    setFormData(prev => ({
      ...prev,
      pricing: {
        ...prev.pricing,
        total_amount: total
      }
    }));
  };

  const loadSampleData = (sampleType: 'luxury' | 'economy' | 'suv') => {
    const samples = {
      luxury: {
        ...defaultData,
        quotation_number: "Q2025-LUX-001",
        customer: {
          ...defaultData.customer,
          name: "Patricia Gonzalez",
          company: "Elite Business Solutions",
          email: "patricia@elitebiz.ph"
        },
        vehicle: {
          make: "BMW",
          model: "X5",
          type: "suv",
          color: "Alpine White",
          year: "2025",
          specifications: {
            engine: "3.0L Twin-Turbo I6",
            power: "335",
            torque: "450",
            fuel_type: "gasoline",
            transmission: "automatic"
          }
        },
        pricing: {
          base_price: 4500000,
          insurance: 120000,
          lto_fees: 25000,
          discounts: 100000,
          total_amount: 4545000,
          add_ons: [
            { name: "M Sport Package", description: "Performance styling and suspension", price: 250000 },
            { name: "Premium Sound", description: "Harman Kardon Audio System", price: 80000 }
          ],
          discount_description: "VIP Customer Discount"
        }
      },
      economy: {
        ...defaultData,
        quotation_number: "Q2025-ECO-001",
        customer: {
          ...defaultData.customer,
          name: "Miguel Santos",
          company: "",
          email: "miguel.santos@gmail.com"
        },
        vehicle: {
          make: "Mitsubishi",
          model: "Mirage",
          type: "hatchback",
          color: "Red Diamond",
          year: "2025",
          specifications: {
            engine: "1.2L 3-Cylinder",
            power: "78",
            torque: "100",
            fuel_type: "gasoline",
            transmission: "manual"
          }
        },
        pricing: {
          base_price: 720000,
          insurance: 25000,
          lto_fees: 12000,
          discounts: 20000,
          total_amount: 737000,
          add_ons: [],
          discount_description: "Cash Payment Discount"
        }
      },
      suv: {
        ...defaultData,
        quotation_number: "Q2025-SUV-001",
        customer: {
          ...defaultData.customer,
          name: "Roberto Cruz",
          company: "Cruz Family Business",
          email: "roberto@cruzfamily.ph"
        },
        vehicle: {
          make: "Ford",
          model: "Territory",
          type: "suv",
          color: "Moondust Silver",
          year: "2025",
          specifications: {
            engine: "1.5L EcoBoost Turbo",
            power: "141",
            torque: "225",
            fuel_type: "gasoline",
            transmission: "cvt"
          }
        },
        pricing: {
          base_price: 1350000,
          insurance: 55000,
          lto_fees: 18000,
          discounts: 30000,
          total_amount: 1393000,
          add_ons: [
            { name: "Roof Rails", description: "Aluminum roof rail system", price: 15000 },
            { name: "Floor Mats", description: "Weather-resistant floor mats", price: 8000 }
          ],
          discount_description: "Family Package Discount"
        }
      }
    };
    
    setFormData(samples[sampleType]);
    calculateTotal();
  };

  const generatePDF = async (preview: boolean = false, withStorage: boolean = false) => {
    setIsLoading(true);
    setResult(null);
    
    try {
      let endpoint: string;
      if (preview) {
        endpoint = 'http://localhost:8000/api/v1/test-pdf-preview';
      } else if (withStorage) {
        endpoint = 'http://localhost:8000/api/v1/test-pdf-generation-with-storage';
      } else {
        endpoint = 'http://localhost:8000/api/v1/test-pdf-generation';
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      if (preview) {
        const htmlContent = await response.text();
        setResult({
          success: true,
          message: 'HTML preview generated successfully!',
          htmlPreview: htmlContent
        });
      } else {
        // For PDF generation, we expect a blob response
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Check for storage information in headers
        const storagePath = response.headers.get('X-Storage-Path');
        const storageUrl = response.headers.get('X-Storage-URL');
        const uploadStatus = response.headers.get('X-Upload-Status');
        const uploadError = response.headers.get('X-Upload-Error');
        
        let message = 'PDF generated successfully!';
        if (withStorage) {
          if (uploadStatus === 'success' && storagePath) {
            message += ` PDF stored in Supabase at: ${storagePath}`;
          } else if (uploadStatus === 'failed') {
            message += ` (Storage failed: ${uploadError || 'Unknown error'})`;
          }
        }
        
        setResult({
          success: true,
          message,
          pdfUrl: url,
          storagePath: storagePath || undefined,
          storageUrl: storageUrl || undefined,
          uploadStatus: uploadStatus || undefined
        });
      }
    } catch (error) {
      setResult({
        success: false,
        message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  const InputField = ({ 
    label, 
    value, 
    onChange, 
    type = 'text', 
    placeholder 
  }: {
    label: string;
    value: string | number;
    onChange: (value: string | number) => void;
    type?: string;
    placeholder?: string;
  }) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
        placeholder={placeholder}
        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white shadow-xl rounded-lg">
          {/* Header */}
          <div className="bg-blue-600 text-white px-6 py-4 rounded-t-lg">
            <h1 className="text-2xl font-bold">PDF Quotation Generator - Test Interface</h1>
            <p className="text-blue-100 mt-1">Generate and preview vehicle quotation PDFs with sample data</p>
          </div>

          <div className="p-6">
            {/* Sample Data Buttons */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">Quick Load Sample Data</h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => loadSampleData('economy')}
                  className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
                >
                  Economy Car (Mirage)
                </button>
                <button
                  onClick={() => loadSampleData('suv')}
                  className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                >
                  Family SUV (Territory)
                </button>
                <button
                  onClick={() => loadSampleData('luxury')}
                  className="px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 transition-colors"
                >
                  Luxury SUV (BMW X5)
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Form Section */}
              <div>
                {/* Tab Navigation */}
                <div className="border-b border-gray-200 mb-6">
                  <nav className="-mb-px flex space-x-8">
                    {[
                      { key: 'customer', label: 'Customer' },
                      { key: 'vehicle', label: 'Vehicle' },
                      { key: 'pricing', label: 'Pricing' },
                      { key: 'employee', label: 'Sales Rep' }
                    ].map(({ key, label }) => (
                      <button
                        key={key}
                        onClick={() => setActiveTab(key as any)}
                        className={clsx(
                          'py-2 px-1 border-b-2 font-medium text-sm',
                          activeTab === key
                            ? 'border-blue-500 text-blue-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                        )}
                      >
                        {label}
                      </button>
                    ))}
                  </nav>
                </div>

                {/* Form Content */}
                <div className="space-y-6">
                  {/* Quotation Number */}
                  <InputField
                    label="Quotation Number"
                    value={formData.quotation_number}
                    onChange={(value) => setFormData(prev => ({ ...prev, quotation_number: value as string }))}
                    placeholder="Q2025-001"
                  />

                  {/* Customer Tab */}
                  {activeTab === 'customer' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-gray-800">Customer Information</h3>
                      <InputField
                        label="Full Name"
                        value={formData.customer.name}
                        onChange={(value) => updateFormData('customer', 'name', value)}
                        placeholder="Juan Dela Cruz"
                      />
                      <InputField
                        label="Company (Optional)"
                        value={formData.customer.company}
                        onChange={(value) => updateFormData('customer', 'company', value)}
                        placeholder="ABC Corporation"
                      />
                      <InputField
                        label="Email Address"
                        value={formData.customer.email}
                        onChange={(value) => updateFormData('customer', 'email', value)}
                        type="email"
                        placeholder="juan@abc.com.ph"
                      />
                      <InputField
                        label="Phone Number"
                        value={formData.customer.phone}
                        onChange={(value) => updateFormData('customer', 'phone', value)}
                        placeholder="09171234567"
                      />
                      <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Address
                        </label>
                        <textarea
                          value={formData.customer.address}
                          onChange={(e) => updateFormData('customer', 'address', e.target.value)}
                          placeholder="123 Makati Avenue, Makati City, Metro Manila"
                          rows={3}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                    </div>
                  )}

                  {/* Vehicle Tab */}
                  {activeTab === 'vehicle' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-gray-800">Vehicle Details</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <InputField
                          label="Make"
                          value={formData.vehicle.make}
                          onChange={(value) => updateFormData('vehicle', 'make', value)}
                          placeholder="Toyota"
                        />
                        <InputField
                          label="Model"
                          value={formData.vehicle.model}
                          onChange={(value) => updateFormData('vehicle', 'model', value)}
                          placeholder="Camry"
                        />
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="mb-4">
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Type
                          </label>
                          <select
                            value={formData.vehicle.type}
                            onChange={(e) => updateFormData('vehicle', 'type', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="sedan">Sedan</option>
                            <option value="suv">SUV</option>
                            <option value="hatchback">Hatchback</option>
                            <option value="pickup">Pickup</option>
                            <option value="van">Van</option>
                            <option value="motorcycle">Motorcycle</option>
                            <option value="truck">Truck</option>
                          </select>
                        </div>
                        <InputField
                          label="Color"
                          value={formData.vehicle.color}
                          onChange={(value) => updateFormData('vehicle', 'color', value)}
                          placeholder="Pearl White"
                        />
                        <InputField
                          label="Year"
                          value={formData.vehicle.year}
                          onChange={(value) => updateFormData('vehicle', 'year', value)}
                          placeholder="2025"
                        />
                      </div>
                      
                      <h4 className="text-md font-semibold text-gray-700 mt-6">Technical Specifications</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <InputField
                          label="Engine"
                          value={formData.vehicle.specifications.engine}
                          onChange={(value) => updateNestedFormData('vehicle', 'specifications', 'engine', value)}
                          placeholder="2.5L 4-Cylinder"
                        />
                        <InputField
                          label="Power (HP)"
                          value={formData.vehicle.specifications.power}
                          onChange={(value) => updateNestedFormData('vehicle', 'specifications', 'power', value)}
                          placeholder="203"
                        />
                        <InputField
                          label="Torque (Nm)"
                          value={formData.vehicle.specifications.torque}
                          onChange={(value) => updateNestedFormData('vehicle', 'specifications', 'torque', value)}
                          placeholder="250"
                        />
                        <div className="mb-4">
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Fuel Type
                          </label>
                          <select
                            value={formData.vehicle.specifications.fuel_type}
                            onChange={(e) => updateNestedFormData('vehicle', 'specifications', 'fuel_type', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="gasoline">Gasoline</option>
                            <option value="diesel">Diesel</option>
                            <option value="hybrid">Hybrid</option>
                            <option value="electric">Electric</option>
                          </select>
                        </div>
                        <div className="mb-4">
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Transmission
                          </label>
                          <select
                            value={formData.vehicle.specifications.transmission}
                            onChange={(e) => updateNestedFormData('vehicle', 'specifications', 'transmission', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="manual">Manual</option>
                            <option value="automatic">Automatic</option>
                            <option value="cvt">CVT</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Pricing Tab */}
                  {activeTab === 'pricing' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-gray-800">Pricing Details</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <InputField
                          label="Base Price (₱)"
                          value={formData.pricing.base_price}
                          onChange={(value) => updateFormData('pricing', 'base_price', value)}
                          type="number"
                          placeholder="1850000"
                        />
                        <InputField
                          label="Insurance (₱)"
                          value={formData.pricing.insurance}
                          onChange={(value) => updateFormData('pricing', 'insurance', value)}
                          type="number"
                          placeholder="45000"
                        />
                        <InputField
                          label="LTO Fees (₱)"
                          value={formData.pricing.lto_fees}
                          onChange={(value) => updateFormData('pricing', 'lto_fees', value)}
                          type="number"
                          placeholder="15000"
                        />
                        <InputField
                          label="Discounts (₱)"
                          value={formData.pricing.discounts}
                          onChange={(value) => updateFormData('pricing', 'discounts', value)}
                          type="number"
                          placeholder="50000"
                        />
                      </div>
                      <InputField
                        label="Discount Description"
                        value={formData.pricing.discount_description}
                        onChange={(value) => updateFormData('pricing', 'discount_description', value)}
                        placeholder="First Time Buyer Discount"
                      />
                      
                      <div className="mt-6">
                        <div className="flex justify-between items-center mb-4">
                          <h4 className="text-md font-semibold text-gray-700">Add-ons</h4>
                          <button
                            onClick={addAddOn}
                            className="px-3 py-1 bg-green-500 text-white text-sm rounded-md hover:bg-green-600 transition-colors"
                          >
                            Add Item
                          </button>
                        </div>
                        
                        {formData.pricing.add_ons.map((addon, index) => (
                          <div key={index} className="border border-gray-200 rounded-md p-4 mb-4">
                            <div className="flex justify-between items-start mb-2">
                              <h5 className="font-medium">Add-on #{index + 1}</h5>
                              <button
                                onClick={() => removeAddOn(index)}
                                className="text-red-500 hover:text-red-700 text-sm"
                              >
                                Remove
                              </button>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              <InputField
                                label="Name"
                                value={addon.name}
                                onChange={(value) => updateAddOn(index, 'name', value)}
                                placeholder="Premium Audio System"
                              />
                              <InputField
                                label="Price (₱)"
                                value={addon.price}
                                onChange={(value) => updateAddOn(index, 'price', value)}
                                type="number"
                                placeholder="25000"
                              />
                            </div>
                            <InputField
                              label="Description"
                              value={addon.description}
                              onChange={(value) => updateAddOn(index, 'description', value)}
                              placeholder="JBL Premium Sound System"
                            />
                          </div>
                        ))}
                      </div>
                      
                      <div className="mt-6 p-4 bg-gray-50 rounded-md">
                        <div className="flex justify-between items-center">
                          <span className="font-semibold text-lg">Total Amount:</span>
                          <div className="text-right">
                            <span className="text-2xl font-bold text-green-600">
                              ₱{formData.pricing.total_amount.toLocaleString()}
                            </span>
                            <br />
                            <button
                              onClick={calculateTotal}
                              className="text-sm text-blue-500 hover:text-blue-700"
                            >
                              Recalculate
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Employee Tab */}
                  {activeTab === 'employee' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-gray-800">Sales Representative</h3>
                      <InputField
                        label="Name"
                        value={formData.employee.name}
                        onChange={(value) => updateFormData('employee', 'name', value)}
                        placeholder="Maria Santos"
                      />
                      <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Position
                        </label>
                        <select
                          value={formData.employee.position}
                          onChange={(e) => updateFormData('employee', 'position', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                          <option value="sales_agent">Sales Agent</option>
                          <option value="account_executive">Account Executive</option>
                          <option value="manager">Manager</option>
                          <option value="director">Director</option>
                        </select>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <InputField
                          label="Email"
                          value={formData.employee.email}
                          onChange={(value) => updateFormData('employee', 'email', value)}
                          type="email"
                          placeholder="maria.santos@premiummotors.ph"
                        />
                        <InputField
                          label="Phone"
                          value={formData.employee.phone}
                          onChange={(value) => updateFormData('employee', 'phone', value)}
                          placeholder="09181234567"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <InputField
                          label="Branch Name"
                          value={formData.employee.branch_name}
                          onChange={(value) => updateFormData('employee', 'branch_name', value)}
                          placeholder="Makati Branch"
                        />
                        <div className="mb-4">
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Region
                          </label>
                          <select
                            value={formData.employee.branch_region}
                            onChange={(e) => updateFormData('employee', 'branch_region', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="north">North</option>
                            <option value="south">South</option>
                            <option value="east">East</option>
                            <option value="west">West</option>
                            <option value="central">Central</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Actions and Results Section */}
              <div>
                <div className="sticky top-8">
                  {/* Action Buttons */}
                  <div className="bg-gray-50 p-6 rounded-lg mb-6">
                    <h3 className="text-lg font-semibold mb-4">Generate PDF</h3>
                    <div className="space-y-3">
                      <button
                        onClick={() => generatePDF(true)}
                        disabled={isLoading}
                        className="w-full px-4 py-3 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {isLoading ? 'Generating...' : 'Preview HTML'}
                      </button>
                      <button
                        onClick={() => generatePDF(false)}
                        disabled={isLoading}
                        className="w-full px-4 py-3 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {isLoading ? 'Generating...' : 'Generate PDF'}
                      </button>
                      <button
                        onClick={() => generatePDF(false, true)}
                        disabled={isLoading}
                        className="w-full px-4 py-3 bg-purple-500 text-white rounded-md hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold"
                      >
                        {isLoading ? 'Generating...' : '🚀 Generate + Store in Supabase'}
                      </button>
                    </div>
                  </div>

                  {/* Results */}
                  {result && (
                    <div className={clsx(
                      'p-4 rounded-lg mb-6',
                      result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                    )}>
                      <div className={clsx(
                        'font-medium',
                        result.success ? 'text-green-800' : 'text-red-800'
                      )}>
                        {result.success ? '✅ Success!' : '❌ Error'}
                      </div>
                      <div className={clsx(
                        'text-sm mt-1',
                        result.success ? 'text-green-700' : 'text-red-700'
                      )}>
                        {result.message}
                      </div>
                      
                      {result.pdfUrl && (
                        <div className="mt-3 space-y-3">
                          <div className="flex flex-wrap gap-2">
                            <a
                              href={result.pdfUrl}
                              download="quotation.pdf"
                              className="inline-block px-4 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
                            >
                              📥 Download PDF
                            </a>
                            {result.storageUrl && (
                              <a
                                href={result.storageUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-block px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
                              >
                                ☁️ View in Supabase
                              </a>
                            )}
                          </div>
                          
                          {result.storagePath && (
                            <div className="text-xs text-gray-600 bg-gray-100 p-2 rounded">
                              <strong>Storage Path:</strong> {result.storagePath}
                              <br />
                              <strong>Upload Status:</strong> <span className={result.uploadStatus === 'success' ? 'text-green-600' : 'text-red-600'}>
                                {result.uploadStatus || 'N/A'}
                              </span>
                            </div>
                          )}
                          
                          <div>
                            <iframe
                              src={result.pdfUrl}
                              className="w-full h-96 border border-gray-300 rounded-md"
                              title="PDF Preview"
                            />
                          </div>
                        </div>
                      )}
                      
                      {result.htmlPreview && (
                        <div className="mt-3">
                          <details className="cursor-pointer">
                            <summary className="font-medium text-green-700 hover:text-green-800">
                              View HTML Source
                            </summary>
                            <pre className="mt-2 p-3 bg-gray-100 text-xs overflow-x-auto rounded border max-h-64">
                              {result.htmlPreview}
                            </pre>
                          </details>
                        </div>
                      )}
                    </div>
                  )}

                  {/* JSON Data Preview */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Current Data (JSON)</h4>
                    <details className="cursor-pointer">
                      <summary className="text-sm text-gray-600 hover:text-gray-800">
                        View/Copy JSON Data
                      </summary>
                      <pre className="mt-2 p-3 bg-white text-xs overflow-x-auto rounded border max-h-64">
                        {JSON.stringify(formData, null, 2)}
                      </pre>
                    </details>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { apiClient } from '@/lib/api';
import { SystemHealth } from '@/types';

// Set document title dynamically
if (typeof document !== 'undefined') {
  document.title = 'Tobi - AI Sales Copilot';
}

export default function ClientHomePage() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [isOnline, setIsOnline] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [featuresVisible, setFeaturesVisible] = useState(false);
  const [howItWorksVisible, setHowItWorksVisible] = useState(false);
  const [ctaVisible, setCtaVisible] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);
  const howItWorksRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkSystemStatus = async () => {
      try {
        const healthResponse = await apiClient.getHealth();
        if (healthResponse.success && healthResponse.data) {
          setSystemHealth(healthResponse.data);
          setIsOnline(true);
        }
      } catch (err) {
        setIsOnline(false);
      }
    };

    checkSystemStatus();

    // Handle scroll effects
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    // Handle keyboard events
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && showLoginModal) {
        setShowLoginModal(false);
      }
    };

    // Handle intersection observer for animations
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.target === heroRef.current) {
            setIsVisible(entry.isIntersecting);
          } else if (entry.target === featuresRef.current) {
            setFeaturesVisible(entry.isIntersecting);
          } else if (entry.target === howItWorksRef.current) {
            setHowItWorksVisible(entry.isIntersecting);
          } else if (entry.target === ctaRef.current) {
            setCtaVisible(entry.isIntersecting);
          }
        });
      },
      { threshold: 0.1 }
    );

    if (heroRef.current) {
      observer.observe(heroRef.current);
    }
    if (featuresRef.current) {
      observer.observe(featuresRef.current);
    }
    if (howItWorksRef.current) {
      observer.observe(howItWorksRef.current);
    }
    if (ctaRef.current) {
      observer.observe(ctaRef.current);
    }

    window.addEventListener('scroll', handleScroll);
    window.addEventListener('keydown', handleKeyDown);
    
    // Initial animation trigger
    setTimeout(() => setIsVisible(true), 100);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('keydown', handleKeyDown);
      if (heroRef.current) {
        observer.unobserve(heroRef.current);
      }
      if (featuresRef.current) {
        observer.unobserve(featuresRef.current);
      }
      if (howItWorksRef.current) {
        observer.unobserve(howItWorksRef.current);
      }
      if (ctaRef.current) {
        observer.unobserve(ctaRef.current);
      }
    };
  }, [showLoginModal]);

  return (
    <div className="min-h-screen bg-white">
      {/* Minimal Header */}
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled 
          ? 'bg-white/90 backdrop-blur-md shadow-lg border-b border-gray-200/50' 
          : 'bg-gradient-to-b from-black/20 to-transparent'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className={`text-2xl font-bold transition-colors duration-300 ${
                isScrolled ? 'text-primary-600' : 'text-white drop-shadow-lg'
              }`}>
                Tobi
              </h1>
            </div>
            <button 
              onClick={() => setShowLoginModal(true)}
              className={`px-6 py-2 rounded-lg font-semibold transition-all duration-300 hover:scale-105 ${
                isScrolled 
                  ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg' 
                  : 'bg-white/20 backdrop-blur-sm text-white border border-white/40 hover:bg-white/30 shadow-lg'
              }`}>
              Log In
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div ref={heroRef} className="relative overflow-hidden min-h-screen flex items-center">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800"></div>
        
        {/* Animated background elements */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-white/5 rounded-full blur-3xl animate-spin slow-spin"></div>
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className={`text-center transition-all duration-1000 ${
            isVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          }`}>
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight">
              Meet Your AI-Powered
              <span className="block text-primary-100 animate-fade-in-up delay-300">Sales Copilot</span>
            </h1>
            <p className={`text-xl text-primary-100 mb-8 max-w-3xl mx-auto transition-all duration-1000 delay-200 ${
              isVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>
              Tobi transforms your sales process with intelligent document analysis, 
              instant information retrieval, and personalized customer insights. 
              Get the answers you need, when you need them.
            </p>
            <div className={`flex flex-col sm:flex-row justify-center gap-4 transition-all duration-1000 delay-400 ${
              isVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>
              <button className="bg-white text-primary-600 px-8 py-4 rounded-lg font-semibold hover:bg-primary-50 transition-all duration-300 text-lg hover:scale-105 hover:shadow-lg">
                Start Conversation
              </button>
              <button className="border-2 border-white text-white px-8 py-4 rounded-lg font-semibold hover:bg-white hover:text-primary-600 transition-all duration-300 text-lg hover:scale-105">
                Watch Demo
              </button>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2 animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div ref={featuresRef} className="bg-gray-50 py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`text-center mb-16 transition-all duration-1000 ${
            featuresVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          }`}>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why Sales Teams Choose Tobi
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Powerful AI capabilities designed specifically for sales professionals
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 hover:shadow-lg hover:scale-105 transition-all duration-300 group ${
              featuresVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>

              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Instant Knowledge Access
              </h3>
              <p className="text-gray-600">
                Upload your sales materials, product docs, and training content. 
                Get instant answers from your knowledge base during customer conversations.
              </p>
            </div>

            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 hover:shadow-lg hover:scale-105 transition-all duration-300 group delay-200 ${
              featuresVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>

              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Smart Conversation Support
              </h3>
              <p className="text-gray-600">
                Get real-time suggestions, objection handling tips, and relevant 
                information tailored to your specific customer conversation.
              </p>
            </div>

            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 hover:shadow-lg hover:scale-105 transition-all duration-300 group delay-400 ${
              featuresVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>

              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Customer Insights
              </h3>
              <p className="text-gray-600">
                Track conversation history, identify patterns, and get personalized 
                recommendations for each customer relationship.
              </p>
            </div>

            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 hover:shadow-lg hover:scale-105 transition-all duration-300 group delay-100 ${
              featuresVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>

              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Lightning Fast
              </h3>
              <p className="text-gray-600">
                Advanced AI retrieval system delivers accurate answers in seconds, 
                so you never miss a beat in your sales conversations.
              </p>
            </div>

            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 hover:shadow-lg hover:scale-105 transition-all duration-300 group delay-300 ${
              featuresVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>

              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Secure & Private
              </h3>
              <p className="text-gray-600">
                Enterprise-grade security ensures your sensitive sales data and 
                customer information remains protected at all times.
              </p>
            </div>

            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 hover:shadow-lg hover:scale-105 transition-all duration-300 group delay-500 ${
              featuresVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>

              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Tailored Responses
              </h3>
              <p className="text-gray-600">
                AI understands your sales methodology and company voice, 
                providing responses that align with your unique approach.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div ref={howItWorksRef} className="py-24 bg-gradient-to-b from-white to-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`text-center mb-16 transition-all duration-1000 ${
            howItWorksVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          }`}>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How Tobi Works
            </h2>
            <p className="text-xl text-gray-600">
              Simple setup, powerful results
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className={`text-center group transition-all duration-1000 delay-200 ${
              howItWorksVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 group-hover:bg-primary-200 transition-all duration-300 animate-float">
                <span className="text-2xl font-bold text-primary-600">1</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Upload Your Content
              </h3>
              <p className="text-gray-600">
                Add your sales playbooks, product documentation, case studies, 
                and training materials to build your knowledge base.
              </p>
            </div>

            <div className={`text-center group transition-all duration-1000 delay-400 ${
              howItWorksVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 group-hover:bg-primary-200 transition-all duration-300 animate-float delay-200">
                <span className="text-2xl font-bold text-primary-600">2</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Start Conversations
              </h3>
              <p className="text-gray-600">
                Ask questions, get guidance, and receive intelligent suggestions 
                tailored to your specific customer and situation.
              </p>
            </div>

            <div className={`text-center group transition-all duration-1000 delay-600 ${
              howItWorksVisible 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-8'
            }`}>
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 group-hover:bg-primary-200 transition-all duration-300 animate-float delay-400">
                <span className="text-2xl font-bold text-primary-600">3</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4 group-hover:text-primary-600 transition-colors duration-300">
                Close More Deals
              </h3>
              <p className="text-gray-600">
                With instant access to relevant information and AI-powered insights, 
                transform every customer interaction into a winning opportunity.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div ref={ctaRef} className="bg-gradient-to-r from-primary-600 via-primary-700 to-primary-800 py-20 relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/6 w-32 h-32 bg-white/5 rounded-full blur-2xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/6 w-48 h-48 bg-white/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative z-10 max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <h2 className={`text-3xl font-bold text-white mb-6 transition-all duration-1000 ${
            ctaVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          }`}>
            Ready to Transform Your Sales Process?
          </h2>
          <p className={`text-xl text-primary-100 mb-8 transition-all duration-1000 delay-200 ${
            ctaVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          }`}>
            Join forward-thinking sales teams who are already using AI to close more deals faster.
          </p>
          <div className={`flex flex-col sm:flex-row justify-center gap-4 transition-all duration-1000 delay-400 ${
            ctaVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          }`}>
            <button className="bg-white text-primary-600 px-8 py-4 rounded-lg font-semibold hover:bg-primary-50 hover:scale-105 transition-all duration-300 text-lg shadow-lg hover:shadow-xl">
              Get Started Now
            </button>
            <button className="border-2 border-white text-white px-8 py-4 rounded-lg font-semibold hover:bg-white hover:text-primary-600 hover:scale-105 transition-all duration-300 text-lg">
              Schedule Demo
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                         <div>
               <h3 className="text-lg font-semibold mb-4">Tobi</h3>
               <p className="text-gray-400">
                 AI-powered sales copilot designed to help you close more deals 
                 with intelligent conversation support.
               </p>
             </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
                Product
              </h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white">Features</a></li>
                <li><a href="#" className="hover:text-white">Pricing</a></li>
                <li><a href="#" className="hover:text-white">Documentation</a></li>
                <li><a href="#" className="hover:text-white">API</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
                Support
              </h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white">Help Center</a></li>
                <li><a href="#" className="hover:text-white">Contact Us</a></li>
                <li><a href="#" className="hover:text-white">Training</a></li>
                <li><a href="#" className="hover:text-white">Status</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
                Company
              </h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white">About</a></li>
                <li><a href="#" className="hover:text-white">Blog</a></li>
                <li><a href="#" className="hover:text-white">Careers</a></li>
                <li><a href="#" className="hover:text-white">Privacy</a></li>
              </ul>
            </div>
          </div>
                     <div className="border-t border-gray-800 mt-8 pt-8 flex justify-center items-center text-gray-400">
             <p>&copy; 2024 Tobi. All rights reserved.</p>
             <div className="ml-6 flex space-x-4">
               <Link
                 href="/console"
                 className="text-xs text-gray-500 hover:text-gray-300 opacity-50 hover:opacity-100 transition-opacity"
               >
                 Sales Console
               </Link>
               <Link
                 href="/watchtower"
                 className="text-xs text-gray-500 hover:text-gray-300 opacity-50 hover:opacity-100 transition-opacity"
               >
                 Sales Watchtower
               </Link>
               <Link
                 href="/dev"
                 className="text-xs text-gray-500 hover:text-gray-300 opacity-50 hover:opacity-100 transition-opacity"
               >
                 Dev Tools
               </Link>
             </div>
           </div>
                 </div>
       </footer>

       {/* Login Modal */}
       {showLoginModal && (
         <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
           <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-8 transform transition-all duration-300 scale-100">
             <div className="flex justify-between items-center mb-6">
               <h2 className="text-2xl font-bold text-gray-900">Welcome to Tobi</h2>
               <button 
                 onClick={() => setShowLoginModal(false)}
                 className="text-gray-400 hover:text-gray-600 transition-colors duration-200"
               >
                 <span className="sr-only">Close</span>
                 <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                 </svg>
               </button>
             </div>
             <div className="space-y-4">
               <div>
                 <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
                 <input
                   type="email"
                   id="email"
                   className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                   placeholder="Enter your email"
                 />
               </div>
               <div>
                 <label htmlFor="password" className="block text-sm font-medium text-gray-700">Password</label>
                 <input
                   type="password"
                   id="password"
                   className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                   placeholder="Enter your password"
                 />
               </div>
               <div className="flex items-center justify-between">
                 <div className="flex items-center">
                   <input
                     id="remember-me"
                     name="remember-me"
                     type="checkbox"
                     className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                   />
                   <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-900">
                     Remember me
                   </label>
                 </div>
                 <div className="text-sm">
                   <a href="#" className="font-medium text-primary-600 hover:text-primary-500">
                     Forgot password?
                   </a>
                 </div>
               </div>
               <button
                 type="submit"
                 className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200"
               >
                 Sign In
               </button>
               <div className="text-center">
                 <span className="text-sm text-gray-600">
                   Don't have an account?{' '}
                   <a href="#" className="font-medium text-primary-600 hover:text-primary-500">
                     Sign up
                   </a>
                 </span>
               </div>
             </div>
           </div>
         </div>
       )}
     </div>
   );
 } 
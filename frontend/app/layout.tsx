import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'RAG-Tobi - Salesperson Copilot',
  description: 'AI-powered salesperson copilot with RAG capabilities',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50 antialiased`}>
        <div className="min-h-full">
          <header className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center h-16">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <h1 className="text-2xl font-bold text-primary-600">
                      🤖 RAG-Tobi
                    </h1>
                  </div>
                  <div className="hidden md:block">
                    <div className="ml-10 flex items-baseline space-x-4">
                      <span className="text-sm text-gray-500">
                        Salesperson Copilot
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  {/* Navigation Links */}
                  <nav className="hidden md:flex space-x-4">
                    <a
                      href="/"
                      className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                    >
                      Dashboard
                    </a>
                    <a
                      href="/manage"
                      className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                    >
                      Manage
                    </a>
                    <a
                      href="/dualagentdebug"
                      className="text-blue-600 hover:text-blue-800 px-3 py-2 rounded-md text-sm font-medium border border-blue-200 hover:border-blue-300"
                    >
                      🚀 Debug
                    </a>
                    <a
                      href="/memorycheck"
                      className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                    >
                      Memory
                    </a>
                  </nav>
                  
                  <div className="hidden md:block">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-sm text-gray-600">Online</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </header>

          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <div className="px-4 py-6 sm:px-0">
              {children}
            </div>
          </main>

          <footer className="bg-white border-t border-gray-200 mt-auto">
            <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-500">
                  © 2024 RAG-Tobi. AI-powered sales assistance.
                </div>
                <div className="flex space-x-6">
                  <span className="text-sm text-gray-500">
                    Version 1.0.0
                  </span>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
} 
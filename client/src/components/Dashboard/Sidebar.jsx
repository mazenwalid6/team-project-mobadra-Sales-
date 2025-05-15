import React, { useEffect } from "react";
import { Link } from "wouter";

export default function Sidebar({ isOpen, isMobile, onDashboardClick, onForecastClick, onClose }) {
  // Sidebar is always visible on desktop, toggled on mobile
  const sidebarClasses = isMobile
    ? `fixed left-0 top-0 w-64 h-screen z-50 bg-black/50 border-r border-border overflow-y-auto transition-transform duration-300 ${isOpen ? 'translate-x-0' : '-translate-x-full'}`
    : 'fixed left-0 top-0 w-64 h-screen bg-black/50 border-r border-border overflow-y-auto';

  // Prevent body scroll when sidebar is open on mobile
  useEffect(() => {
    if (isMobile && isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isMobile, isOpen]);

  return (
    <>
      {/* Overlay for mobile when sidebar is open */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={onClose}
        />
      )}
      <div className={sidebarClasses} style={{ pointerEvents: isMobile && !isOpen ? 'none' : 'auto' }}>
        <div className="flex items-center px-5 h-16 border-b border-border">
          <div className="flex items-center">
            <svg
              viewBox="0 0 24 24"
              className="h-8 w-8 text-white mr-2"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M8 14s1.5 2 4 2 4-2 4-2" />
              <line x1="9" y1="9" x2="9.01" y2="9" />
              <line x1="15" y1="9" x2="15.01" y2="9" />
            </svg>
            <span className="text-lg font-bold text-white">Sales<span className="text-secondary">Forecast</span></span>
          </div>
        </div>

        <div className="flex flex-col flex-grow overflow-y-auto">
          <div className="pt-2">
            <div className="flex items-center px-3 py-2 text-xs uppercase tracking-wider text-gray-400 mb-1">
              <span>Project Overview</span>
            </div>
            <div className="px-3 py-2 text-sm text-gray-300">
              <p className="mb-2">A sales forecasting and optimization system using historical sales data to analyze and predict sales trends.</p>
              <div className="mt-4 space-y-2">
                <h4 className="font-semibold text-white">Tech Stack:</h4>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li>Python (Pandas, NumPy, Matplotlib)</li>
                  <li>Seaborn, Plotly, Statsmodels</li>
                  <li>Prophet, MLflow</li>
                  <li>MERN Stack</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="pt-4">
            <div className="flex items-center px-3 py-2 text-xs uppercase tracking-wider text-gray-400 mb-1">
              <span>Team Members</span>
            </div>
            <div className="space-y-2">
              <div className="px-3 py-2 bg-black/30 rounded-md">
                <div className="flex items-center">
                  <div className="h-8 w-8 bg-secondary/20 rounded-full flex items-center justify-center mr-2">
                    <span className="text-xs font-medium text-secondary">MW</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Mazen Walid</p>
                    <p className="text-xs text-gray-400">Team Leader</p>
                  </div>
                </div>
              </div>

              <div className="px-3 py-2 bg-black/30 rounded-md">
                <div className="flex items-center">
                  <div className="h-8 w-8 bg-secondary/20 rounded-full flex items-center justify-center mr-2">
                    <span className="text-xs font-medium text-secondary">YO</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Youssef Osama</p>
                    <p className="text-xs text-gray-400">Data Analysis</p>
                  </div>
                </div>
              </div>

              <div className="px-3 py-2 bg-black/30 rounded-md">
                <div className="flex items-center">
                  <div className="h-8 w-8 bg-secondary/20 rounded-full flex items-center justify-center mr-2">
                    <span className="text-xs font-medium text-secondary">ME</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Mohamed Elbadry</p>
                    <p className="text-xs text-gray-400">ML Engineer & Full Stack</p>
                  </div>
                </div>
              </div>

              <div className="px-3 py-2 bg-black/30 rounded-md">
                <div className="flex items-center">
                  <div className="h-8 w-8 bg-secondary/20 rounded-full flex items-center justify-center mr-2">
                    <span className="text-xs font-medium text-secondary">AO</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Abdallah Osama</p>
                    <p className="text-xs text-gray-400">ML Engineer</p>
                  </div>
                </div>
              </div>

              <div className="px-3 py-2 bg-black/30 rounded-md">
                <div className="flex items-center">
                  <div className="h-8 w-8 bg-secondary/20 rounded-full flex items-center justify-center mr-2">
                    <span className="text-xs font-medium text-secondary">MA</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Mohamed Ali</p>
                    <p className="text-xs text-gray-400">ML Engineer</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="pt-4">
            <div className="flex items-center px-3 py-2 text-xs uppercase tracking-wider text-gray-400 mb-1">
              <span>Quick Links</span>
            </div>
            <button 
              onClick={onDashboardClick}
              className="w-full group flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-300 hover:bg-black/60 hover:text-white"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="mr-3 h-5 w-5 text-gray-300"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="7" height="9"/>
                <rect x="14" y="3" width="7" height="5"/>
                <rect x="14" y="12" width="7" height="9"/>
                <rect x="3" y="16" width="7" height="5"/>
              </svg>
              Dashboard
            </button>
            <button 
              onClick={onForecastClick}
              className="w-full group flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-300 hover:bg-black/60 hover:text-white"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="mr-3 h-5 w-5 text-gray-300"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                <path d="M3 3v5h5"/>
                <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
                <path d="M16 21h5v-5"/>
              </svg>
              Forecast
            </button>
          </div>
        </div>

        <div className="flex-shrink-0 flex border-t border-border p-4">
          <Link href="/logout" className="w-full text-sm text-center text-gray-400 hover:text-white">
            Logout
          </Link>
        </div>
      </div>
    </>
  );
}
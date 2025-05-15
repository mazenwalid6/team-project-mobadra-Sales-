import React, { useState, useEffect } from "react";
import { useToast } from "../hooks/use-toast";
import KPICards from "../components/Dashboard/KPICards";
import SalesForecastChart from "../components/Dashboard/SalesForecastChart";
import AccuracyChart from "../components/Dashboard/AccuracyChart";
import TrendsChart from "../components/Dashboard/TrendsChart";
import ModelMetrics from "../components/Dashboard/ModelMetrics";
import ForecastsTable from "../components/Dashboard/ForecastsTable";
import Sidebar from "../components/Dashboard/Sidebar";
import Header from "../components/Dashboard/Header";

export default function Forecasts() {
  const [forecastData, setForecastData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1024);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:5001/data");
        const data = await response.json();
        if (data.error) {
          throw new Error(data.error);
        }
        setForecastData(data);
      } catch (error) {
        console.error("Error fetching forecast data:", error);
        toast({
          title: "Error",
          description: "Failed to load forecast data.",
          variant: "destructive",
        });
      }
      setIsLoading(false);
    };
    fetchData();
  }, [toast]);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  if (!forecastData) {
    return <div className="flex h-screen items-center justify-center text-foreground">Loading...</div>;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        isOpen={sidebarOpen || !isMobile}
        isMobile={isMobile}
        onClose={() => setSidebarOpen(false)}
        onDashboardClick={() => setSidebarOpen(false)}
        onForecastClick={() => setSidebarOpen(false)}
      />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header onMenuClick={() => setSidebarOpen(true)} />
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          <div className="max-w-7xl mx-auto space-y-6">
            <div className="bg-gradient-to-r from-secondary/20 to-black/50 rounded-lg p-6 animate-fade-in">
              <h1 className="text-2xl font-bold text-foreground">Sales Forecasts</h1>
              <p className="text-sm text-muted-foreground mt-1">
                View detailed sales forecasts and analysis.
              </p>
            </div>
            <KPICards data={forecastData} isLoading={isLoading} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SalesForecastChart data={forecastData.salesForecast} isLoading={isLoading} />
              <AccuracyChart data={forecastData.accuracyData} isLoading={isLoading} />
              <TrendsChart data={forecastData.trendsData} isLoading={isLoading} />
              <ModelMetrics data={forecastData.modelMetrics} isLoading={isLoading} />
            </div>
            <ForecastsTable data={forecastData.forecasts} isLoading={isLoading} />
          </div>
        </main>
      </div>
    </div>
  );
}
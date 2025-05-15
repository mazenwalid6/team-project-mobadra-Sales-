import React, { useState, useEffect } from "react";
import { useToast } from "../hooks/use-toast";
import UploadForm from "../components/Dashboard/UploadForm";
import KPICards from "../components/Dashboard/KPICards";
import SalesForecastChart from "../components/Dashboard/SalesForecastChart";
import AccuracyChart from "../components/Dashboard/AccuracyChart";
import TrendsChart from "../components/Dashboard/TrendsChart";
import ModelMetrics from "../components/Dashboard/ModelMetrics";
import ForecastsTable from "../components/Dashboard/ForecastsTable";
import Sidebar from "../components/Dashboard/Sidebar";
import Header from "../components/Dashboard/Header";
import { Card, CardContent } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";
import { Alert, AlertDescription } from "../components/ui/alert";
import { AlertCircle } from "lucide-react";

const processDashboardData = (data) => {
  if (!data || data.error) {
    console.log("Invalid or error data received:", data);
    return null;
  }

  console.log("Raw data received:", data);

  const processedData = {
    metrics: {
      mae: Number(data.metrics?.mae) || 0,
      mse: Number(data.metrics?.mse) || 0,
      accuracy: Number(data.metrics?.accuracy) || 0,
    },
    forecast: {
      labels: data.forecast?.labels || [],
      historicalData: (data.forecast?.historicalData || []).map(val => Number(val) || 0),
      historicalPredictions: (data.forecast?.historicalPredictions || []).map(val => Number(val) || 0),
      forecastData: (data.forecast?.forecastData || []).map(val => Number(val) || 0),
      confidence: (data.forecast?.confidence || []).map(val => Number(val) || 0),
      monthly: {
        labels: data.forecast?.monthly?.labels || [],
        historicalData: (data.forecast?.monthly?.historicalData || []).map(val => Number(val) || 0),
        forecastData: (data.forecast?.monthly?.forecastData || []).map(val => Number(val) || 0),
      },
    },
    tableData: (data.tableData || []).map(item => ({
      date: item.date || "",
      actualSales: Number(item.actualSales) || 0,
      predictedSales: Number(item.predictedSales) || 0,
      accuracy: Number(item.accuracy) || 0,
      error: Number(item.error) || 0,
    })),
    accuracyData: {
      labels: data.accuracyData?.labels || [],
      data: (data.accuracyData?.data || []).map(val => Number(val) || 0),
    },
    trendsData: {
      labels: data.trendsData?.labels || [],
      data: (data.trendsData?.data || []).map(val => Number(val) || 0),
    },
    modelMetrics: {
      mae: Number(data.modelMetrics?.mae) || 0,
      mse: Number(data.modelMetrics?.mse) || 0,
      featureImportance: (data.modelMetrics?.featureImportance || []).map(feature => ({
        name: feature.name || "",
        value: Number(feature.value) || 0
      })),
    },
    kpi: {
      forecastedRevenue: data.kpi?.forecastedRevenue || "$0",
      revenueChange: Number(data.kpi?.revenueChange) || 0,
      forecastAccuracy: data.kpi?.forecastAccuracy || "0%",
      accuracyChange: Number(data.kpi?.accuracyChange) || 0,
      grossRevenue: data.kpi?.grossRevenue || "$0",
      grossRevenueChange: Number(data.kpi?.grossRevenueChange) || 0,
      netRevenue: data.kpi?.netRevenue || "$0",
      netRevenueChange: Number(data.kpi?.netRevenueChange) || 0,
      longTermForecastedRevenue: data.kpi?.longTermForecastedRevenue || "$0",
      longTermRevenueChange: Number(data.kpi?.longTermRevenueChange) || 0,
    },
  };

  console.log("Processed dashboard data:", processedData);
  console.log("KPI data for KPICards:", processedData.kpi);

  return processedData;
};

export default function Dashboard() {
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1024);
  const { toast } = useToast();

  useEffect(() => {
    setDashboardData(null);
    setIsLoading(false);
    setIsGenerating(false);
    setError(null);
  }, []);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleUploadSuccess = async (data) => {
    try {
      console.log("Upload success data received:", data);

      if (data.isGenerating !== undefined) {
        setIsGenerating(data.isGenerating);
        return;
      }

      if (data.error) {
        throw new Error(data.error);
      }

      const processedData = processDashboardData(data);
      console.log("Processed upload data:", processedData);

      if (!processedData) {
        throw new Error("Invalid data format received from server");
      }

      setDashboardData(processedData);
      setError(null);

      toast({
        title: "Forecast Generated Successfully!",
        description: (
          <div className="mt-2 space-y-1">
            <p>• Accuracy: {processedData.metrics.accuracy.toFixed(1)}%</p>
            <p>• MAE: {processedData.metrics.mae.toFixed(2)}</p>
            <p>• MSE: {processedData.metrics.mse.toFixed(2)}</p>
          </div>
        ),
        variant: "default",
        className: "bg-green-50 border-green-200",
        duration: 5000,
      });
    } catch (err) {
      console.error("Error processing upload data:", err);
      setError(err.message);
      toast({
        title: "Error",
        description: `Failed to generate forecast: ${err.message}`,
        variant: "destructive",
        duration: 5000,
      });
    }
  };

  const handleUploadError = (error) => {
    setError(error);
    setIsGenerating(false);
    toast({
      title: "Error",
      description: error,
      variant: "destructive",
      duration: 5000,
    });
  };

  if (isLoading && !dashboardData) {
    return (
      <div className="min-h-screen bg-background">
        <div className="flex h-screen">
          <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
          <div className="flex-1 flex flex-col">
            <Header onMenuClick={() => setSidebarOpen(true)} />
            <main className="flex-1 p-4 md:p-6">
              <Skeleton className="h-8 w-[200px] mb-4" />
              <Skeleton className="h-[400px] w-full" />
            </main>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="flex h-screen">
        <Sidebar 
          isOpen={sidebarOpen} 
          isMobile={isMobile} 
          onClose={() => setSidebarOpen(false)}
          onDashboardClick={() => setSidebarOpen(false)}
          onForecastClick={() => setSidebarOpen(false)}
        />
        <div className={`flex-1 flex flex-col ${isMobile ? '' : 'ml-64'}`}>
          <Header onMenuClick={() => setSidebarOpen(true)} />
          <main className="flex-1 p-4 md:p-6 overflow-auto">
            <div className="max-w-7xl mx-auto space-y-4">
              <UploadForm 
                onUploadSuccess={handleUploadSuccess} 
                onUploadError={handleUploadError}
                isGenerating={isGenerating}
              />
              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              {dashboardData ? (
                <>
                  <KPICards data={dashboardData.kpi} isLoading={isGenerating} />
                  <div className="grid gap-4 md:grid-cols-2">
                    <SalesForecastChart data={dashboardData.forecast} isLoading={isGenerating} />
                    <AccuracyChart data={dashboardData.accuracyData} isLoading={isGenerating} />
                  </div>
                  <TrendsChart data={dashboardData.trendsData} isLoading={isGenerating} />
                  <ModelMetrics data={dashboardData.modelMetrics} isLoading={isGenerating} />
                  <ForecastsTable data={dashboardData.tableData} isLoading={isGenerating} />
                </>
              ) : (
                <Card>
                  <CardContent className="p-6 text-center">
                    <p className="text-muted-foreground">Upload a CSV file to generate forecasts.</p>
                  </CardContent>
                </Card>
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
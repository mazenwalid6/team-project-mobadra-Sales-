import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Skeleton } from "../ui/skeleton";
import Chart from "chart.js/auto";

export default function SalesForecastChart({ data, isLoading }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [period, setPeriod] = useState("all");
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log("SalesForecastChart received data:", data);
    
    if (!chartRef.current || isLoading || !data) {
      console.log("Chart not ready:", { 
        hasRef: !!chartRef.current, 
        isLoading, 
        hasData: !!data
      });
      return;
    }

    // Validate required data properties
    if (!data.labels || !data.historicalData || !data.forecastData || !data.historicalPredictions) {
      console.log("Missing required data properties:", {
        hasLabels: !!data.labels,
        hasHistorical: !!data.historicalData,
        hasForecast: !!data.forecastData,
        hasPredictions: !!data.historicalPredictions,
        dataKeys: Object.keys(data)
      });
      return;
    }

    try {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      // Filter data based on period
      let filteredLabels = data.labels;
      let filteredHistorical = data.historicalData;
      let filteredHistoricalPredictions = data.historicalPredictions;
      let filteredForecast = data.forecastData;

      if (period !== "all") {
        const months = period === "3m" ? 3 : period === "6m" ? 6 : 12;
        const cutoffDate = new Date(Math.max(...data.labels.map(d => new Date(d))));
        cutoffDate.setMonth(cutoffDate.getMonth() - months);
        const indices = data.labels
          .map((label, idx) => ({ label, idx }))
          .filter(({ label }) => new Date(label) >= cutoffDate)
          .map(({ idx }) => idx);
        filteredLabels = indices.map(i => data.labels[i]);
        filteredHistorical = indices.map(i => data.historicalData[i]);
        filteredHistoricalPredictions = indices.map(i => data.historicalPredictions[i]);
        filteredForecast = indices.map(i => data.forecastData[i]);
      }

      console.log("Filtered data lengths:", {
        labels: filteredLabels.length,
        historical: filteredHistorical.length,
        predictions: filteredHistoricalPredictions.length,
        forecast: filteredForecast.length
      });

      // Format labels
      const formattedLabels = filteredLabels.map((date) =>
        new Date(date).toLocaleDateString("en-US", { month: "short", year: "numeric" })
      );

      const ctx = chartRef.current.getContext("2d");
      if (!ctx) {
        console.error("Could not get canvas context");
        return;
      }

      chartInstance.current = new Chart(ctx, {
        type: "line",
        data: {
          labels: formattedLabels,
    datasets: [
      {
              label: "Actual Sales",
              data: filteredHistorical,
              borderColor: "#2563eb",
              backgroundColor: "rgba(37, 99, 235, 0.1)",
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.3,
              fill: true
            },
            {
              label: "Model Predictions",
              data: filteredHistoricalPredictions,
              borderColor: "#eab308",
              backgroundColor: "rgba(234, 179, 8, 0.1)",
        borderWidth: 2,
              pointRadius: 0,
        tension: 0.3,
              fill: true
            },
            {
              label: "Short-term Forecast",
              data: filteredForecast,
              borderColor: "#16a34a",
              backgroundColor: "rgba(22, 163, 74, 0.1)",
        borderWidth: 2,
              pointRadius: 0,
        tension: 0.3,
        fill: true,
              borderDash: [5, 5]
            }
          ]
      },
        options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
              labels: { color: "rgb(243, 244, 246)" }
      },
      tooltip: {
        mode: "index",
        intersect: false,
        backgroundColor: "rgb(31, 41, 55)",
        titleColor: "rgb(243, 244, 246)",
        bodyColor: "rgb(243, 244, 246)",
        borderColor: "rgb(75, 85, 99)",
        borderWidth: 1,
        callbacks: {
          label: function (context) {
            const value = context.parsed.y || 0;
            return `${context.dataset.label}: $${value.toLocaleString()}`;
                }
              }
            }
    },
    scales: {
      x: {
        grid: { color: "rgba(75, 85, 99, 0.3)" },
              ticks: { color: "rgb(156, 163, 175)" }
      },
      y: {
        grid: { color: "rgba(75, 85, 99, 0.3)" },
        ticks: {
          color: "rgb(156, 163, 175)",
          callback: function (value) {
                  if (value >= 1000000) {
                    return `$${(value / 1000000).toFixed(1)}M`;
                  } else if (value >= 1000) {
                    return `$${(value / 1000).toFixed(0)}K`;
                  }
                  return `$${value}`;
                }
              }
            }
          },
          animation: {
            duration: 1000,
            easing: 'easeInOutQuart'
          }
        }
      });

      console.log("Chart created successfully");
    } catch (error) {
      console.error("Error creating chart:", error);
      setError(error.message);
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [period, data, isLoading]);

  if (error) {
    return (
      <Card className="h-full border border-border bg-card shadow-none animate-fade-in">
        <CardContent className="p-6">
          <div className="text-red-500">Error: {error}</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full border border-border bg-card shadow-none animate-fade-in">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Sales Forecast</h3>
            <p className="text-xs text-muted-foreground">Historical and forecasted sales trends</p>
          </div>
          <div className="flex items-center space-x-2">
            <Select value={period} onValueChange={(value) => setPeriod(value)}>
              <SelectTrigger
                className="w-[140px] h-8 text-xs bg-background border border-border text-foreground"
                aria-label="Select time period"
              >
                <SelectValue placeholder="Select period" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Time</SelectItem>
                <SelectItem value="3m">Last 3 Months</SelectItem>
                <SelectItem value="6m">Last 6 Months</SelectItem>
                <SelectItem value="12m">Last 12 Months</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="h-[350px] w-full relative">
          {isLoading || !data?.labels?.length || !data?.historicalData?.length || !data?.forecastData?.length ? (
            <Skeleton className="h-full w-full" />
          ) : (
            <canvas ref={chartRef} aria-label="Sales forecast chart" />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
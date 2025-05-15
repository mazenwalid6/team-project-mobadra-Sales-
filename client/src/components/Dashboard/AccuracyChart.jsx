import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent } from "../ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Skeleton } from "../ui/skeleton";
import Chart from "chart.js/auto";

export default function AccuracyChart({ data, isLoading }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [period, setPeriod] = useState("all");

  useEffect(() => {
    if (!chartRef.current || isLoading || !data?.labels?.length || !data?.data?.length) {
      return;
    }

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Filter data based on period
    let filteredLabels = data.labels;
    let filteredData = data.data;

    if (period !== "all") {
      const months = period === "3m" ? 3 : period === "6m" ? 6 : 12;
      const cutoffDate = new Date(Math.max(...data.labels.map(d => new Date(d))));
      cutoffDate.setMonth(cutoffDate.getMonth() - months);
      const indices = data.labels
        .map((label, idx) => ({ label, idx }))
        .filter(({ label }) => new Date(label) >= cutoffDate)
        .map(({ idx }) => idx);
      filteredLabels = indices.map(i => data.labels[i]);
      filteredData = indices.map(i => data.data[i]);
    }

    // Format labels
    const formattedLabels = filteredLabels.map((date) =>
      new Date(date).toLocaleDateString("en-US", { month: "short", year: "numeric" })
    );

    const ctx = chartRef.current.getContext("2d");
    if (!ctx) return;

    chartInstance.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: formattedLabels,
        datasets: [
          {
            label: "Forecast Accuracy",
            data: filteredData,
            borderColor: "#22c55e",
            backgroundColor: "rgba(34, 197, 94, 0.1)",
            borderWidth: 2,
            pointBackgroundColor: "#22c55e",
            tension: 0.3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "rgb(243, 244, 246)" },
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
                return `Accuracy: ${value.toFixed(1)}%`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(75, 85, 99, 0.3)" },
            ticks: { color: "rgb(156, 163, 175)" },
          },
          y: {
            grid: { color: "rgba(75, 85, 99, 0.3)" },
            ticks: {
              color: "rgb(156, 163, 175)",
              callback: function (value) {
                return value + "%";
              },
            },
            min: 0,
            max: 100,
          },
        },
      },
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [period, data, isLoading]);

  return (
    <Card className="h-full border border-border bg-card shadow-none animate-fade-in">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Forecast Accuracy</h3>
            <p className="text-xs text-muted-foreground">Trend of prediction accuracy over time</p>
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
          {isLoading || !data?.labels?.length || !data?.data?.length ? (
            <Skeleton className="h-full w-full" />
          ) : (
            <canvas ref={chartRef} aria-label="Accuracy chart" />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
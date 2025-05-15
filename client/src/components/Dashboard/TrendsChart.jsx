import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Line } from "react-chartjs-2";
import { Skeleton } from "../ui/skeleton";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function TrendsChart({ data, isLoading }) {
  if (isLoading || !data?.labels || !data?.data) {
    return (
      <Card className="border border-border bg-card shadow-sm">
        <CardContent className="p-6">
          <div className="animate-pulse">
            <Skeleton className="h-4 w-1/4 mb-4" />
            <Skeleton className="h-64 w-full rounded" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const chartData = {
    labels: data.labels.map(date => new Date(date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })),
    datasets: [
      {
        label: "Sales Trend",
        data: data.data,
        borderColor: "#8b5cf6",
        backgroundColor: "rgba(139, 92, 246, 0.2)",
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: "#8b5cf6",
        tension: 0.3,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "rgb(255, 255, 255)",
          font: { size: 14 },
          padding: 20,
        },
      },
      title: {
        display: true,
        text: "Sales Trend (Historical + Forecast)",
        color: "rgb(255, 255, 255)",
        font: { size: 18, weight: "bold" },
        padding: { top: 10, bottom: 20 },
      },
      tooltip: {
        mode: "index",
        intersect: false,
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleColor: "rgb(255, 255, 255)",
        bodyColor: "rgb(255, 255, 255)",
        borderColor: "rgb(107, 114, 128)",
        borderWidth: 1,
        callbacks: {
          label: function (context) {
            const value = context.parsed.y || 0;
            return `Sales: $${value.toLocaleString()}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(107, 114, 128, 0.2)" },
        ticks: {
          color: "rgb(209, 213, 219)",
          maxRotation: 45,
          minRotation: 45,
          font: { size: 12 },
        },
      },
      y: {
        grid: { color: "rgba(107, 114, 128, 0.2)" },
        ticks: {
          color: "rgb(209, 213, 219)",
          font: { size: 12 },
          callback: function (value) {
            return "$" + value.toLocaleString();
          },
        },
        beginAtZero: true,
      },
    },
  };

  return (
    <Card className="border border-border bg-card shadow-sm">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-foreground">Sales Trend</CardTitle>
      </CardHeader>
      <CardContent className="h-[400px] p-4">
        <Line data={chartData} options={options} />
      </CardContent>
    </Card>
  );
}
import React from "react";
import { Card, CardContent } from "../ui/card";
import { Skeleton } from "../ui/skeleton";
import { TrendingUp, Target, DollarSign, Wallet } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@radix-ui/react-tooltip";

export default function KPICards({ data, isLoading }) {
  // Debug log to verify incoming data
  console.log("KPICards received data:", data);
  console.log("KPICards isLoading:", isLoading);

  const kpiData = [
    {
      title: "Sales",
      value: data?.forecastedRevenue || "$0",
      icon: <TrendingUp className="h-5 w-5 text-primary" />,
      tooltip: "Total predicted sales revenue for the next 12 weeks based on the forecast model."
    },
    {
      title: "Forecast Accuracy",
      value: data?.forecastAccuracy || "0%",
      icon: <Target className="h-5 w-5 text-primary" />,
      tooltip: "Percentage showing how closely the model's predictions match actual historical sales."
    },
    {
      title: "Gross Revenue",
      value: data?.grossRevenue || "$0",
      icon: <DollarSign className="h-5 w-5 text-primary" />,
      tooltip: "Total sales revenue from historical data before any deductions (e.g., returns)."
    },
    {
      title: "Net Revenue",
      value: data?.netRevenue || "$0",
      icon: <Wallet className="h-5 w-5 text-primary" />,
      tooltip: "Total sales revenue after applying deductions, such as returns or discounts."
    },
  ];

  return (
    <TooltipProvider>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpiData.map((kpi, index) => (
          <Tooltip key={index}>
            <TooltipTrigger asChild>
              <Card className="border border-border bg-card animate-fade-in hover:shadow-md transition-shadow">
                <CardContent className="p-4">
                  {isLoading ? (
                    <Skeleton className="h-24 w-full" />
                  ) : (
                    <div className="flex items-center space-x-4">
                      <div className="p-2 rounded-full bg-primary/10">{kpi.icon}</div>
                      <div>
                        <p className="text-xs text-muted-foreground">{kpi.title}</p>
                        <h3 className="text-lg font-semibold text-foreground">{kpi.value}</h3>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TooltipTrigger>
            <TooltipContent className="bg-background border border-border p-2 rounded-md max-w-xs">
              <p className="text-sm text-foreground">{kpi.tooltip}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  );
}
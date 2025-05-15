import React from "react";
import { Card, CardContent } from "../ui/card";
import { Skeleton } from "../ui/skeleton";

export default function ModelMetrics({ data, isLoading }) {
  console.log("ModelMetrics received data:", data); // Debug log

  // Ensure we have valid data
  const mae = data?.mae || 0;
  const mse = data?.mse || 0;
  const featureImportance = data?.featureImportance || [
    { name: "Store", value: 0 },
    { name: "Dept", value: 0 },
    { name: "IsHoliday", value: 0 },
    { name: "Type", value: 0 },
    { name: "Size", value: 0 },
  ];

  // Sort feature importance by value
  const sortedFeatures = [...featureImportance].sort((a, b) => b.value - a.value);

  return (
    <Card className="border border-border bg-card shadow-md hover:border-secondary/30 transition-colors duration-200 animate-fade-in">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Model Performance</h3>
            <p className="text-xs text-muted-foreground">Accuracy metrics & feature importance</p>
          </div>
          <div
            className="h-8 w-8 rounded-full bg-black/60 flex items-center justify-center border border-border cursor-pointer shadow-inner"
            aria-label="Settings"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 text-muted-foreground"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="1" />
              <circle cx="19" cy="12" r="1" />
              <circle cx="5" cy="12" r="1" />
            </svg>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-black/40 border border-border rounded-lg p-4 shadow-md">
            <div className="flex flex-col">
              <span className="text-xs text-muted-foreground mb-2">Mean Absolute Error</span>
              {isLoading ? (
                <Skeleton className="h-7 w-24" />
              ) : (
                <span className="text-xl font-bold text-foreground">
                  {mae.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </span>
              )}
            </div>
          </div>
          <div className="bg-black/40 border border-border rounded-lg p-4 shadow-md">
            <div className="flex flex-col">
              <span className="text-xs text-muted-foreground mb-2">Mean Squared Error</span>
              {isLoading ? (
                <Skeleton className="h-7 w-24" />
              ) : (
                <span className="text-xl font-bold text-foreground">
                  {mse.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </span>
              )}
            </div>
          </div>
        </div>
        <div>
          <h4 className="text-sm font-medium text-foreground mb-4">Feature Importance</h4>
          <div className="space-y-4">
            {isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : (
              sortedFeatures.map((feature, index) => (
                <div key={index} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-medium text-foreground">{feature.name}</div>
                    <div className="text-xs font-medium text-secondary glow-text-subtle">
                      {(feature.value * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="relative">
                    <div className="overflow-hidden h-1.5 text-xs flex rounded-sm bg-black/60 border border-border shadow-inner">
                      <div
                        style={{ width: `${feature.value * 100}%` }}
                        className="shadow-sm flex flex-col text-center whitespace-nowrap justify-center bg-secondary glow-bar-subtle"
                      ></div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
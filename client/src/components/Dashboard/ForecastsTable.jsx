import React, { useState, useEffect, useMemo } from "react";
import { Card, CardContent } from "../ui/card";
import { Skeleton } from "../ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
  PaginationEllipsis,
} from "../ui/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Badge } from "../ui/badge";

export default function ForecastsTable({ data = [], isLoading, searchTerm = "" }) {
  console.log("ForecastsTable received data:", data); // Debug log

  const [currentPage, setCurrentPage] = useState(1);
  const [filter, setFilter] = useState("all");
  const pageSize = 5;

  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm]);

  const filteredData = useMemo(() => {
    let filtered = data;
    if (filter !== "all") {
      filtered = data.filter((forecast) => {
        if (filter === "high") return forecast.accuracy > 90;
        if (filter === "medium") return forecast.accuracy >= 70 && forecast.accuracy <= 90;
        if (filter === "low") return forecast.accuracy < 70;
        return true;
      });
    }
    return filtered;
  }, [data, filter]);

  const paginatedForecasts = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    return filteredData.slice(startIndex, startIndex + pageSize);
  }, [filteredData, currentPage]);

  const filteredForecasts = useMemo(() => {
    if (!searchTerm.trim()) return paginatedForecasts;
    const searchTermLower = searchTerm.toLowerCase();
    return paginatedForecasts.filter(
      (forecast) =>
        forecast.date.toLowerCase().includes(searchTermLower) ||
        forecast.actualSales.toString().includes(searchTermLower) ||
        forecast.predictedSales.toString().includes(searchTermLower) ||
        forecast.accuracy.toString().includes(searchTermLower)
    );
  }, [paginatedForecasts, searchTerm]);

  const totalForecasts = searchTerm.trim() ? filteredForecasts.length : filteredData.length;
  const totalPages = Math.ceil(totalForecasts / pageSize);

  const getPageNumbers = () => {
    const pages = [];
    const maxDisplayedPages = 5;
    if (totalPages <= maxDisplayedPages) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      pages.push(1);
      let startPage = Math.max(2, currentPage - 1);
      let endPage = Math.min(totalPages - 1, currentPage + 1);
      if (currentPage <= 2) {
        endPage = Math.min(totalPages - 1, 4);
      }
      if (currentPage >= totalPages - 1) {
        startPage = Math.max(2, totalPages - 3);
      }
      if (startPage > 2) {
        pages.push("ellipsis1");
      }
      for (let i = startPage; i <= endPage; i++) {
        pages.push(i);
      }
      if (endPage < totalPages - 1) {
        pages.push("ellipsis2");
      }
      if (totalPages > 1) {
        pages.push(totalPages);
      }
    }
    return pages;
  };

  const formatCurrency = (value) => {
    const numValue = typeof value === "number" ? value : parseFloat(value);
    return `$${numValue.toLocaleString()}`;
  };

  const getAccuracyClass = (accuracy) => {
    const accuracyNum = typeof accuracy === "number" ? accuracy : parseFloat(accuracy);
    if (accuracyNum >= 90) return "bg-success/10 text-success";
    if (accuracyNum >= 70) return "bg-warning/10 text-warning";
    return "bg-destructive/10 text-destructive";
  };

  return (
    <Card className="border border-border bg-card shadow-none animate-fade-in">
      <CardContent className="p-0">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex space-x-2 items-center">
            <h3 className="text-base font-semibold text-foreground">Forecast History</h3>
          </div>
          <div className="flex items-center space-x-2">
            <Select value={filter} onValueChange={setFilter}>
              <SelectTrigger
                className="w-[140px] h-8 text-xs bg-background border border-border text-foreground"
                aria-label="Filter forecasts"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All forecasts</SelectItem>
                <SelectItem value="high">High accuracy (&gt;90%)</SelectItem>
                <SelectItem value="medium">Medium (70-90%)</SelectItem>
                <SelectItem value="low">Low (&lt;70%)</SelectItem>
              </SelectContent>
            </Select>
            <div
              className="h-8 w-8 rounded-full bg-card flex items-center justify-center border border-border cursor-pointer"
              aria-label="Table settings"
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
        </div>
        <div className="overflow-x-auto">
          <Table aria-label="Forecasts table">
            <TableHeader>
              <TableRow className="border-b-0 hover:bg-transparent">
                <TableHead className="text-xs text-muted-foreground font-normal">Date</TableHead>
                <TableHead className="text-xs text-muted-foreground font-normal">Actual Sales</TableHead>
                <TableHead className="text-xs text-muted-foreground font-normal">Predicted Sales</TableHead>
                <TableHead className="text-xs text-muted-foreground font-normal">Accuracy</TableHead>
                <TableHead className="text-xs text-muted-foreground font-normal">Error</TableHead>
                <TableHead className="text-xs text-muted-foreground font-normal text-right">
                  Actions
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8">
                    <Skeleton className="h-8 w-full" />
                  </TableCell>
                </TableRow>
              ) : filteredForecasts.length === 0 && searchTerm ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    No matching forecasts found for "
                    <span className="font-medium text-foreground">{searchTerm}</span>"
                  </TableCell>
                </TableRow>
              ) : filteredForecasts.length > 0 ? (
                filteredForecasts.map((item, index) => (
                  <TableRow key={`${item.date}-${index}`} className="hover:bg-background/50">
                    <TableCell className="text-xs py-3">{item.date}</TableCell>
                    <TableCell className="text-xs font-medium py-3">
                      {formatCurrency(item.actualSales)}
                    </TableCell>
                    <TableCell className="text-xs py-3">{formatCurrency(item.predictedSales)}</TableCell>
                    <TableCell className="py-3">
                      <span
                        className={`px-2 py-1 inline-flex text-xs leading-none font-medium rounded-sm ${getAccuracyClass(
                          item.accuracy
                        )}`}
                      >
                        {(typeof item.accuracy === "number"
                          ? item.accuracy
                          : parseFloat(item.accuracy)
                        ).toFixed(1)}
                        %
                      </span>
                    </TableCell>
                    <TableCell className="text-xs py-3">{formatCurrency(item.error)}</TableCell>
                    <TableCell className="text-right py-3">
                      <button
                        type="button"
                        className="text-xs text-secondary hover:text-secondary/80"
                        aria-label={`View details for forecast on ${item.date}`}
                      >
                        View Details
                      </button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    {searchTerm ? (
                      <>
                        No matching forecasts found for "
                        <span className="font-medium text-foreground">{searchTerm}</span>"
                      </>
                    ) : (
                      "No forecast data available. Upload data to generate forecasts."
                    )}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        <div className="flex items-center justify-between px-6 py-4 border-t border-border">
          <div className="text-xs text-muted-foreground">
            Showing <span className="font-medium">{filteredForecasts.length}</span> of{" "}
            <span className="font-medium">{totalForecasts}</span> records
            {searchTerm && (
              <Badge variant="outline" className="ml-2 text-[10px] bg-black/20">
                Search: {searchTerm}
              </Badge>
            )}
          </div>
          {totalPages > 1 && (
            <Pagination aria-label="Table pagination">
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    href="#"
                    onClick={(e) => {
                      e.preventDefault();
                      if (currentPage > 1) setCurrentPage(currentPage - 1);
                    }}
                    aria-disabled={currentPage === 1}
                    className={`${
                      currentPage === 1 ? "pointer-events-none opacity-50" : ""
                    } text-xs h-8 w-8 p-0 flex items-center justify-center`}
                  />
                </PaginationItem>
                {getPageNumbers().map((page, i) => {
                  if (page === "ellipsis1" || page === "ellipsis2") {
                    return (
                      <PaginationItem key={`ellipsis-${i}`}>
                        <PaginationEllipsis />
                      </PaginationItem>
                    );
                  }
                  return (
                    <PaginationItem key={page}>
                      <PaginationLink
                        href="#"
                        isActive={currentPage === page}
                        onClick={(e) => {
                          e.preventDefault();
                          setCurrentPage(Number(page));
                        }}
                        className="text-xs h-8 w-8 p-0 flex items-center justify-center"
                        aria-label={`Go to page ${page}`}
                      >
                        {page}
                      </PaginationLink>
                    </PaginationItem>
                  );
                })}
                <PaginationItem>
                  <PaginationNext
                    href="#"
                    onClick={(e) => {
                      e.preventDefault();
                      if (currentPage < totalPages) setCurrentPage(currentPage + 1);
                    }}
                    aria-disabled={currentPage === totalPages}
                    className={`${
                      currentPage === totalPages ? "pointer-events-none opacity-50" : ""
                    } text-xs h-8 w-8 p-0 flex items-center justify-center`}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
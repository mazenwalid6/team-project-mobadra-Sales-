import React from "react";
import { ChevronLeft, ChevronRight, MoreHorizontal } from "lucide-react";
import { cn } from "../../lib/utils";

const Pagination = React.forwardRef(({ className, ...props }, ref) => (
  <nav
    ref={ref}
    role="navigation"
    aria-label="pagination"
    className={cn("mx-auto flex w-full justify-center", className)}
    {...props}
  />
));
Pagination.displayName = "Pagination";

const PaginationContent = React.forwardRef(({ className, ...props }, ref) => (
  <ul
    ref={ref}
    className={cn("flex flex-row items-center gap-1", className)}
    {...props}
  />
));
PaginationContent.displayName = "PaginationContent";

const PaginationItem = React.forwardRef(({ className, ...props }, ref) => (
  <li ref={ref} className={cn("", className)} {...props} />
));
PaginationItem.displayName = "PaginationItem";

const PaginationLink = React.forwardRef(({
  className,
  isActive,
  isPrevious,
  isNext,
  isEllipsis,
  size = "default",
  ...props
}, ref) => {
  const baseCls = "relative inline-flex items-center px-4 py-2 border border-border text-sm font-medium";
  
  return (
    <a 
      ref={ref}
      className={cn(
        baseCls,
        isActive && "bg-secondary text-foreground font-bold border-primary",
        isEllipsis && "bg-muted text-muted-foreground",
        (isPrevious || isNext) && "px-2 rounded-md bg-muted text-muted-foreground hover:bg-muted/75",
        !isActive && !isEllipsis && !isPrevious && !isNext && "bg-muted text-muted-foreground hover:bg-muted/75",
        isPrevious && "rounded-l-md",
        isNext && "rounded-r-md",
        className
      )}
      {...props}
    >
      {isEllipsis ? "..." : isPrevious ? <ChevronLeft className="h-4 w-4" /> : isNext ? <ChevronRight className="h-4 w-4" /> : props.children}
    </a>
  );
});
PaginationLink.displayName = "PaginationLink";

const PaginationPrevious = React.forwardRef(({ className, ...props }, ref) => (
  <PaginationLink
    ref={ref}
    aria-label="Go to previous page"
    size="default"
    className={cn("", className)}
    isPrevious
    {...props}
  />
));
PaginationPrevious.displayName = "PaginationPrevious";

const PaginationNext = React.forwardRef(({ className, ...props }, ref) => (
  <PaginationLink
    ref={ref}
    aria-label="Go to next page"
    size="default"
    className={cn("", className)}
    isNext
    {...props}
  />
));
PaginationNext.displayName = "PaginationNext";

const PaginationEllipsis = React.forwardRef(({ className, ...props }, ref) => (
  <PaginationLink
    ref={ref}
    aria-hidden
    isEllipsis
    className={cn("", className)}
    {...props}
  />
));
PaginationEllipsis.displayName = "PaginationEllipsis";

export {
  Pagination,
  PaginationContent,
  PaginationLink,
  PaginationItem,
  PaginationPrevious,
  PaginationNext,
  PaginationEllipsis,
};
import * as React from "react"
import PropTypes from 'prop-types'
import { cn } from "../../lib/utils"

const Table = React.forwardRef(({ className, ...props }, ref) => (
  <div className="relative w-full overflow-auto">
    <table
      ref={ref}
      className={cn("w-full caption-bottom text-sm", className)}
      {...props}
    />
  </div>
))
Table.displayName = "Table"
Table.propTypes = {
  className: PropTypes.string
}

const TableHeader = React.forwardRef(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("bg-muted", className)} {...props} />
))
TableHeader.displayName = "TableHeader"
TableHeader.propTypes = {
  className: PropTypes.string
}

const TableBody = React.forwardRef(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("[&_tr:last-child]:border-0", className)}
    {...props}
  />
))
TableBody.displayName = "TableBody"
TableBody.propTypes = {
  className: PropTypes.string
}

const TableFooter = React.forwardRef(({ className, ...props }, ref) => (
  <tfoot
    ref={ref}
    className={cn(
      "border-t bg-muted/50 font-medium [&>tr]:last:border-b-0",
      className
    )}
    {...props}
  />
))
TableFooter.displayName = "TableFooter"
TableFooter.propTypes = {
  className: PropTypes.string
}

const TableRow = React.forwardRef(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "border-b border-border transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted",
      className
    )}
    {...props}
  />
))
TableRow.displayName = "TableRow"
TableRow.propTypes = {
  className: PropTypes.string
}

const TableHead = React.forwardRef(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider",
      className
    )}
    {...props}
  />
))
TableHead.displayName = "TableHead"
TableHead.propTypes = {
  className: PropTypes.string
}

const TableCell = React.forwardRef(({ className, ...props }, ref) => (
  <td
    ref={ref}
    className={cn("px-6 py-4 whitespace-nowrap text-sm text-foreground", className)}
    {...props}
  />
))
TableCell.displayName = "TableCell"
TableCell.propTypes = {
  className: PropTypes.string
}

const TableCaption = React.forwardRef(({ className, ...props }, ref) => (
  <caption
    ref={ref}
    className={cn("mt-4 text-sm text-muted-foreground", className)}
    {...props}
  />
))
TableCaption.displayName = "TableCaption"
TableCaption.propTypes = {
  className: PropTypes.string
}

export {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableHead,
  TableRow,
  TableCell,
  TableCaption
}

import React from 'react';
import PropTypes from 'prop-types';
import { cn } from "../../lib/utils";

const Skeleton = React.forwardRef(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  );
});

Skeleton.displayName = "Skeleton";

Skeleton.propTypes = {
  className: PropTypes.string
};

export { Skeleton }; 
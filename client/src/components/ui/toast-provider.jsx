import React from "react";
import { Toast } from "./toast";
import { useToast } from "../../hooks/use-toast";

export function ToastProvider({ children }) {
  const { toasts, removeToast } = useToast();

  return (
    <>
      {children}
      <div className="fixed bottom-0 right-0 z-50 flex flex-col gap-2 p-4">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            title={toast.title}
            description={toast.description}
            variant={toast.variant}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </>
  );
} 
import { cn } from "@/lib/utils";

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

export const LoadingSpinner = ({ size = "md", className }: LoadingSpinnerProps) => {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-8 h-8", 
    lg: "w-12 h-12"
  };

  return (
    <div className={cn("flex items-center justify-center", className)}>
      <div 
        className={cn(
          "animate-spin rounded-full border-2 border-muted-foreground border-t-primary",
          sizeClasses[size]
        )}
      />
    </div>
  );
};

export const LoadingCard = ({ title = "Loading..." }: { title?: string }) => {
  return (
    <div className="bg-card rounded-lg shadow-card p-6 animate-pulse">
      <div className="flex items-center space-x-4">
        <LoadingSpinner size="md" />
        <div className="space-y-2">
          <div className="h-4 bg-muted rounded w-32"></div>
          <div className="h-3 bg-muted rounded w-24"></div>
        </div>
      </div>
      <div className="mt-4 space-y-3">
        <div className="h-3 bg-muted rounded w-full"></div>
        <div className="h-3 bg-muted rounded w-5/6"></div>
        <div className="h-3 bg-muted rounded w-4/6"></div>
      </div>
    </div>
  );
};
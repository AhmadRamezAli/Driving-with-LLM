import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Clipboard, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PredictionLogsTable } from "./PredictionLogsTable";
import { PredictionLogDetails } from "./PredictionLogDetails";
import { PredictionLog } from "@/types/prediction";

export function PredictionLogsContainer() {
  const [selectedLog, setSelectedLog] = useState<PredictionLog | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleViewDetails = (log: PredictionLog) => {
    setSelectedLog(log);
    setDetailsOpen(true);
  };

  const handleCloseDetails = () => {
    setDetailsOpen(false);
  };

  const handleRefresh = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  // Force refresh of child components when refreshTrigger changes
  const key = `prediction-logs-${refreshTrigger}`;

  return (
    <div className="space-y-6">
      <Card className="bg-card shadow-card">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Clipboard className="w-5 h-5 text-primary" />
              Prediction Logs
            </CardTitle>
            <CardDescription>
              Historical records of AI predictions and scene data
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </CardHeader>
        <CardContent>
          <PredictionLogsTable key={key} onViewDetails={handleViewDetails} />
        </CardContent>
      </Card>

      <PredictionLogDetails
        log={selectedLog}
        open={detailsOpen}
        onClose={handleCloseDetails}
      />
    </div>
  );
} 
import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Code, Info, Map, MessageSquare } from "lucide-react";
import { PredictionLog } from "@/types/prediction";
import { SceneMap } from "@/components/SceneMap";

type PredictionLogDetailsProps = {
  log: PredictionLog | null;
  open: boolean;
  onClose: () => void;
};

export function PredictionLogDetails({
  log,
  open,
  onClose,
}: PredictionLogDetailsProps) {
  const [activeTab, setActiveTab] = useState("summary");

  if (!log) {
    return null;
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl">Prediction Log Details</DialogTitle>
          <DialogDescription>
            Logged at {formatDate(log.timestamp)}
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-4">
          <TabsList className="grid grid-cols-4">
            <TabsTrigger value="summary" className="flex items-center gap-2">
              <Info className="w-4 h-4" />
              Summary
            </TabsTrigger>
            <TabsTrigger value="prediction" className="flex items-center gap-2">
              <MessageSquare className="w-4 h-4" />
              Prediction
            </TabsTrigger>
            <TabsTrigger value="scene" className="flex items-center gap-2">
              <Code className="w-4 h-4" />
              Request Data
            </TabsTrigger>
            <TabsTrigger value="map" className="flex items-center gap-2">
              <Map className="w-4 h-4" />
              Map View
            </TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="py-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-lg mb-2">Prediction Summary</h3>
                <div className="space-y-2">
                  <div>
                    <span className="font-medium">Caption:</span> {log.caption || "N/A"}
                  </div>
                  <div>
                    <span className="font-medium">Timestamp:</span> {formatDate(log.timestamp)}
                  </div>
                  <div>
                    <span className="font-medium">Time Taken:</span>{" "}
                    {log.time_taken !== undefined 
                      ? (log.time_taken * 1000).toFixed(2) 
                      : "N/A"} ms
                  </div>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-2">Control Values</h3>
                <div className="space-y-2">
                  <div>
                    <span className="font-medium">Accelerate:</span>{" "}
                    {log.accelerate !== undefined 
                      ? log.accelerate.toFixed(4) 
                      : "N/A"}
                  </div>
                  <div>
                    <span className="font-medium">Brake:</span>{" "}
                    {log.brake !== undefined 
                      ? log.brake.toFixed(4) 
                      : "N/A"}
                  </div>
                  <div>
                    <span className="font-medium">Steering:</span>{" "}
                    {log.steering !== undefined 
                      ? log.steering.toFixed(4) 
                      : "N/A"}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="prediction" className="py-4">
            <div className="rounded-md bg-slate-950 p-4 overflow-auto max-h-[100vw]">
              <pre className="text-sm text-slate-50 font-mono max-w-[50vh]">
                {JSON.stringify({
                  caption: log.caption,
                  accelerate: log.accelerate,
                  brake: log.brake,
                  steering: log.steering,
                  time_taken: log.time_taken
                }, null, 2)}
              </pre>
            </div>
          </TabsContent>

          <TabsContent value="scene" className="py-4">
            <div className="flex justify-end mb-4">
              <Button 
                variant="outline" 
                size="sm" 
                className="flex items-center gap-2"
                onClick={() => setActiveTab("map")}
              >
                <Map className="w-4 h-4" />
                Show on Map
              </Button>
            </div>
            <div className="rounded-md bg-slate-950 p-4 overflow-auto max-h-[50vh] max-w-[100vw]">
              <pre className="text-sm text-slate-50 font-mono">
                {JSON.stringify(log.request || {}, null, 2)}
              </pre>
            </div>
          </TabsContent>

          <TabsContent value="map" className="py-4">
            <SceneMap scene={log.request} />
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
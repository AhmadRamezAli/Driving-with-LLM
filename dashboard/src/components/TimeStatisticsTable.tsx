import { useState, useEffect } from "react";
import { apiClient, TimeStatistic, TimeStatisticsParams } from "@/lib/api";
import { LoadingSpinner, LoadingCard } from "@/components/LoadingSpinner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar as CalendarIcon, Clock, RefreshCw } from "lucide-react";
import { DateTimePicker } from "@/components/ui/date-time-picker";

const getDefaultTimestamps = () => {
  const today = new Date();
  const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const endOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate(), 23, 59, 59, 999);
  
  return {
    from_timestamp: startOfDay.toISOString(),
    to_timestamp: endOfDay.toISOString()
  };
};

export const TimeStatisticsTable = () => {
  const [data, setData] = useState<TimeStatistic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<TimeStatisticsParams>({
    limit: 100,
    skip: 0,
    ...getDefaultTimestamps()
  });

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiClient.fetchTimeStatistics(params);
      setData(response.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRefresh = () => {
    fetchData();
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatTimeTaken = (seconds: number) => {
    return `${seconds.toFixed(3)}s`;
  };

  // Transform data for chart
  const chartData = data.map((stat) => ({
    time_taken: stat.time_taken,
    timestamp: new Date(stat.timestamp).getTime(),
    displayTime: new Date(stat.timestamp).toLocaleTimeString(),
    fullTimestamp: stat.timestamp
  }));

  const chartConfig = {
    time_taken: {
      label: "Time Taken (seconds)",
      color: "hsl(var(--primary))",
    },
  };

  if (loading) {
    return <LoadingCard title="Loading time statistics..." />;
  }

  if (error) {
    return (
      <Card className="bg-card shadow-card">
        <CardHeader>
          <CardTitle className="text-destructive">Error</CardTitle>
          <CardDescription>{error}</CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={handleRefresh} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-card shadow-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            Prediction Time Statistics
          </CardTitle>
          <CardDescription>
            Execution time data for prediction requests ({data.length} records)
          </CardDescription>
        </div>
        <Button onClick={handleRefresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex gap-4">
          <div className="flex-1">
            <label className="text-sm font-medium mb-2 block">From Timestamp</label>
            <DateTimePicker
              value={params.from_timestamp || ''}
              onChange={(date) => setParams(prev => ({ 
                ...prev, 
                from_timestamp: date ? date.toISOString() : undefined 
              }))}
            />
          </div>
          <div className="flex-1">
            <label className="text-sm font-medium mb-2 block">To Timestamp</label>
            <DateTimePicker
              value={params.to_timestamp || ''}
              onChange={(date) => setParams(prev => ({ 
                ...prev, 
                to_timestamp: date ? date.toISOString() : undefined 
              }))}
            />
          </div>
          <div className="flex items-end">
            <Button onClick={fetchData} className="bg-gradient-primary">
              <CalendarIcon className="w-4 h-4 mr-2" />
              Filter
            </Button>
          </div>
        </div>

        <div className="w-full h-[300px] sm:h-[400px] lg:h-[500px]">
          <ChartContainer config={chartConfig} className="w-full h-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="timestamp"
                  type="number"
                  scale="time"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  label={{ value: 'Timestamp', position: 'insideBottom', offset: -5 }}
                  className="text-muted-foreground"
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  label={{ value: 'Time Taken (seconds)', angle: -90, position: 'insideLeft' }}
                  className="text-muted-foreground"
                />
                <ChartTooltip 
                  content={<ChartTooltipContent 
                    formatter={(value, name) => [
                      `${Number(value).toFixed(3)}s`,
                      'Time Taken'
                    ]}
                    labelFormatter={(label) => new Date(Number(label)).toLocaleString()}
                  />} 
                />
                <Line 
                  type="monotone" 
                  dataKey="time_taken" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={2}
                  dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 3 }}
                  activeDot={{ r: 5, fill: "hsl(var(--primary))" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
};

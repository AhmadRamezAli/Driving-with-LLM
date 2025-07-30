import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, ScatterChart, Scatter,
  Cell, ComposedChart, Legend
} from "recharts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar as CalendarIcon, Database, RefreshCw, AlertCircle } from "lucide-react";
import { LoadingCard } from "@/components/LoadingSpinner";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

interface SystemHealth {
  total_entries: number;
  database_size_mb: number;
  entries_over_time: Record<string, number>;
}

interface ModelPerformance {
  time_taken_distribution: Record<string, number>;
  accelerate_distribution: {
    min: number;
    max: number;
    avg: number;
    stdDev: number;
  };
  brake_distribution: {
    min: number;
    max: number;
    avg: number;
    stdDev: number;
  };
  steering_distribution: {
    min: number;
    max: number;
    avg: number;
    stdDev: number;
  };
  common_caption_words: Array<{
    word: string;
    count: number;
  }>;
}

interface SceneComposition {
  situation_frequencies: Record<string, number>;
  vehicles_distribution: Record<string, number>;
  pedestrians_distribution: Record<string, number>;
  road_features_frequency: Record<string, number>;
}

interface EgoActionCorrelation {
  speed_distribution: Record<string, number>;
  speed_vs_accelerate: Array<{
    speed: number;
    accelerate: number;
  }>;
  speed_vs_steering: Array<{
    speed: number;
    steering: number;
  }>;
  tl_stop_vs_brake: {
    true: {
      avg: number;
      min: number;
      max: number;
      count: number;
    };
    false: {
      avg: number;
      min: number;
      max: number;
      count: number;
    };
  };
}

interface DBStatisticsResponse {
  system_health: SystemHealth;
  model_performance: ModelPerformance;
  scene_composition: SceneComposition;
  ego_action_correlation: EgoActionCorrelation;
  time_range: {
    from_time: string;
    to_time: string;
  };
}

interface DBStatisticsParams {
  from_time?: string;
  to_time?: string;
}

const getDefaultTimestamps = () => {
  const today = new Date();
  const startDate = new Date();
  startDate.setDate(today.getDate() - 7); // Default to last 7 days

  return {
    from_time: startDate.toISOString(),
    to_time: today.toISOString()
  };
};

// Helper to get Date object from ISO string
const getDateFromISOString = (isoString?: string) => {
  if (!isoString) return undefined;
  try {
    const date = new Date(isoString);
    return isNaN(date.getTime()) ? undefined : date;
  } catch (e) {
    return undefined;
  }
};

export const DBStatistics = () => {
  const [data, setData] = useState<DBStatisticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<DBStatisticsParams>(getDefaultTimestamps());
  const [dateError, setDateError] = useState<string | null>(null);

  // Add chart configuration
  const chartConfig = {
    count: {
      label: "Entries",
      color: "hsl(var(--primary))"
    },
    value: {
      label: "Value",
      color: "hsl(var(--primary))"
    },
    speed: {
      label: "Speed",
      color: "hsl(var(--primary))"
    },
    accelerate: {
      label: "Acceleration",
      color: "hsl(var(--primary))"
    },
    steering: {
      label: "Steering",
      color: "hsl(var(--info))"
    }
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const queryParams = new URLSearchParams();
      if (params.from_time) {
        queryParams.append("from_time", params.from_time);
      }
      if (params.to_time) {
        queryParams.append("to_time", params.to_time);
      }

      const response = await fetch(`http://localhost:8000/db_statistics/?${queryParams.toString()}`);

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const result: DBStatisticsResponse = await response.json();
      setData(result);
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

  const handleDateChange = (field: 'from_time' | 'to_time', value: Date | undefined) => {
    if (!value) {
      setParams(prev => ({ ...prev, [field]: undefined }));
      return;
    }

    try {
      setDateError(null);

      const newParams = { ...params, [field]: value.toISOString() };

      // Validate start date is before end date
      if (newParams.from_time && newParams.to_time) {
        const start = new Date(newParams.from_time);
        const end = new Date(newParams.to_time);

        if (start > end) {
          setDateError("Start date must be before end date");
          return;
        }
      }

      setParams(newParams);
    } catch (e) {
      setDateError("Invalid date format");
    }
  };

  // Transform entries over time data for chart
  const entriesOverTimeData = data ? Object.entries(data.system_health.entries_over_time).map(
    ([date, count]) => ({ date, count })
  ) : [];

  // Transform time taken distribution data for chart
  const timeTakenData = data ? Object.entries(data.model_performance.time_taken_distribution).map(
    ([range, count]) => ({ range, count })
  ) : [];

  // Transform situation frequencies data for chart
  const situationData = data ? Object.entries(data.scene_composition.situation_frequencies).map(
    ([situation, count]) => ({ situation, count })
  ) : [];

  // Transform vehicles distribution data for chart
  const vehiclesData = data ? Object.entries(data.scene_composition.vehicles_distribution).map(
    ([range, count]) => ({ range, count })
  ) : [];

  // Transform pedestrians distribution data for chart
  const pedestriansData = data ? Object.entries(data.scene_composition.pedestrians_distribution).map(
    ([range, count]) => ({ range, count })
  ) : [];

  // Transform road features frequency data for chart
  const roadFeaturesData = data ? Object.entries(data.scene_composition.road_features_frequency).map(
    ([feature, count]) => ({ feature, count })
  ) : [];

  // Transform speed distribution data for chart
  const speedDistributionData = data ? Object.entries(data.ego_action_correlation.speed_distribution).map(
    ([range, count]) => ({ range, count })
  ) : [];

  if (loading) {
    return <LoadingCard title="Loading database statistics..." />;
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
            <Database className="w-5 h-5 text-primary" />
            Database Statistics
          </CardTitle>
          <CardDescription>
            Performance metrics and insights from the database
          </CardDescription>
        </div>
        <Button onClick={handleRefresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <label className="text-sm font-medium mb-2 block">From</label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-start text-left font-normal"
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {params.from_time ? (
                    format(new Date(params.from_time), "PPP HH:mm")
                  ) : (
                    <span className="text-muted-foreground">Pick a date</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={getDateFromISOString(params.from_time)}
                  onSelect={(date) => handleDateChange('from_time', date)}
                  initialFocus
                />
                <div className="p-3 border-t border-border">
                  <Input
                    type="time"
                    value={params.from_time ? format(new Date(params.from_time), "HH:mm") : ""}
                    onChange={(e) => {
                      const currentDate = getDateFromISOString(params.from_time) || new Date();
                      const [hours, minutes] = e.target.value.split(':').map(Number);
                      currentDate.setHours(hours, minutes);
                      handleDateChange('from_time', currentDate);
                    }}
                  />
                </div>
              </PopoverContent>
            </Popover>
          </div>
          <div className="flex-1">
            <label className="text-sm font-medium mb-2 block">To</label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-start text-left font-normal"
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {params.to_time ? (
                    format(new Date(params.to_time), "PPP HH:mm")
                  ) : (
                    <span className="text-muted-foreground">Pick a date</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={getDateFromISOString(params.to_time)}
                  onSelect={(date) => handleDateChange('to_time', date)}
                  initialFocus
                />
                <div className="p-3 border-t border-border">
                  <Input
                    type="time"
                    value={params.to_time ? format(new Date(params.to_time), "HH:mm") : ""}
                    onChange={(e) => {
                      const currentDate = getDateFromISOString(params.to_time) || new Date();
                      const [hours, minutes] = e.target.value.split(':').map(Number);
                      currentDate.setHours(hours, minutes);
                      handleDateChange('to_time', currentDate);
                    }}
                  />
                </div>
              </PopoverContent>
            </Popover>
          </div>
          <div className="flex items-end">
            <Button
              onClick={fetchData}
              className="bg-gradient-primary w-full sm:w-auto"
              disabled={!!dateError}
            >
              <Database className="w-4 h-4 mr-2" />
              Generate Report
            </Button>
          </div>
        </div>

        {dateError && (
          <div className="mb-4 p-2 bg-destructive/10 text-destructive rounded flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">{dateError}</span>
          </div>
        )}

        {data && (
          <>
            {/* System Health Section */}
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-4">System Health</h2>

              <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 mb-4">
                <Card className="bg-card/50 shadow-sm">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <h3 className="text-muted-foreground mb-2">Total Entries</h3>
                      <p className="text-3xl font-bold text-primary">{data.system_health.total_entries.toLocaleString()}</p>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-card/50 shadow-sm">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <h3 className="text-muted-foreground mb-2">Database Size</h3>
                      <p className="text-3xl font-bold text-primary">{data.system_health.database_size_mb.toFixed(2)} MB</p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <h3 className="text-lg font-medium mb-2">Entries Over Time</h3>
              <div className="w-full h-[250px]">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={entriesOverTimeData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="date"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        className="text-muted-foreground"
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Entries']}
                        labelFormatter={(label) => `Date: ${label}`}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="count"
                        name="Entries"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </div>

            {/* Model Performance Section */}
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-4">Model Performance</h2>

              <h3 className="text-lg font-medium mb-2">Time Taken Distribution</h3>
              <div className="w-full h-[250px] mb-6">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={timeTakenData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="range"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        className="text-muted-foreground"
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Instances']}
                        labelFormatter={(label) => `Time Range: ${label}`}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Bar dataKey="count" name="Count" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>

              <div className="grid gap-6 grid-cols-1 lg:grid-cols-3 mb-6">
                {/* Accelerate Distribution */}
                <Card className="bg-card/50 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Acceleration Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Min</p>
                        <p className="font-medium">{data.model_performance.accelerate_distribution.min.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Max</p>
                        <p className="font-medium">{data.model_performance.accelerate_distribution.max.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Avg</p>
                        <p className="font-medium">{data.model_performance.accelerate_distribution.avg.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Std Dev</p>
                        <p className="font-medium">{data.model_performance.accelerate_distribution.stdDev.toFixed(2)}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Brake Distribution */}
                <Card className="bg-card/50 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Brake Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Min</p>
                        <p className="font-medium">{data.model_performance.brake_distribution.min.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Max</p>
                        <p className="font-medium">{data.model_performance.brake_distribution.max.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Avg</p>
                        <p className="font-medium">{data.model_performance.brake_distribution.avg.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Std Dev</p>
                        <p className="font-medium">{data.model_performance.brake_distribution.stdDev.toFixed(2)}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Steering Distribution */}
                <Card className="bg-card/50 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Steering Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Min</p>
                        <p className="font-medium">{data.model_performance.steering_distribution.min.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Max</p>
                        <p className="font-medium">{data.model_performance.steering_distribution.max.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Avg</p>
                        <p className="font-medium">{data.model_performance.steering_distribution.avg.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm">Std Dev</p>
                        <p className="font-medium">{data.model_performance.steering_distribution.stdDev.toFixed(2)}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Common Caption Words */}
              <h3 className="text-lg font-medium mb-2">Common Caption Words</h3>
              <div className="w-full h-[250px]">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.model_performance.common_caption_words}
                      margin={{ top: 10, right: 30, left: 20, bottom: 40 }}
                      layout="vertical"
                    >
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis type="number" className="text-muted-foreground" />
                      <YAxis
                        type="category"
                        dataKey="word"
                        className="text-muted-foreground"
                        width={80}
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Occurrences']}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Bar dataKey="count" name="Count" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </div>

            {/* Scene Composition Section */}
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-4">Scene Composition</h2>

              <h3 className="text-lg font-medium mb-2">Situation Frequencies</h3>
              <div className="w-full h-[250px] mb-6">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={situationData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="situation"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        className="text-muted-foreground"
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Instances']}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Bar dataKey="count" name="Count">
                        {situationData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={index === 0 ? "hsl(var(--primary))" :
                              index === 1 ? "hsl(var(--info))" :
                                "hsl(var(--warning))"}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>

              <div className="grid gap-6 grid-cols-1 lg:grid-cols-2 mb-6">
                {/* Vehicles Distribution */}
                <div>
                  <h3 className="text-lg font-medium mb-2">Vehicles Distribution</h3>
                  <div className="w-full h-[250px]">
                    <ChartContainer config={chartConfig} className="w-full h-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={vehiclesData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis
                            dataKey="range"
                            className="text-muted-foreground"
                          />
                          <YAxis
                            className="text-muted-foreground"
                          />
                          <Tooltip
                            formatter={(value: number) => [value.toLocaleString(), 'Count']}
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                          />
                          <Bar dataKey="count" name="Count" fill="hsl(var(--info))" />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </div>
                </div>

                {/* Pedestrians Distribution */}
                <div>
                  <h3 className="text-lg font-medium mb-2">Pedestrians Distribution</h3>
                  <div className="w-full h-[250px]">
                    <ChartContainer config={chartConfig} className="w-full h-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={pedestriansData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis
                            dataKey="range"
                            className="text-muted-foreground"
                          />
                          <YAxis
                            className="text-muted-foreground"
                          />
                          <Tooltip
                            formatter={(value: number) => [value.toLocaleString(), 'Count']}
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                          />
                          <Bar dataKey="count" name="Count" fill="hsl(var(--success))" />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </div>
                </div>
              </div>

              {/* Road Features Frequency */}
              <h3 className="text-lg font-medium mb-2">Road Features Frequency</h3>
              <div className="w-full h-[250px]">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={roadFeaturesData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="feature"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        className="text-muted-foreground"
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Count']}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Bar dataKey="count" name="Count" fill="hsl(var(--warning))" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </div>

            {/* Ego-Vehicle & Action Correlation */}
            <div>
              <h2 className="text-xl font-bold mb-4">Ego-Vehicle & Action Correlation</h2>

              {/* Speed Distribution */}
              <h3 className="text-lg font-medium mb-2">Speed Distribution</h3>
              <div className="w-full h-[250px] mb-6">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={speedDistributionData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="range"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        className="text-muted-foreground"
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Count']}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Bar dataKey="count" name="Count" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>

              <div className="grid gap-6 grid-cols-1 lg:grid-cols-2 mb-6">
                {/* Speed vs Acceleration */}
                <div>
                  <h3 className="text-lg font-medium mb-2">Speed vs Acceleration</h3>
                  <div className="w-full h-[250px]">
                    <ChartContainer config={chartConfig} className="w-full h-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis
                            type="number"
                            dataKey="speed"
                            name="Speed"
                            unit=" km/h"
                            className="text-muted-foreground"
                          />
                          <YAxis
                            type="number"
                            dataKey="accelerate"
                            name="Acceleration"
                            className="text-muted-foreground"
                          />
                          <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                          />
                          <Scatter
                            name="Speed vs Acceleration"
                            data={data.ego_action_correlation.speed_vs_accelerate}
                            fill="hsl(var(--primary))"
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </div>
                </div>

                {/* Speed vs Steering */}
                <div>
                  <h3 className="text-lg font-medium mb-2">Speed vs Steering</h3>
                  <div className="w-full h-[250px]">
                    <ChartContainer config={chartConfig} className="w-full h-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis
                            type="number"
                            dataKey="speed"
                            name="Speed"
                            unit=" km/h"
                            className="text-muted-foreground"
                          />
                          <YAxis
                            type="number"
                            dataKey="steering"
                            name="Steering"
                            className="text-muted-foreground"
                          />
                          <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                          />
                          <Scatter
                            name="Speed vs Steering"
                            data={data.ego_action_correlation.speed_vs_steering}
                            fill="hsl(var(--info))"
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </div>
                </div>
              </div>

              {/* Traffic Light Stop vs Brake */}
              <h3 className="text-lg font-medium mb-2">Traffic Light Stop vs Brake</h3>
              <div className="w-full h-[250px]">
                <ChartContainer config={chartConfig} className="w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        {
                          name: "TL Stop True",
                          value: data.ego_action_correlation.tl_stop_vs_brake.true.avg,
                          min: data.ego_action_correlation.tl_stop_vs_brake.true.min,
                          max: data.ego_action_correlation.tl_stop_vs_brake.true.max,
                          count: data.ego_action_correlation.tl_stop_vs_brake.true.count
                        },
                        {
                          name: "TL Stop False",
                          value: data.ego_action_correlation.tl_stop_vs_brake.false.avg,
                          min: data.ego_action_correlation.tl_stop_vs_brake.false.min,
                          max: data.ego_action_correlation.tl_stop_vs_brake.false.max,
                          count: data.ego_action_correlation.tl_stop_vs_brake.false.count
                        }
                      ]}
                      margin={{ top: 10, right: 30, left: 20, bottom: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="name"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        className="text-muted-foreground"
                        label={{ value: 'Brake Value (avg)', angle: -90, position: 'insideLeft' }}
                        domain={[0, 1]}
                      />
                      <Tooltip
                        formatter={(value: number, name: string, props: any) => {
                          if (name === "Avg") return [value.toFixed(2), name];
                          if (name === "Min") return [props.payload.min.toFixed(2), name];
                          if (name === "Max") return [props.payload.max.toFixed(2), name];
                          return [value, name];
                        }}
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}
                      />
                      <Bar dataKey="value" name="Avg" fill="hsl(var(--primary))">
                        <Legend />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}; 
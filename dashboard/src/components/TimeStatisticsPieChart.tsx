import { useState, useEffect } from "react";
import { apiClient, PieChartSegment, PieChartParams } from "@/lib/api";
import { LoadingSpinner, LoadingCard } from "@/components/LoadingSpinner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { PieChart as PieChartIcon, RefreshCw } from "lucide-react";
import { DateTimePicker } from "@/components/ui/date-time-picker";

const COLORS = [
  '#FF6384', // pink/red
  '#36A2EB', // blue
  '#FFCE56', // yellow
  '#4BC0C0', // teal
  '#9966FF', // purple
  '#FF9F40'  // orange
];

const getDefaultTimestamps = () => {
  const today = new Date();
  const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const endOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate(), 23, 59, 59, 999);
  
  return {
    from_timestamp: startOfDay.toISOString(),
    to_timestamp: endOfDay.toISOString()
  };
};

export const TimeStatisticsPieChart = () => {
  const [data, setData] = useState<PieChartSegment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [params, setParams] = useState<PieChartParams>({
    segment_size: 0.5,
    ...getDefaultTimestamps()
  });

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiClient.fetchPieChartData(params);
      setData(response.segments);
      setTotalCount(response.total_count);
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

  const chartData = data.map(segment => ({
    name: segment.label,
    value: segment.count,
    percentage: totalCount > 0 ? ((segment.count / totalCount) * 100).toFixed(1) : '0'
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border rounded-lg p-3 shadow-floating">
          <p className="font-medium">{data.name}</p>
          <p className="text-sm text-muted-foreground">
            Count: {data.value} ({data.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return <LoadingCard title="Loading performance distribution..." />;
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
            <PieChartIcon className="w-5 h-5 text-primary" />
            Performance Distribution
          </CardTitle>
          <CardDescription>
            Execution time distribution across segments (Total: {totalCount} predictions)
          </CardDescription>
        </div>
        <Button onClick={handleRefresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        <div className="mb-6 flex gap-4">
          <div className="flex-1">
            <label className="text-sm font-medium mb-2 block">Segment Size (seconds)</label>
            <Input
              type="number"
              min="0.1"
              max="10"
              step="0.1"
              value={params.segment_size || 0.5}
              onChange={(e) => setParams(prev => ({ 
                ...prev, 
                segment_size: parseFloat(e.target.value) || 0.5 
              }))}
            />
          </div>
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
              Update
            </Button>
          </div>
        </div>

        {chartData.length > 0 ? (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name} (${percentage}%)`}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={`url(#gradient-${index})`} 
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                {/* Add gradient definitions */}
                <defs>
                  {COLORS.map((color, index) => (
                    <linearGradient 
                      key={`gradient-${index}`} 
                      id={`gradient-${index}`} 
                      x1="0" y1="0" x2="1" y2="1"
                    >
                      <stop offset="0%" stopColor={color} stopOpacity={1} />
                      <stop offset="100%" stopColor={color} stopOpacity={0.6} />
                    </linearGradient>
                  ))}
                </defs>
              </PieChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-96 flex items-center justify-center text-muted-foreground">
            No data available for the selected parameters
          </div>
        )}

        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.map((segment, index) => (
            <div key={segment.label} className="bg-muted/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <div 
                  className="w-4 h-4 rounded-full" 
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="font-medium">{segment.label}</span>
              </div>
              <div className="text-2xl font-bold text-primary">
                {segment.count}
              </div>
              <div className="text-sm text-muted-foreground">
                {totalCount > 0 ? ((segment.count / totalCount) * 100).toFixed(1) : '0'}% of total
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar as CalendarIcon, Car, RefreshCw, AlertCircle } from "lucide-react";
import { LoadingCard } from "@/components/LoadingSpinner";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

interface CarStatistic {
    number_of_pedestrian: number;
    number_of_vehicle: number;
    accel: number;
    speed: number;
    brake_pressure: number;
    timestamp: string;
}

interface CarStatisticsResponse {
    statistics: CarStatistic[];
    count: number;
}

interface CarStatisticsParams {
    start_timestamp?: string;
    end_timestamp?: string;
}

const getDefaultTimestamps = () => {
    const today = new Date();
    const startDate = new Date();
    startDate.setDate(today.getDate() - 2);

    return {
        start_timestamp: startDate.toISOString(),
        end_timestamp: today.toISOString()
    };
};

// Helper function to format ISO string for datetime-local input
const formatDateForInput = (isoString?: string) => {
    if (!isoString) return '';
    try {
        // Format to YYYY-MM-DDThh:mm which is required by datetime-local inputs
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return '';
        
        return date.toISOString().substring(0, 16);
    } catch (e) {
        return '';
    }
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

export const CarTrackingStatistics = () => {
    const [data, setData] = useState<CarStatistic[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [params, setParams] = useState<CarStatisticsParams>(getDefaultTimestamps());
    const [dateError, setDateError] = useState<string | null>(null);

    const fetchData = async () => {
        try {
            setLoading(true);
            setError(null);

            const queryParams = new URLSearchParams();
            if (params.start_timestamp) {
                queryParams.append("start_timestamp", params.start_timestamp);
            }
            if (params.end_timestamp) {
                queryParams.append("end_timestamp", params.end_timestamp);
            }

            const response = await fetch(`http://localhost:8000/statistics?${queryParams.toString()}`);
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            console.log(response);

            const result: CarStatisticsResponse = await response.json();
            setData(result.statistics);
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

    const handleDateChange = (field: 'start_timestamp' | 'end_timestamp', value: Date | undefined) => {
        if (!value) {
            setParams(prev => ({ ...prev, [field]: undefined }));
            return;
        }
        
        try {
            setDateError(null);
            
            const newParams = { ...params, [field]: value.toISOString() };
            
            // Validate start date is before end date
            if (newParams.start_timestamp && newParams.end_timestamp) {
                const start = new Date(newParams.start_timestamp);
                const end = new Date(newParams.end_timestamp);
                
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

    // Transform data for chart
    const chartData = data.map((stat) => ({
        ...stat,
        chartTimestamp: new Date(stat.timestamp).getTime(),
        displayTime: new Date(stat.timestamp).toLocaleTimeString(),
    }));

    const chartConfig = {
        brake_pressure: {
            label: "Brake Pressure",
            color: "hsl(var(--primary))",
        },
        speed: {
            label: "Speed",
            color: "hsl(var(--info))",
        },
        accel: {
            label: "Acceleration",
            color: "hsl(var(--success))",
        },
        number_of_vehicle: {
            label: "Number of Vehicles",
            color: "hsl(var(--warning))",
        },
        number_of_pedestrian: {
            label: "Number of Pedestrians",
            color: "hsl(var(--destructive))",
        },
    };

    if (loading) {
        return <LoadingCard title="Loading car tracking statistics..." />;
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
                        <Car className="w-5 h-5 text-primary" />
                        Car Tracking Statistics
                    </CardTitle>
                    <CardDescription>
                        Performance metrics for self-driving car ({data.length} records)
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
                                    {params.start_timestamp ? (
                                        format(new Date(params.start_timestamp), "PPP HH:mm")
                                    ) : (
                                        <span className="text-muted-foreground">Pick a date</span>
                                    )}
                                </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-auto p-0" align="start">
                                <Calendar
                                    mode="single"
                                    selected={getDateFromISOString(params.start_timestamp)}
                                    onSelect={(date) => handleDateChange('start_timestamp', date)}
                                    initialFocus
                                />
                                <div className="p-3 border-t border-border">
                                    <Input
                                        type="time"
                                        value={params.start_timestamp ? format(new Date(params.start_timestamp), "HH:mm") : ""}
                                        onChange={(e) => {
                                            const currentDate = getDateFromISOString(params.start_timestamp) || new Date();
                                            const [hours, minutes] = e.target.value.split(':').map(Number);
                                            currentDate.setHours(hours, minutes);
                                            handleDateChange('start_timestamp', currentDate);
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
                                    {params.end_timestamp ? (
                                        format(new Date(params.end_timestamp), "PPP HH:mm")
                                    ) : (
                                        <span className="text-muted-foreground">Pick a date</span>
                                    )}
                                </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-auto p-0" align="start">
                                <Calendar
                                    mode="single"
                                    selected={getDateFromISOString(params.end_timestamp)}
                                    onSelect={(date) => handleDateChange('end_timestamp', date)}
                                    initialFocus
                                />
                                <div className="p-3 border-t border-border">
                                    <Input
                                        type="time"
                                        value={params.end_timestamp ? format(new Date(params.end_timestamp), "HH:mm") : ""}
                                        onChange={(e) => {
                                            const currentDate = getDateFromISOString(params.end_timestamp) || new Date();
                                            const [hours, minutes] = e.target.value.split(':').map(Number);
                                            currentDate.setHours(hours, minutes);
                                            handleDateChange('end_timestamp', currentDate);
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
                            <CalendarIcon className="w-4 h-4 mr-2" />
                            Filter
                        </Button>
                    </div>
                </div>
                
                {dateError && (
                    <div className="mb-4 p-2 bg-destructive/10 text-destructive rounded flex items-center gap-2">
                        <AlertCircle className="h-4 w-4" />
                        <span className="text-sm">{dateError}</span>
                    </div>
                )}

                {/* Brake Pressure Chart */}
                <div className="mb-8">
                    <h3 className="text-lg font-medium mb-2">Brake Pressure</h3>
                    <div className="w-full h-[200px] sm:h-[250px]">
                        <ChartContainer config={chartConfig} className="w-full h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis
                                        dataKey="chartTimestamp"
                                        type="number"
                                        scale="time"
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                        className="text-muted-foreground"
                                    />
                                    <YAxis
                                        className="text-muted-foreground"
                                    />
                                    <ChartTooltip
                                        content={<ChartTooltipContent
                                            formatter={(value) => [`${Number(value).toFixed(2)}`, 'Brake Pressure']}
                                            labelFormatter={(label) => new Date(Number(label)).toLocaleString()}
                                        />}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="brake_pressure"
                                        stroke="hsl(var(--primary))"
                                        strokeWidth={2}
                                        dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 3 }}
                                        activeDot={{ r: 5, fill: "hsl(var(--primary))" }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartContainer>
                    </div>
                </div>

                {/* Speed Chart */}
                <div className="mb-8">
                    <h3 className="text-lg font-medium mb-2">Speed</h3>
                    <div className="w-full h-[200px] sm:h-[250px]">
                        <ChartContainer config={chartConfig} className="w-full h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis
                                        dataKey="chartTimestamp"
                                        type="number"
                                        scale="time"
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                        className="text-muted-foreground"
                                    />
                                    <YAxis
                                        className="text-muted-foreground"
                                    />
                                    <ChartTooltip
                                        content={<ChartTooltipContent
                                            formatter={(value) => [`${Number(value).toFixed(2)}`, 'Speed']}
                                            labelFormatter={(label) => new Date(Number(label)).toLocaleString()}
                                        />}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="speed"
                                        stroke="hsl(var(--info))"
                                        strokeWidth={2}
                                        dot={{ fill: "hsl(var(--info))", strokeWidth: 2, r: 3 }}
                                        activeDot={{ r: 5, fill: "hsl(var(--info))" }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartContainer>
                    </div>
                </div>

                {/* Acceleration Chart */}
                <div className="mb-8">
                    <h3 className="text-lg font-medium mb-2">Acceleration</h3>
                    <div className="w-full h-[200px] sm:h-[250px]">
                        <ChartContainer config={chartConfig} className="w-full h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis
                                        dataKey="chartTimestamp"
                                        type="number"
                                        scale="time"
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                        className="text-muted-foreground"
                                    />
                                    <YAxis
                                        className="text-muted-foreground"
                                    />
                                    <ChartTooltip
                                        content={<ChartTooltipContent
                                            formatter={(value) => [`${Number(value).toFixed(2)}`, 'Acceleration']}
                                            labelFormatter={(label) => new Date(Number(label)).toLocaleString()}
                                        />}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="accel"
                                        stroke="hsl(var(--success))"
                                        strokeWidth={2}
                                        dot={{ fill: "hsl(var(--success))", strokeWidth: 2, r: 3 }}
                                        activeDot={{ r: 5, fill: "hsl(var(--success))" }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartContainer>
                    </div>
                </div>

                {/* Number of Vehicles Chart */}
                <div className="mb-8">
                    <h3 className="text-lg font-medium mb-2">Number of Vehicles</h3>
                    <div className="w-full h-[200px] sm:h-[250px]">
                        <ChartContainer config={chartConfig} className="w-full h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis
                                        dataKey="chartTimestamp"
                                        type="number"
                                        scale="time"
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                        className="text-muted-foreground"
                                    />
                                    <YAxis
                                        className="text-muted-foreground"
                                    />
                                    <ChartTooltip
                                        content={<ChartTooltipContent
                                            formatter={(value) => [`${Number(value)}`, 'Vehicles']}
                                            labelFormatter={(label) => new Date(Number(label)).toLocaleString()}
                                        />}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="number_of_vehicle"
                                        stroke="hsl(var(--warning))"
                                        strokeWidth={2}
                                        dot={{ fill: "hsl(var(--warning))", strokeWidth: 2, r: 3 }}
                                        activeDot={{ r: 5, fill: "hsl(var(--warning))" }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartContainer>
                    </div>
                </div>

                {/* Number of Pedestrians Chart */}
                <div className="mb-4">
                    <h3 className="text-lg font-medium mb-2">Number of Pedestrians</h3>
                    <div className="w-full h-[200px] sm:h-[250px]">
                        <ChartContainer config={chartConfig} className="w-full h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis
                                        dataKey="chartTimestamp"
                                        type="number"
                                        scale="time"
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                        className="text-muted-foreground"
                                    />
                                    <YAxis
                                        className="text-muted-foreground"
                                    />
                                    <ChartTooltip
                                        content={<ChartTooltipContent
                                            formatter={(value) => [`${Number(value)}`, 'Pedestrians']}
                                            labelFormatter={(label) => new Date(Number(label)).toLocaleString()}
                                        />}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="number_of_pedestrian"
                                        stroke="hsl(var(--destructive))"
                                        strokeWidth={2}
                                        dot={{ fill: "hsl(var(--destructive))", strokeWidth: 2, r: 3 }}
                                        activeDot={{ r: 5, fill: "hsl(var(--destructive))" }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartContainer>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}; 
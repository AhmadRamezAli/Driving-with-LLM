import { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import {
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Eye
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { LoadingSpinner } from "./LoadingSpinner";
import { PredictionLog, LogsResponse } from "@/types/prediction";

type PredictionLogsTableProps = {
  onViewDetails: (log: PredictionLog) => void;
};

export function PredictionLogsTable({ onViewDetails }: PredictionLogsTableProps) {
  const [logs, setLogs] = useState<PredictionLog[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(10);
  const [skip, setSkip] = useState<number>(0);
  const [total, setTotal] = useState<number>(0);
  const [sortBy, setSortBy] = useState<string>("timestamp");
  const [sortOrder, setSortOrder] = useState<number>(-1); // Default descending

  useEffect(() => {
    const fetchLogs = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `http://localhost:8000/prediction-logs?limit=${limit}&skip=${skip}&sort_by=${sortBy}&sort_order=${sortOrder}`
        );
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data: LogsResponse = await response.json();
        console.log(data);
        setLogs(data.logs);
        setTotal(data.count);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch logs");
        console.error("Error fetching prediction logs:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [limit, skip, sortBy, sortOrder]);

  const handleSort = (column: string) => {
    if (sortBy === column) {
      // Toggle sort order if clicking the same column
      setSortOrder(sortOrder === 1 ? -1 : 1);
    } else {
      // Default to descending when selecting a new column
      setSortBy(column);
      setSortOrder(-1);
    }
    // Reset pagination when sorting changes
    setSkip(0);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(skip / limit) + 1;

  const goToPage = (page: number) => {
    const newSkip = (page - 1) * limit;
    setSkip(newSkip);
  };

  const renderPagination = () => {
    return (
      <div className="flex items-center justify-between space-x-2 py-4">
        <div className="text-sm text-muted-foreground">
          Showing {logs.length > 0 ? skip + 1 : 0} to{" "}
          {Math.min(skip + limit, total)} of {total} entries
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4" />
            <span className="sr-only">Previous Page</span>
          </Button>
          <div className="text-sm font-medium">
            Page {currentPage} of {totalPages}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            <ChevronRight className="h-4 w-4" />
            <span className="sr-only">Next Page</span>
          </Button>
        </div>
      </div>
    );
  };

  const renderSortIcon = (column: string) => {
    if (sortBy === column) {
      return sortOrder === 1 ?
        <ArrowUp className="ml-2 h-4 w-4" /> :
        <ArrowDown className="ml-2 h-4 w-4" />;
    }
    return <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />;
  };

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center text-red-500">
            Error loading logs: {error}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-0">
        {loading ? (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner />
          </div>
        ) : (
          <>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead
                      className="cursor-pointer"
                      onClick={() => handleSort("timestamp")}
                    >
                      <div className="flex items-center">
                        Timestamp
                        {renderSortIcon("timestamp")}
                      </div>
                    </TableHead>
                    <TableHead
                      className="cursor-pointer"
                      onClick={() => handleSort("caption")}
                    >
                      <div className="flex items-center">
                        Caption
                        {renderSortIcon("caption")}
                      </div>
                    </TableHead>
                    <TableHead
                      className="cursor-pointer text-right"
                      onClick={() => handleSort("accelerate")}
                    >
                      <div className="flex items-center justify-end">
                        Accelerate
                        {renderSortIcon("accelerate")}
                      </div>
                    </TableHead>
                    <TableHead
                      className="cursor-pointer text-right"
                      onClick={() => handleSort("brake")}
                    >
                      <div className="flex items-center justify-end">
                        Brake
                        {renderSortIcon("brake")}
                      </div>
                    </TableHead>
                    <TableHead
                      className="cursor-pointer text-right"
                      onClick={() => handleSort("steering")}
                    >
                      <div className="flex items-center justify-end">
                        Steering
                        {renderSortIcon("steering")}
                      </div>
                    </TableHead>
                    <TableHead
                      className="cursor-pointer text-right"
                      onClick={() => handleSort("time_taken")}
                    >
                      <div className="flex items-center justify-end">
                        Time (ms)
                        {renderSortIcon("time_taken")}
                      </div>
                    </TableHead>
                    <TableHead className="text-center">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {logs.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={7}
                        className="h-24 text-center"
                      >
                        No logs found
                      </TableCell>
                    </TableRow>
                  ) : (
                    logs.map((log, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">
                          {formatDate(log.timestamp)}
                        </TableCell>
                        <TableCell className="max-w-[200px] truncate">
                          {log.caption || "N/A"}
                        </TableCell>
                        <TableCell className="text-right">
                          {log.accelerate !== undefined
                            ? log.accelerate.toFixed(2)
                            : "N/A"}
                        </TableCell>
                        <TableCell className="text-right">
                          {log.brake !== undefined
                            ? log.brake.toFixed(2)
                            : "N/A"}
                        </TableCell>
                        <TableCell className="text-right">
                          {log.steering !== undefined
                            ? log.steering.toFixed(2)
                            : "N/A"}
                        </TableCell>
                        <TableCell className="text-right">
                          {log.time_taken !== undefined
                            ? (log.time_taken * 1000).toFixed(1)
                            : "N/A"}
                        </TableCell>
                        <TableCell className="text-center">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onViewDetails(log)}
                            title="View Details"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
            <div className="px-4">{renderPagination()}</div>
          </>
        )}
      </CardContent>
    </Card>
  );
} 
import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TimeStatisticsTable } from "@/components/TimeStatisticsTable";
import { TimeStatisticsPieChart } from "@/components/TimeStatisticsPieChart";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft,
  BarChart3,
  Car,
  Clock,
  Clipboard,
  Database,
  PieChart,
  TrendingUp,
} from "lucide-react";
import { Link } from "react-router-dom";
import { PredictionLogsContainer } from "@/components/PredictionLogsContainer";
import { CarTrackingStatistics } from "@/components/CarTrackingStatistics";
import { DBStatistics } from "@/components/DBStatistics";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-dashboard-bg">
      {/* Header */}
      <div className="bg-gradient-hero border-b">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* <Link to="/">
                <Button variant="outline" size="sm">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Home
                </Button> */}
              {/* </Link> */}
              <div>
                <h1 className="text-3xl font-bold text-foreground">
                  Analytics Dashboard
                </h1>
                <p className="text-muted-foreground mt-1">
                  Self-driving car performance monitoring and insights
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Content */}
      <div className="container mx-auto px-6 py-8">
        <Tabs defaultValue="time-analysis" className="space-y-6">
          <TabsList className="grid w-full grid-cols-7 lg:w-auto lg:grid-cols-4 bg-card shadow-card">
            <TabsTrigger
              value="time-analysis"
              className="flex items-center gap-2"
            >
              <Clock className="w-4 h-4" />
              <span className="hidden sm:inline">Time Analysis</span>
            </TabsTrigger>
            <TabsTrigger
              value="car-tracking"
              className="flex items-center gap-2"
            >
              <Car className="w-4 h-4" />
              <span className="hidden sm:inline">Car Tracking</span>
            </TabsTrigger>

            <TabsTrigger value="logs" className="flex items-center gap-2">
              <Clipboard className="w-4 h-4" />
              <span className="hidden sm:inline">Logs</span>
            </TabsTrigger>
            <TabsTrigger
              value="db-statistics"
              className="flex items-center gap-2"
            >
              <Database className="w-4 h-4" />
              <span className="hidden sm:inline">DB Stats</span>
            </TabsTrigger>
          </TabsList>

          {/* Time Analysis Tab */}
          <TabsContent value="time-analysis" className="space-y-6">
            <div className="grid gap-6">
              <TimeStatisticsPieChart />
              <TimeStatisticsTable />
            </div>
          </TabsContent>

          {/* Car Tracking Tab */}
          <TabsContent value="car-tracking" className="space-y-6">
            <CarTrackingStatistics />
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <Card className="bg-card shadow-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  Performance Metrics
                </CardTitle>
                <CardDescription>
                  Advanced performance analysis and optimization insights
                </CardDescription>
              </CardHeader>
              <CardContent className="py-8">
                <div className="text-center text-muted-foreground">
                  <TrendingUp className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">
                    Performance Analytics Coming Soon
                  </h3>
                  <p>
                    Advanced performance metrics and trend analysis will be
                    available here.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="space-y-6">
            <PredictionLogsContainer />
          </TabsContent>

          {/* DB Statistics Tab */}
          <TabsContent value="db-statistics" className="space-y-6">
            <DBStatistics />
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights" className="space-y-6">
            <Card className="bg-card shadow-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-primary" />
                  AI Insights
                </CardTitle>
                <CardDescription>
                  Intelligent analysis and recommendations for optimization
                </CardDescription>
              </CardHeader>
              <CardContent className="py-8">
                <div className="text-center text-muted-foreground">
                  <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">
                    AI Insights Coming Soon
                  </h3>
                  <p>
                    Intelligent recommendations and predictive insights will be
                    available here.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent value="reports" className="space-y-6">
            <Card className="bg-card shadow-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-primary" />
                  Custom Reports
                </CardTitle>
                <CardDescription>
                  Generate and export detailed analytics reports
                </CardDescription>
              </CardHeader>
              <CardContent className="py-8">
                <div className="text-center text-muted-foreground">
                  <PieChart className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">
                    Custom Reports Coming Soon
                  </h3>
                  <p>
                    Advanced reporting and export capabilities will be available
                    here.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Dashboard;

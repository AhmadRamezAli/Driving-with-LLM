const API_BASE_URL = 'http://localhost:8000';

export interface TimeStatistic {
  timestamp: string;
  time_taken: number;
}

export interface TimeStatisticsResponse {
  stats: TimeStatistic[];
  count: number;
  from_timestamp: string;
  to_timestamp: string;
}

export interface PieChartSegment {
  label: string;
  count: number;
  min_value: number;
  max_value: number | null;
}

export interface PieChartResponse {
  segments: PieChartSegment[];
  total_count: number;
  segment_size: number;
  from_timestamp: string;
  to_timestamp: string;
}

export interface TimeStatisticsParams {
  from_timestamp?: string;
  to_timestamp?: string;
  limit?: number;
  skip?: number;
}

export interface PieChartParams {
  segment_size?: number;
  from_timestamp?: string;
  to_timestamp?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async fetchTimeStatistics(params: TimeStatisticsParams = {}): Promise<TimeStatisticsResponse> {
    const searchParams = new URLSearchParams();
    
    if (params.from_timestamp) searchParams.append('from_timestamp', params.from_timestamp);
    if (params.to_timestamp) searchParams.append('to_timestamp', params.to_timestamp);
    if (params.limit) searchParams.append('limit', params.limit.toString());
    if (params.skip) searchParams.append('skip', params.skip.toString());

    const url = `${this.baseUrl}/time-statistics${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch time statistics: ${response.statusText}`);
    }
    
    return response.json();
  }

  async fetchPieChartData(params: PieChartParams = {}): Promise<PieChartResponse> {
    const searchParams = new URLSearchParams();
    
    if (params.segment_size) searchParams.append('segment_size', params.segment_size.toString());
    if (params.from_timestamp) searchParams.append('from_timestamp', params.from_timestamp);
    if (params.to_timestamp) searchParams.append('to_timestamp', params.to_timestamp);

    const url = `${this.baseUrl}/time-statistics/pie-chart${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch pie chart data: ${response.statusText}`);
    }
    
    return response.json();
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
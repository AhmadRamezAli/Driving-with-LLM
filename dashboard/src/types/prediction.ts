export type Vehicle = {
  active: number;
  dynamic: number;
  speed: number;
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  pitch: number;
  half_length: number;
  half_width: number;
  half_height: number;
};

export type Pedestrian = {
  active: number;
  speed: number;
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  crossing: number;
};

export type Route = {
  x: number;
  y: number;
  z: number;
  tangent_dx: number;
  tangent_dy: number;
  pitch: number;
  speed_limit: number;
  has_junction: number;
  road_width0: number;
  road_width1: number;
  has_tl: number;
  tl_go: number;
  tl_gotostop: number;
  tl_stop: number;
  tl_stoptogo: number;
  is_giveway: number;
  is_roundabout: number;
};

export type Ego = {
  accel: number;
  speed: number;
  brake_pressure: number;
  steering_angle: number;
  pitch: number;
  half_length: number;
  half_width: number;
  half_height: number;
  class_start: number;
  class_end: number;
  dynamics_start: number;
  dynamics_end: number;
  prev_action_start: number;
  prev_action_end: number;
  rays_left_start: number;
  rays_left_end: number;
  rays_right_start: number;
  rays_right_end: number;
};

export type Situation = {
  collection: string;
};

export type Scene = {
  vehicles?: Vehicle[];
  pedestrians?: Pedestrian[];
  routes?: Route[];
  ego?: Ego;
  situation?: Situation;
};

export type Prediction = {
  caption: string;
  accelerate: number;
  brake: number;
  steering: number;
  time_taken: number;
};

export type PredictionLog = {
  request: Scene;
  timestamp: string;
  accelerate: number;
  brake: number;
  steering: number;
  caption: string;
  time_taken: number;
};

export type LogsResponse = {
  logs: PredictionLog[];
  count: number;
  skip: number;
  limit: number;
}; 
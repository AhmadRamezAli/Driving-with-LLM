import { useRef, useEffect, useState, useCallback } from 'react';

// Updated interface to match the provided data format
interface Ego {
  accel: number;
  speed: number;
  brake_pressure: number;
  steering_angle: number;
  pitch: number;
  half_length: number;
  half_width: number;
  half_height: number;
  class_start?: number;
  class_end?: number;
  dynamics_start?: number;
  dynamics_end?: number;
  prev_action_start?: number;
  prev_action_end?: number;
  rays_left_start?: number;
  rays_left_end?: number;
  rays_right_start?: number;
  rays_right_end?: number;
}

interface Vehicle {
  active: number;
  dynamic?: number;
  speed: number;
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  pitch?: number;
  half_length: number;
  half_width: number;
  half_height: number;
}

interface Pedestrian {
  active: number;
  speed: number;
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  crossing?: number;
}

interface Route {
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
}

interface Situation {
  collection: string;
}

interface Scene {
  vehicles?: Vehicle[];
  pedestrians?: Pedestrian[];
  routes?: Route[];
  ego?: Ego;
  situation?: Situation;
}

type SceneMapProps = {
  scene: Scene | null;
  width?: number;
  height?: number;
};

// Constants for visualization
const SCALE = 100; // Base scale factor to convert coordinates to pixels
const METRES_M_SCALE = 10.0; // Scale factor for meters
const MS_TO_MPH = 2.23694; // Conversion from m/s to mph
const VELOCITY_MS_SCALE = 5.0; // Scale factor for velocity

const VEHICLE_COLOR = '#3b82f6'; // Blue for vehicles
const EGO_COLOR = '#10b981'; // Green for ego vehicle
const PEDESTRIAN_COLOR = '#ef4444'; // Red for pedestrians
const ROUTE_COLOR = '#f59e0b'; // Amber for route points
const ROUTE_LINE_COLOR = '#d97706'; // Dark amber for route lines
const BACKGROUND_COLOR = '#f8fafc'; // Light gray background
const GRID_COLOR = '#e2e8f0'; // Lighter gray for grid

export function SceneMap({ scene, width = 780, height = 400 }: SceneMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // State for panning/scrolling
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1.0);
  const [autoFitApplied, setAutoFitApplied] = useState(false);

  // Center the view in the middle of the canvas, adjusted by offset
  const centerX = width / 2 + offset.x;
  const centerY = height / 2 + offset.y;

  // Calculate adjusted scale based on METRES_M_SCALE and zoom
  const adjustedScale = SCALE * METRES_M_SCALE * zoom;

  // Auto-fit all data in view
  useEffect(() => {
    if (!scene || autoFitApplied) return;

    // Find bounds of all objects
    let minX = 0, maxX = 0, minY = 0, maxY = 0;
    let hasData = false;

    // Check vehicles
    if (scene.vehicles && scene.vehicles.length > 0) {
      scene.vehicles.forEach(vehicle => {
        if (vehicle.active !== 1) return;
        const x = parseFloat(vehicle.x.toString());
        const y = parseFloat(vehicle.y.toString());
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        hasData = true;
      });
    }

    // Check pedestrians
    if (scene.pedestrians && scene.pedestrians.length > 0) {
      scene.pedestrians.forEach(pedestrian => {
        if (pedestrian.active !== 1) return;
        const x = parseFloat(pedestrian.x.toString());
        const y = parseFloat(pedestrian.y.toString());
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        hasData = true;
      });
    }

    // Check routes
    if (scene.routes && scene.routes.length > 0) {
      scene.routes.forEach(route => {
        const x = parseFloat(route.x.toString());
        const y = parseFloat(route.y.toString());
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        hasData = true;
      });
    }

    if (hasData) {
      // Add padding around the bounds
      const paddingFactor = 0.2; // 20% padding
      const rangeX = maxX - minX;
      const rangeY = maxY - minY;

      minX -= rangeX * paddingFactor;
      maxX += rangeX * paddingFactor;
      minY -= rangeY * paddingFactor;
      maxY += rangeY * paddingFactor;

      // Calculate center of data
      const dataCenterX = (minX + maxX) / 2;
      const dataCenterY = (minY + maxY) / 2;

      // Calculate required zoom to fit all data
      const scaleX = width / (rangeX * SCALE * METRES_M_SCALE);
      const scaleY = height / (rangeY * SCALE * METRES_M_SCALE);
      const newZoom = Math.min(scaleX, scaleY) * 0.8; // 80% to ensure everything fits

      // Prevent extreme zoom levels
      const clampedZoom = Math.max(0.001, Math.min(5, newZoom));

      // Calculate offset to center the data
      const newOffsetX = -dataCenterX * SCALE * METRES_M_SCALE * clampedZoom;
      const newOffsetY = dataCenterY * SCALE * METRES_M_SCALE * clampedZoom;

      // Update state
      setZoom(clampedZoom);
      setOffset({ x: newOffsetX, y: newOffsetY });
    }

    setAutoFitApplied(true);
  }, [scene, width, height, autoFitApplied]);

  // Handle mouse down for dragging
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  }, []);

  // Handle mouse move for dragging
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;

    setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
    setDragStart({ x: e.clientX, y: e.clientY });
  }, [isDragging, dragStart]);

  // Handle mouse up to stop dragging
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Handle mouse wheel for zooming
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();

    // Determine zoom direction
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;

    // Calculate mouse position relative to canvas
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Calculate position relative to center
    const relX = mouseX - centerX;
    const relY = mouseY - centerY;

    // Apply zoom
    setZoom(prev => {
      const newZoom = Math.max(0.001, Math.min(5, prev * zoomFactor));

      // Adjust offset to zoom toward mouse position
      const zoomRatio = 1 - newZoom / prev;
      setOffset(prev => ({
        x: prev.x - relX * zoomRatio,
        y: prev.y - relY * zoomRatio
      }));

      return newZoom;
    });
  }, [centerX, centerY]);

  // Reset view button handler - now includes fitting all data
  const resetView = useCallback(() => {
    setAutoFitApplied(false); // This will trigger the auto-fit effect
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !scene) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = BACKGROUND_COLOR;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    drawGrid(ctx, width, height, centerX, centerY);

    // Draw routes if available
    if (scene.routes && scene.routes.length > 0) {
      drawRoutes(ctx, scene.routes, centerX, centerY, adjustedScale);
    }

    // Draw vehicles if available
    if (scene.vehicles && scene.vehicles.length > 0) {
      scene.vehicles.forEach(vehicle => {
        drawVehicle(ctx, vehicle, centerX, centerY, adjustedScale);
      });
    }

    // Draw pedestrians if available
    if (scene.pedestrians && scene.pedestrians.length > 0) {
      scene.pedestrians.forEach(pedestrian => {
        drawPedestrian(ctx, pedestrian, centerX, centerY, adjustedScale);
      });
    }

    // Draw ego vehicle (always at 0,0)
    if (scene.ego) {
      drawEgoVehicle(ctx, centerX, centerY, adjustedScale, scene.ego);
    }

  }, [scene, width, height, centerX, centerY, adjustedScale, zoom]);

  const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number, centerX: number, centerY: number) => {
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;

    // Draw grid lines
    const gridSize = 20; // Size of grid cells in pixels

    // Vertical lines
    for (let x = centerX % gridSize; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = centerY % gridSize; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.lineWidth = 2;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Add scale indicators
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';

    // X-axis labels
    for (let x = centerX + gridSize; x < width; x += gridSize * 5) {
      const meters = ((x - centerX) / adjustedScale).toFixed(1);
      ctx.fillText(`${meters}m`, x, centerY + 15);
    }

    for (let x = centerX - gridSize; x > 0; x -= gridSize * 5) {
      const meters = ((x - centerX) / adjustedScale).toFixed(1);
      ctx.fillText(`${meters}m`, x, centerY + 15);
    }

    // Y-axis labels
    for (let y = centerY + gridSize; y < height; y += gridSize * 5) {
      const meters = ((centerY - y) / adjustedScale).toFixed(1);
      ctx.fillText(`${meters}m`, centerX + 20, y);
    }

    for (let y = centerY - gridSize; y > 0; y -= gridSize * 5) {
      const meters = ((centerY - y) / adjustedScale).toFixed(1);
      ctx.fillText(`${meters}m`, centerX + 20, y);
    }
  };

  const drawVehicle = (ctx: CanvasRenderingContext2D, vehicle: Vehicle, centerX: number, centerY: number, scale: number) => {
    // Only draw active vehicles
    if (vehicle.active !== 1) return;

    // Convert scientific notation to regular numbers if needed
    const x = centerX + parseFloat(vehicle.x.toString()) * scale;
    const y = centerY - parseFloat(vehicle.y.toString()) * scale; // Invert y-axis as canvas y increases downward

    // Get vehicle dimensions
    const length = vehicle.half_length * 2 * scale;
    const width = vehicle.half_width * 2 * scale;

    // Save current transform
    ctx.save();

    // Move to vehicle position
    ctx.translate(x, y);

    // Calculate orientation angle from dx, dy values
    // Handle very small values that might be in scientific notation
    const dx = parseFloat(vehicle.dx.toString());
    const dy = parseFloat(vehicle.dy.toString());

    // Calculate angle, handling potential zero or near-zero values
    let angle = 0;
    if (Math.abs(dx) > 1e-10 || Math.abs(dy) > 1e-10) {
      angle = Math.atan2(dy, dx);
    }

    ctx.rotate(-angle); // Canvas rotation is clockwise, so we negate

    // Draw vehicle rectangle
    ctx.fillStyle = VEHICLE_COLOR;
    ctx.fillRect(-length / 2, -width / 2, length, width);

    // Draw direction indicator (front of vehicle)
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.moveTo(length / 2, 0);
    ctx.lineTo(length / 4, -width / 4);
    ctx.lineTo(length / 4, width / 4);
    ctx.closePath();
    ctx.fill();

    // Draw speed indicator
    ctx.fillStyle = 'white';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    const speedMph = (vehicle.speed * MS_TO_MPH).toFixed(1);
    ctx.fillText(`${speedMph} mph`, 0, 0);

    // Restore transform
    ctx.restore();
  };

  const drawPedestrian = (ctx: CanvasRenderingContext2D, pedestrian: Pedestrian, centerX: number, centerY: number, scale: number) => {
    // Only draw active pedestrians
    if (pedestrian.active !== 1) return;

    // Convert scientific notation to regular numbers if needed
    const x = centerX + parseFloat(pedestrian.x.toString()) * scale;
    const y = centerY - parseFloat(pedestrian.y.toString()) * scale; // Invert y-axis as canvas y increases downward
    const radius = 8; // Increased size of pedestrian dot for better visibility

    // Draw pedestrian circle with outline
    ctx.fillStyle = PEDESTRIAN_COLOR;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Add white outline for better visibility
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw a simple person icon inside the circle
    ctx.fillStyle = 'white';

    // Head
    ctx.beginPath();
    ctx.arc(x, y - radius / 3, radius / 3, 0, Math.PI * 2);
    ctx.fill();

    // Body
    ctx.beginPath();
    ctx.moveTo(x, y - radius / 4);
    ctx.lineTo(x, y + radius / 3);
    ctx.stroke();

    // Arms
    ctx.beginPath();
    ctx.moveTo(x - radius / 2, y);
    ctx.lineTo(x + radius / 2, y);
    ctx.stroke();

    // Legs
    ctx.beginPath();
    ctx.moveTo(x, y + radius / 3);
    ctx.lineTo(x - radius / 3, y + radius / 1.5);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y + radius / 3);
    ctx.lineTo(x + radius / 3, y + radius / 1.5);
    ctx.stroke();

    // Draw direction indicator with velocity scale
    // Handle very small values that might be in scientific notation
    const dx = parseFloat(pedestrian.dx.toString());
    const dy = parseFloat(pedestrian.dy.toString());

    const dirLength = pedestrian.speed * VELOCITY_MS_SCALE;

    // Only draw direction line if there's meaningful movement
    if (Math.abs(dx) > 1e-10 || Math.abs(dy) > 1e-10) {
      const dirX = x + dx * dirLength;
      const dirY = y - dy * dirLength;

      // Draw direction arrow
      ctx.strokeStyle = PEDESTRIAN_COLOR;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(dirX, dirY);
      ctx.stroke();

      // Draw arrowhead
      const headlen = 10; // length of arrow head in pixels
      const angle = Math.atan2(-dy, dx); // note we use -dy because canvas y is inverted

      ctx.beginPath();
      ctx.moveTo(dirX, dirY);
      ctx.lineTo(dirX - headlen * Math.cos(angle - Math.PI / 6), dirY - headlen * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(dirX - headlen * Math.cos(angle + Math.PI / 6), dirY - headlen * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fillStyle = PEDESTRIAN_COLOR;
      ctx.fill();
    }

    // Add speed label with background for better readability
    const speedMph = (pedestrian.speed * MS_TO_MPH).toFixed(1);
    const labelText = `${speedMph} mph`;

    // Draw text background
    ctx.font = 'bold 10px Arial';
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.fillRect(x - textWidth / 2 - 2, y - radius - 18, textWidth + 4, 14);

    // Draw text
    ctx.fillStyle = PEDESTRIAN_COLOR;
    ctx.textAlign = 'center';
    ctx.fillText(labelText, x, y - radius - 8);

    // Add position label if needed for debugging
    // ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    // ctx.fillText(`(${pedestrian.x.toFixed(2)}, ${pedestrian.y.toFixed(2)})`, x, y + radius + 12);
  };

  const drawEgoVehicle = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, scale: number, ego?: Ego) => {
    if (!ego) return;

    // Use ego dimensions
    const length = ego.half_length * 2 * scale;
    const width = ego.half_width * 2 * scale;
    const steeringAngle = ego.steering_angle || 0;

    // Save current transform
    ctx.save();

    // Move to ego vehicle position (center of canvas)
    ctx.translate(centerX, centerY);

    // Rotate based on steering angle if available
    if (steeringAngle) {
      const steeringRadians = steeringAngle * (Math.PI / 180); // Convert to radians
      ctx.rotate(steeringRadians);
    }

    // Draw ego vehicle rectangle
    ctx.fillStyle = EGO_COLOR;
    ctx.fillRect(-length / 2, -width / 2, length, width);

    // Draw direction indicator (front of vehicle)
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.moveTo(length / 2, 0);
    ctx.lineTo(length / 4, -width / 4);
    ctx.lineTo(length / 4, width / 4);
    ctx.closePath();
    ctx.fill();

    // Draw speed indicator if available
    if (ego.speed !== undefined) {
      ctx.fillStyle = 'white';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      const speedMph = (ego.speed * MS_TO_MPH).toFixed(1);
      ctx.fillText(`${speedMph} mph`, 0, 0);
    }

    // Restore transform
    ctx.restore();

    // Draw additional ego vehicle data as text outside the vehicle
    if (ego) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';

      const textX = centerX + length / 2 + 10;
      const textY = centerY - 40;
      const lineHeight = 16;

      ctx.fillText(`Acceleration: ${ego.accel?.toFixed(2) || 'N/A'}`, textX, textY);
      ctx.fillText(`Brake Pressure: ${ego.brake_pressure?.toFixed(2) || 'N/A'}`, textX, textY + lineHeight);
      ctx.fillText(`Steering: ${ego.steering_angle?.toFixed(2) || 'N/A'}°`, textX, textY + lineHeight * 2);
      ctx.fillText(`Speed: ${(ego.speed * MS_TO_MPH).toFixed(1) || 'N/A'} mph`, textX, textY + lineHeight * 3);
    }
  };

  const drawRoutes = (ctx: CanvasRenderingContext2D, routes: Route[], centerX: number, centerY: number, scale: number) => {
    if (routes.length === 0) return;

    // Draw route points
    routes.forEach(route => {
      const x = centerX + route.x * scale;
      const y = centerY - route.y * scale; // Invert y-axis as canvas y increases downward

      // Draw route point
      ctx.fillStyle = ROUTE_COLOR;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Draw road width indicator if needed
      if (route.road_width0 > 0 || route.road_width1 > 0) {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.3)'; // Amber with transparency
        ctx.lineWidth = 1;

        // Calculate perpendicular direction to tangent
        const perpX = -route.tangent_dy;
        const perpY = route.tangent_dx;

        // Draw road width lines
        const width0 = route.road_width0 * scale;
        const width1 = route.road_width1 * scale;

        ctx.beginPath();
        ctx.moveTo(x + perpX * width0, y - perpY * width0);
        ctx.lineTo(x - perpX * width1, y + perpY * width1);
        ctx.stroke();
      }

      // Indicate if this is a junction or roundabout
      if (route.has_junction === 1) {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (route.is_roundabout === 1) {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    // Connect route points with lines
    ctx.strokeStyle = ROUTE_LINE_COLOR;
    ctx.lineWidth = 2;
    ctx.beginPath();

    // Move to first point
    const firstX = centerX + routes[0].x * scale;
    const firstY = centerY - routes[0].y * scale;
    ctx.moveTo(firstX, firstY);

    // Draw lines to subsequent points
    for (let i = 1; i < routes.length; i++) {
      const x = centerX + routes[i].x * scale;
      const y = centerY - routes[i].y * scale;
      ctx.lineTo(x, y);
    }

    ctx.stroke();
  };

  return (
    <div className="border rounded-md p-4 bg-white">
      <div className="flex justify-between mb-4">
        <div className="text-sm text-gray-500">
          <span className="font-medium">Scale:</span> {(METRES_M_SCALE * zoom).toFixed(1)} pixels per meter
        </div>
        <div className="text-sm text-gray-500">
          <span className="font-medium">Velocity Scale:</span> {VELOCITY_MS_SCALE}x
        </div>
        <div className="text-sm text-gray-500">
          <span className="font-medium">Zoom:</span> {zoom.toFixed(1)}x
        </div>
        <button
          onClick={resetView}
          className="px-2 py-1 bg-gray-200 text-xs rounded hover:bg-gray-300"
        >
          Fit All Data
        </button>
      </div>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border rounded-md cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        />

      {/* <div className="absolute bottom-2 right-2 bg-white bg-opacity-70 p-1 rounded text-xs">
        Drag to pan • Scroll to zoom
      </div> */}
      </div>
      <div className="flex justify-between mt-4 text-sm text-gray-500">
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-green-500 rounded-full"></span>
          <span>Ego Vehicle</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-blue-500 rounded-full"></span>
          <span>Vehicles</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-red-500 rounded-full"></span>
          <span>Pedestrians</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-amber-500 rounded-full"></span>
          <span>Route Points</span>
        </div>
      </div>
      {scene?.ego && (
        <div className="mt-4 p-3 bg-gray-50 rounded-md border">
          <h3 className="font-medium text-sm mb-2">Ego Vehicle Data</h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div><span className="font-medium">Acceleration:</span> {scene.ego.accel?.toFixed(2) || 'N/A'}</div>
            <div><span className="font-medium">Speed:</span> {scene.ego.speed ? `${(scene.ego.speed * MS_TO_MPH).toFixed(1)} mph` : 'N/A'}</div>
            <div><span className="font-medium">Brake Pressure:</span> {scene.ego.brake_pressure?.toFixed(2) || 'N/A'}</div>
            <div><span className="font-medium">Steering Angle:</span> {scene.ego.steering_angle?.toFixed(2) || 'N/A'}°</div>
            <div><span className="font-medium">Dimensions:</span> {scene.ego.half_length ? `${(scene.ego.half_length * 2).toFixed(2)}m × ${(scene.ego.half_width * 2).toFixed(2)}m` : 'N/A'}</div>
            <div><span className="font-medium">Height:</span> {scene.ego.half_height ? `${(scene.ego.half_height * 2).toFixed(2)}m` : 'N/A'}</div>
          </div>
        </div>
      )}
      {scene?.situation && (
        <div className="mt-2 text-xs text-gray-500">
          <span className="font-medium">Collection:</span> {scene.situation.collection}
        </div>
      )}
    </div>
  );
} 
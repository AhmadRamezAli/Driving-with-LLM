import React from "react";
import { motion } from "framer-motion";

interface AnimatedRoadProps {
  className?: string;
  height?: number;
  speed?: number;
}

const AnimatedRoad: React.FC<AnimatedRoadProps> = ({
  className = "",
  height = 50,
  speed = 2
}) => {
  // Generate lane markings
  const laneMarkings = Array.from({ length: 20 }).map((_, index) => ({
    id: index,
    left: `${index * 10}%`,
    width: "8%",
    delay: index * 0.1
  }));

  return (
    <div 
      className={`w-full bg-gray-800 relative overflow-hidden ${className}`} 
      style={{ height: `${height}px` }}
    >
      {/* Road surface */}
      <div className="absolute inset-0 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900" />
      
      {/* Lane markings */}
      {laneMarkings.map((marking) => (
        <motion.div
          key={marking.id}
          className="absolute h-[10px] bg-yellow-400"
          style={{
            left: marking.left,
            width: marking.width,
            top: "calc(50% - 5px)",
          }}
          initial={{ opacity: 0.7, x: "100vw" }}
          animate={{ x: "-100vw" }}
          transition={{
            duration: speed,
            repeat: Infinity,
            ease: "linear",
            delay: marking.delay
          }}
        />
      ))}
      
      {/* Road shine effect */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent to-white/5" />
      
      {/* Road edges */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gray-600" />
      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-gray-600" />
    </div>
  );
};

export default AnimatedRoad; 
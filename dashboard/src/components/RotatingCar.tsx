import React, { useState } from "react";
import { motion } from "framer-motion";
import { Car } from "lucide-react";

interface RotatingCarProps {
  className?: string;
  size?: number;
  color?: string;
  glowColor?: string;
  speed?: number;
}

const RotatingCar: React.FC<RotatingCarProps> = ({
  className = "",
  size = 80,
  color = "text-primary",
  glowColor = "bg-electric-cyan/30",
  speed = 10
}) => {
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <div 
      className={`relative ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Glow effect */}
      <motion.div
        className={`absolute inset-0 rounded-full ${glowColor} blur-xl`}
        initial={{ scale: 0.8, opacity: 0.5 }}
        animate={{ 
          scale: isHovered ? 1.2 : 1,
          opacity: isHovered ? 0.8 : 0.5
        }}
        transition={{ duration: 0.5 }}
      />
      
      {/* 3D rotating container with perspective */}
      <motion.div
        className="relative w-full h-full flex items-center justify-center"
        style={{ perspective: 800 }}
        animate={{ 
          rotateY: [0, 360], 
        }}
        transition={{ 
          duration: speed,
          repeat: Infinity,
          ease: "linear"
        }}
      >
        {/* Car icon */}
        <motion.div
          className={`${color}`}
          animate={{ 
            rotateX: isHovered ? [0, 10, -10, 0] : 0,
            rotateZ: isHovered ? [0, -5, 5, 0] : 0,
            scale: isHovered ? 1.2 : 1
          }}
          transition={{ 
            duration: isHovered ? 2 : 0.5,
            repeat: isHovered ? Infinity : 0,
            ease: "easeInOut"
          }}
        >
          <Car size={size} strokeWidth={1.5} />
        </motion.div>
      </motion.div>
      
      {/* Shadow */}
      <motion.div
        className="absolute bottom-0 left-1/2 -translate-x-1/2 bg-black/10 rounded-full blur-sm"
        style={{ width: size * 0.8, height: size * 0.2 }}
        animate={{ 
          width: isHovered ? size * 0.9 : size * 0.8,
          opacity: isHovered ? 0.2 : 0.1
        }}
        transition={{ duration: 0.5 }}
      />
    </div>
  );
};

export default RotatingCar; 
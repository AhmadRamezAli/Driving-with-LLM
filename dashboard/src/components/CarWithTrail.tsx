import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Car } from "lucide-react";

interface CarWithTrailProps {
  className?: string;
  pathType?: "horizontal" | "wave" | "zigzag";
  carColor?: string;
  trailColor?: string;
  speed?: number;
}

const CarWithTrail: React.FC<CarWithTrailProps> = ({
  className = "",
  pathType = "wave",
  carColor = "text-primary",
  trailColor = "bg-primary/10",
  speed = 15
}) => {
  const [trail, setTrail] = useState<{x: number, y: number, id: number}[]>([]);
  const trailLength = 15;
  const maxTrailSize = 15;
  const minTrailSize = 5;
  
  // Generate animation path based on type
  const getPath = () => {
    const screenWidth = typeof window !== 'undefined' ? window.innerWidth : 1000;
    const amplitude = 100; // Height of wave/zigzag
    
    switch(pathType) {
      case "horizontal":
        return {
          x: [-100, screenWidth + 100],
          y: [0, 0]
        };
      case "wave":
        return {
          x: [-100, screenWidth + 100],
          y: [0, amplitude, 0, -amplitude, 0]
        };
      case "zigzag":
        return {
          x: [-100, screenWidth/4, screenWidth/2, screenWidth*3/4, screenWidth + 100],
          y: [0, amplitude, 0, amplitude, 0]
        };
      default:
        return {
          x: [-100, screenWidth + 100],
          y: [0, 0]
        };
    }
  };

  // Update trail positions
  useEffect(() => {
    const interval = setInterval(() => {
      const carElement = document.getElementById("animated-car");
      if (carElement) {
        const rect = carElement.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        setTrail(prev => {
          const newTrail = [{ x: centerX, y: centerY, id: Date.now() }, ...prev];
          if (newTrail.length > trailLength) {
            return newTrail.slice(0, trailLength);
          }
          return newTrail;
        });
      }
    }, 100);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`fixed inset-0 pointer-events-none overflow-hidden ${className}`}>
      {/* Trail particles */}
      {trail.map((particle, index) => {
        // Calculate size based on position in trail
        const size = maxTrailSize - ((maxTrailSize - minTrailSize) * (index / trailLength));
        const opacity = 1 - (index / trailLength);
        
        return (
          <motion.div
            key={particle.id}
            className={`absolute rounded-full ${trailColor}`}
            style={{
              left: particle.x,
              top: particle.y,
              width: size,
              height: size,
              opacity,
              transform: 'translate(-50%, -50%)'
            }}
            initial={{ scale: 1 }}
            animate={{ scale: 0 }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />
        );
      })}
      
      {/* Car */}
      <motion.div
        id="animated-car"
        className="absolute"
        style={{ top: '50%', left: 0 }}
        animate={getPath()}
        transition={{
          duration: speed,
          repeat: Infinity,
          repeatType: "loop",
          ease: pathType === "zigzag" ? "easeInOut" : "linear",
        }}
      >
        <motion.div 
          className={carColor}
          animate={{ rotate: pathType === "horizontal" ? 0 : [-5, 5, -5] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <Car size={32} />
        </motion.div>
      </motion.div>
    </div>
  );
};

export default CarWithTrail; 
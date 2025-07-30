import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface ParallaxBackgroundProps {
  className?: string;
}

const ParallaxBackground: React.FC<ParallaxBackgroundProps> = ({ className = "" }) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: e.clientX / window.innerWidth - 0.5,
        y: e.clientY / window.innerHeight - 0.5,
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  const shapes = [
    {
      color: "bg-electric-cyan/10",
      size: "w-64 h-64",
      position: "top-[-10%] right-[20%]",
      blur: "blur-3xl",
      factor: { x: 20, y: -20 },
    },
    {
      color: "bg-primary/10",
      size: "w-96 h-96",
      position: "bottom-[-20%] left-[10%]",
      blur: "blur-3xl",
      factor: { x: -15, y: 15 },
    },
    {
      color: "bg-automotive-blue/5",
      size: "w-72 h-72",
      position: "top-[20%] left-[-10%]",
      blur: "blur-2xl",
      factor: { x: -10, y: -10 },
    },
    {
      color: "bg-electric-cyan-light/5",
      size: "w-48 h-48",
      position: "bottom-[10%] right-[-5%]",
      blur: "blur-2xl",
      factor: { x: 25, y: 25 },
    },
  ];

  return (
    <div className={`fixed inset-0 overflow-hidden pointer-events-none ${className}`}>
      {shapes.map((shape, index) => (
        <motion.div
          key={index}
          className={`absolute rounded-full ${shape.color} ${shape.size} ${shape.position} ${shape.blur}`}
          animate={{
            x: mousePosition.x * shape.factor.x,
            y: mousePosition.y * shape.factor.y,
          }}
          transition={{
            type: "spring",
            stiffness: 50,
            damping: 30,
            mass: 1,
          }}
        />
      ))}
    </div>
  );
};

export default ParallaxBackground; 
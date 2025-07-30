import React from "react";
import { motion } from "framer-motion";
import { Car, CarFront, CarTaxiFront, Truck } from "lucide-react";

interface FloatingCarsProps {
  count?: number;
  className?: string;
}

const FloatingCars: React.FC<FloatingCarsProps> = ({ 
  count = 6,
  className = "" 
}) => {
  // Different car components to use
  const carComponents = [
    <Car size={24} />,
    <CarFront size={24} />,
    <CarTaxiFront size={24} />,
    <Truck size={24} />
  ];
  
  // Generate random positions and animation parameters for each car
  const cars = Array.from({ length: count }).map((_, i) => {
    const size = Math.floor(Math.random() * 16) + 16; // Random size between 16-32px
    const top = Math.floor(Math.random() * 80) + 10; // Random top position 10-90%
    const left = Math.floor(Math.random() * 80) + 10; // Random left position 10-90%
    const duration = Math.random() * 20 + 15; // Random duration between 15-35s
    const delay = Math.random() * 5; // Random delay 0-5s
    const carIndex = i % carComponents.length;
    const opacity = Math.random() * 0.5 + 0.2; // Random opacity between 0.2-0.7
    
    // Random path parameters
    const pathRadius = Math.random() * 50 + 30; // Path radius 30-80px
    const pathType = Math.floor(Math.random() * 3); // 0: circle, 1: figure-8, 2: wave
    
    return {
      id: i,
      size,
      top: `${top}%`,
      left: `${left}%`,
      duration,
      delay,
      carIndex,
      opacity,
      pathRadius,
      pathType
    };
  });

  // Get animation path based on path type
  const getAnimationPath = (pathType: number, radius: number) => {
    switch(pathType) {
      case 0: // Circle
        return {
          x: [0, radius, 0, -radius, 0],
          y: [0, -radius, 0, radius, 0],
        };
      case 1: // Figure-8
        return {
          x: [0, radius, 0, -radius, 0],
          y: [0, -radius/2, 0, -radius/2, 0],
          rotate: [0, 90, 180, 270, 360]
        };
      case 2: // Wave
        return {
          x: [0, radius, 2*radius, 3*radius, 4*radius],
          y: [0, radius, 0, radius, 0],
          rotate: [0, 10, 0, -10, 0]
        };
      default:
        return {
          x: [0, radius, 0, -radius, 0],
          y: [0, -radius, 0, radius, 0],
        };
    }
  };

  return (
    <div className={`fixed inset-0 pointer-events-none overflow-hidden ${className}`}>
      {cars.map((car) => (
        <motion.div
          key={car.id}
          className="absolute"
          style={{
            top: car.top,
            left: car.left,
          }}
          initial={{ opacity: 0 }}
          animate={{
            opacity: car.opacity,
            ...getAnimationPath(car.pathType, car.pathRadius)
          }}
          transition={{
            duration: car.duration,
            repeat: Infinity,
            repeatType: "loop",
            ease: "linear",
            delay: car.delay
          }}
        >
          <motion.div 
            className="text-primary"
            animate={{
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
            }}
          >
            {React.cloneElement(carComponents[car.carIndex], { size: car.size })}
          </motion.div>
        </motion.div>
      ))}
    </div>
  );
};

export default FloatingCars; 
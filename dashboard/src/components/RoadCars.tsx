import React from "react";
import { motion } from "framer-motion";
import { Car, CarFront, CarTaxiFront, Truck, Ambulance, Bus } from "lucide-react";

interface RoadCarsProps {
  className?: string;
  count?: number;
}

const RoadCars: React.FC<RoadCarsProps> = ({
  className = "",
  count = 6
}) => {
  // Different car components to use
  const carComponents = [
    { icon: <Car size={24} />, color: "text-blue-500" },
    { icon: <CarFront size={24} />, color: "text-red-500" },
    { icon: <CarTaxiFront size={24} />, color: "text-yellow-500" },
    { icon: <Truck size={24} />, color: "text-green-500" },
    { icon: <Ambulance size={24} />, color: "text-white" },
    { icon: <Bus size={24} />, color: "text-orange-500" }
  ];
  
  // Generate cars with random properties
  const cars = Array.from({ length: count }).map((_, index) => {
    const carType = index % carComponents.length;
    const speed = Math.random() * 3 + 2; // Random speed between 2-5s
    const delay = Math.random() * 2; // Random delay 0-2s
    const lane = Math.random() > 0.5 ? "top" : "bottom"; // Two lanes
    const direction = Math.random() > 0.5 ? "left" : "right"; // Two directions
    const size = Math.floor(Math.random() * 10) + 20; // Random size between 20-30px
    
    return {
      id: index,
      carType,
      speed,
      delay,
      lane,
      direction,
      size
    };
  });

  return (
    <div className={`relative ${className}`}>
      {cars.map((car) => {
        const { id, carType, speed, delay, lane, direction, size } = car;
        const carComponent = carComponents[carType];
        const isLeftToRight = direction === "right";
        
        return (
          <motion.div
            key={id}
            className={`absolute ${carComponent.color}`}
            style={{
              top: lane === "top" ? "25%" : "75%",
              left: isLeftToRight ? "-50px" : "calc(100% + 50px)",
              transform: `translateY(-50%) ${isLeftToRight ? "" : "scaleX(-1)"}`,
            }}
            initial={{ x: isLeftToRight ? "-50px" : "calc(100% + 50px)" }}
            animate={{ 
              x: isLeftToRight ? "calc(100% + 50px)" : "-50px"
            }}
            transition={{
              duration: speed,
              repeat: Infinity,
              delay,
              ease: "linear"
            }}
          >
            {React.cloneElement(carComponent.icon, { size })}
          </motion.div>
        );
      })}
    </div>
  );
};

export default RoadCars; 
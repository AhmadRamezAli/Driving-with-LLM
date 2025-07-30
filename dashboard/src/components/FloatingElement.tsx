import { motion } from "framer-motion";
import React from "react";

interface FloatingElementProps {
  children: React.ReactNode;
  className?: string;
  amplitude?: number; // How far it moves
  duration?: number;  // How long each cycle takes
  delay?: number;     // Delay before animation starts
}

export const FloatingElement: React.FC<FloatingElementProps> = ({
  children,
  className = "",
  amplitude = 10,
  duration = 4,
  delay = 0
}) => {
  return (
    <motion.div
      className={className}
      animate={{
        y: [`${-amplitude}px`, `${amplitude}px`, `${-amplitude}px`],
      }}
      transition={{
        duration,
        ease: "easeInOut",
        repeat: Infinity,
        delay,
        repeatDelay: 0,
      }}
    >
      {children}
    </motion.div>
  );
};

export default FloatingElement; 
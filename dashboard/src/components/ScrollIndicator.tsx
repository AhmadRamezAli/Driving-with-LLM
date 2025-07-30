import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";

interface ScrollIndicatorProps {
  className?: string;
}

const ScrollIndicator = ({ className = "" }: ScrollIndicatorProps) => {
  return (
    <motion.div
      className={`flex flex-col items-center justify-center ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 1.5, duration: 0.5 }}
    >
      <motion.span 
        className="text-sm text-muted-foreground mb-2 font-medium"
      >
        Scroll Down
      </motion.span>
      <motion.div
        animate={{
          y: [0, 10, 0],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          repeatType: "loop",
          ease: "easeInOut",
        }}
        className="flex items-center justify-center"
      >
        <ChevronDown className="h-6 w-6 text-primary" />
      </motion.div>
    </motion.div>
  );
};

export default ScrollIndicator; 
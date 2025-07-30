import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, BarChart3, Clock, PieChart, TrendingUp, Zap, Car, Volume2, VolumeX } from "lucide-react";
import { Link } from "react-router-dom";
import heroImage from "@/assets/hero-image.jpg";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import FloatingElement from "@/components/FloatingElement";
import ScrollIndicator from "@/components/ScrollIndicator";
import ParallaxBackground from "@/components/ParallaxBackground";
import FloatingCars from "@/components/FloatingCars";
import CarWithTrail from "@/components/CarWithTrail";
import RotatingCar from "@/components/RotatingCar";
import AnimatedRoad from "@/components/AnimatedRoad";
import RoadCars from "@/components/RoadCars";
import { useEffect, useRef, useState } from "react";
import carStartSound from "@/assets/Sports Car Start Gear Grind - QuickSounds.com.mp3";

const fadeIn = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.6 } }
};

const slideUp = {
  hidden: { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { duration: 0.6 } }
};

const staggerChildren = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.3
    }
  }
};

const Index = () => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const hasScrolledRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const playSound = () => {
    if (audioRef.current && !isPlaying) {
      audioRef.current.currentTime = 0;
      audioRef.current.play()
        .then(() => {
          setIsPlaying(true);
          console.log("Audio playback started successfully");

          // Stop after 10 seconds
          setTimeout(() => {
            if (audioRef.current) {
              audioRef.current.pause();
              audioRef.current.currentTime = 0;
              setIsPlaying(false);
            }
          }, 5000);
        })
        .catch(error => {
          console.error("Audio playback failed:", error);
        });
    }
  };

  const stopSound = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  // Add scroll event listener to play sound on scroll
  useEffect(() => {
    const handleScroll = () => {
      if (!hasScrolledRef.current) {
        hasScrolledRef.current = true;
        playSound();
      }
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      stopSound();
    };
  }, []); // Empty dependency array since we only want to set up the listener once

  return (
    <div className="min-h-screen bg-gradient-hero"
      onClick={isPlaying ? stopSound : playSound}>
      {/* Audio element */}
      <audio ref={audioRef} src={carStartSound} preload="auto" />

      {/* Sound control button */}
      {/* <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        onClick={isPlaying ? stopSound : playSound}
        className="fixed top-24 right-6 z-50 bg-card/80 backdrop-blur-md p-2 rounded-full shadow-card hover:shadow-floating transition-all duration-300"
        title={isPlaying ? "Mute engine sound" : "Play engine sound"}
      >
        {isPlaying ? (
          <VolumeX className="w-6 h-6 text-primary" />
        ) : (
          <Volume2 className="w-6 h-6 text-primary" />
        )}
      </motion.button> */}

      {/* Parallax Background */}
      <ParallaxBackground />

      {/* Floating Cars */}
      <FloatingCars count={8} />

      {/* Car with trail - horizontal at top */}
      <CarWithTrail
        pathType="horizontal"
        className="top-[15%] -translate-y-1/2"
        speed={20}
        trailColor="bg-electric-cyan/20"
      />

      {/* Car with trail - wave at bottom */}
      <CarWithTrail
        pathType="wave"
        className="top-[85%] -translate-y-1/2"
        speed={25}
      />

      {/* Navigation */}
      <motion.nav
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-card/80 backdrop-blur-md border-b shadow-card sticky top-0 z-50"
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="flex items-center gap-2"
            >
              <Car className="w-8 h-8 text-primary" />
              <span className="text-xl font-bold text-foreground">
                DriveTime Analytics
              </span>
            </motion.div>
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
            >
              <Link to="/dashboard">
                <Button className="bg-gradient-primary hover:shadow-glow transition-all duration-300">
                  View Dashboard
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </motion.div>
          </div>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden">
        <div className="container mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={staggerChildren}
              className="space-y-8"
            >
              <motion.div variants={slideUp} className="space-y-4">
                <div className="flex items-center gap-4 mb-4">
                  <RotatingCar size={60} className="w-16 h-16" />
                  <h1 className="text-5xl lg:text-6xl font-bold text-foreground leading-tight">
                    <AnimatedText text="Self-Driving Car" className="block" delay={0.5} />
                    <motion.span
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.7, duration: 0.6 }}
                      className="block bg-gradient-primary bg-clip-text text-transparent"
                    >
                      <AnimatedText text="Analytics Platform" delay={1.2} />
                    </motion.span>
                  </h1>
                </div>
                <motion.p variants={slideUp} className="text-xl text-muted-foreground leading-relaxed">
                  Monitor, analyze, and optimize your autonomous vehicle performance
                  with real-time insights and comprehensive analytics dashboard.
                </motion.p>
              </motion.div>

              <motion.div variants={slideUp} className="flex gap-4">
                <FloatingElement amplitude={3} duration={2.5}>
                  <Link to="/dashboard">
                    <Button size="lg" className="bg-gradient-primary hover:shadow-glow transition-all duration-300">
                      Launch Dashboard
                      <ArrowRight className="w-5 h-5 ml-2" />
                    </Button>
                  </Link>
                </FloatingElement>
                <Button size="lg" variant="outline" className="border-2">
                  Learn More
                </Button>
              </motion.div>

              <motion.div
                variants={{
                  hidden: { opacity: 0 },
                  visible: {
                    opacity: 1,
                    transition: {
                      staggerChildren: 0.1,
                      delayChildren: 0.8
                    }
                  }
                }}
                className="grid grid-cols-3 gap-4 pt-8"
              >
                <motion.div
                  variants={fadeIn}
                  className="text-center"
                >
                  <div className="text-2xl font-bold text-primary">99.9%</div>
                  <div className="text-sm text-muted-foreground">Uptime</div>
                </motion.div>
                <motion.div
                  variants={fadeIn}
                  className="text-center"
                >
                  <div className="text-2xl font-bold text-primary">24/7</div>
                  <div className="text-sm text-muted-foreground">Monitoring</div>
                </motion.div>
                <motion.div
                  variants={fadeIn}
                  className="text-center"
                >
                  <div className="text-2xl font-bold text-primary">Real-time</div>
                  <div className="text-sm text-muted-foreground">Analytics</div>
                </motion.div>
              </motion.div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                duration: 0.8,
                ease: [0.22, 1, 0.36, 1]
              }}
              className="relative"
            >
              <FloatingElement amplitude={5} duration={6}>
                <div className="relative rounded-2xl overflow-hidden shadow-floating"
                >
                  <img
                    src={heroImage}
                    alt="Self-driving car analytics dashboard"
                    className="w-full h-auto"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-primary/20 to-transparent" />
                </div>
              </FloatingElement>
              <motion.div
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.1, 0.2, 0.1]
                }}
                transition={{
                  duration: 8,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="absolute -top-4 -right-4 w-24 h-24 bg-electric-cyan rounded-full blur-3xl opacity-20"
              />
              <motion.div
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.05, 0.1, 0.05]
                }}
                transition={{
                  duration: 10,
                  repeat: Infinity,
                  delay: 4,
                  ease: "easeInOut"
                }}
                className="absolute -bottom-4 -left-4 w-32 h-32 bg-primary rounded-full blur-3xl opacity-20"
              />
            </motion.div>
          </div>

          <motion.div className="absolute bottom-5 left-1/2 -translate-x-1/2">
            <ScrollIndicator />
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-dashboard-bg">
        <div className="container mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={slideUp}
            className="text-center mb-16"
          >
            <div className="flex items-center justify-center gap-4 mb-4">
              <RotatingCar size={40} className="w-12 h-12" speed={15} />
              <h2 className="text-4xl font-bold text-foreground">
                <AnimatedText text="Comprehensive Analytics Suite" />
              </h2>
              <RotatingCar size={40} className="w-12 h-12" speed={15} />
            </div>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Everything you need to monitor and optimize your self-driving car's performance
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={{
              hidden: { opacity: 0 },
              visible: {
                opacity: 1,
                transition: { staggerChildren: 0.15 }
              }
            }}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {[
              { icon: <Clock className="w-6 h-6 text-primary-foreground" />, title: "Time Analysis", description: "Monitor prediction execution times and performance metrics" },
              { icon: <PieChart className="w-6 h-6 text-primary-foreground" />, title: "Distribution Charts", description: "Visualize performance distribution across time segments" },
              { icon: <TrendingUp className="w-6 h-6 text-primary-foreground" />, title: "Performance Trends", description: "Track improvements and identify optimization opportunities" },
              { icon: <BarChart3 className="w-6 h-6 text-primary-foreground" />, title: "Custom Reports", description: "Generate detailed reports and export analytics data" }
            ].map((feature, index) => (
              <motion.div
                key={index}
                variants={{
                  hidden: { y: 20, opacity: 0 },
                  visible: { y: 0, opacity: 1, transition: { duration: 0.5 } }
                }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
              >
                <Card className="bg-card shadow-card hover:shadow-floating transition-all duration-300 group">
                  <CardHeader className="text-center">
                    <motion.div
                      whileHover={{ scale: 1.1, rotate: 5 }}
                      transition={{ type: "spring", stiffness: 400, damping: 10 }}
                      className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300"
                    >
                      {feature.icon}
                    </motion.div>
                    <CardTitle>{feature.title}</CardTitle>
                    <CardDescription>
                      {feature.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-primary relative overflow-hidden">
        <div className="absolute inset-0 bg-primary/10" />
        <div className="container mx-auto px-6 relative">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerChildren}
            className="text-center space-y-8"
          >
            <motion.h2
              variants={fadeIn}
              className="text-4xl font-bold text-primary-foreground"
            >
              <AnimatedText text="Ready to Optimize Your Fleet?" />
            </motion.h2>
            <motion.p
              variants={fadeIn}
              className="text-xl text-primary-foreground/80 max-w-2xl mx-auto"
            >
              Start monitoring your self-driving car performance with our advanced analytics platform
            </motion.p>
            <motion.div
              variants={slideUp}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link to="/dashboard">
                <Button
                  size="lg"
                  variant="secondary"
                  className="bg-card text-primary hover:bg-card/90 shadow-floating hover:shadow-glow transition-all duration-300"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  Get Started Now
                </Button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.1, 0.2, 0.1]
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute top-0 left-0 w-64 h-64 bg-electric-cyan rounded-full blur-3xl opacity-10"
        />
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.05, 0.1, 0.05]
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            delay: 4,
            ease: "easeInOut"
          }}
          className="absolute bottom-0 right-0 w-96 h-96 bg-primary-foreground rounded-full blur-3xl opacity-5"
        />
      </section>

      {/* Animated Road with Cars */}
      <div className="relative">
        <AnimatedRoad height={70} speed={3} />
        <RoadCars count={10} className="absolute inset-0" />
      </div>

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="bg-card border-t py-12"
      >
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Car className="w-6 h-6 text-primary" />
              <span className="text-lg font-semibold text-foreground">
                DriveTime Analytics
              </span>
            </div>
            <p className="text-muted-foreground">
              © 2024 DriveTime Analytics. Advanced self-driving car monitoring.
            </p>
          </div>
        </div>
      </motion.footer>
    </div>
  );
};

export default Index;
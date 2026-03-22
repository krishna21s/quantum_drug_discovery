import React, { useRef, useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface MagicCardProps extends React.HTMLAttributes<HTMLDivElement> {
  gradientSize?: number;
  gradientColor?: string;
  gradientOpacity?: number;
  containerClassName?: string;
  children: React.ReactNode;
}

export function MagicCard({
  children,
  className,
  containerClassName,
  gradientSize = 400,
  gradientColor = "hsl(var(--primary) / 0.15)",
  gradientOpacity = 1,
  ...props
}: MagicCardProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const updateMousePosition = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    };

    if (isHovered) {
      window.addEventListener("mousemove", updateMousePosition);
    }
    return () => window.removeEventListener("mousemove", updateMousePosition);
  }, [isHovered]);

  return (
    <div
      ref={containerRef}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={cn(
        "group relative flex size-full overflow-hidden rounded-3xl bg-neutral-100/50 dark:bg-neutral-900/40 border border-neutral-200/50 dark:border-white/10 backdrop-blur-xl transition-colors duration-300",
        containerClassName
      )}
      {...props}
    >
      <div className={cn("relative z-10 size-full w-full", className)}>
        {children}
      </div>
      <motion.div
        animate={{
          x: mousePosition.x - gradientSize / 2,
          y: mousePosition.y - gradientSize / 2,
          opacity: isHovered ? gradientOpacity : 0,
        }}
        transition={{ type: "tween", ease: "backOut", duration: 0.4 }}
        className="pointer-events-none absolute left-0 top-0 z-0 rounded-full blur-[80px]"
        style={{
          width: gradientSize,
          height: gradientSize,
          background: gradientColor,
        }}
      />
    </div>
  );
}

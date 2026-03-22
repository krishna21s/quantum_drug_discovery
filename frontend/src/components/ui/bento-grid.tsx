import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { ReactNode } from "react";

export function BentoGrid({
  className,
  children,
}: {
  className?: string;
  children?: ReactNode;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-1 md:grid-cols-3 gap-4 max-w-7xl mx-auto",
        className
      )}
    >
      {children}
    </div>
  );
}

export function BentoGridItem({
  className,
  title,
  description,
  header,
  icon,
  children,
  colSpan = 1,
  rowSpan = 1,
}: {
  className?: string;
  title?: string | ReactNode;
  description?: string | ReactNode;
  header?: ReactNode;
  icon?: ReactNode;
  children?: ReactNode;
  colSpan?: number;
  rowSpan?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.4, 0.25, 1] }}
      className={cn(
        "group/bento relative overflow-hidden rounded-3xl",
        "bg-white/40 dark:bg-gradient-to-br dark:from-[rgba(15,20,40,0.6)] dark:to-[rgba(8,12,28,0.8)]",
        "border border-black/5 dark:border-white/[0.06]",
        "backdrop-blur-xl",
        "shadow-lg dark:shadow-[0_8px_32px_-8px_rgba(0,0,0,0.5)]",
        "transition-all duration-500 ease-out",
        "hover:border-black/10 dark:hover:border-white/[0.12]",
        "hover:shadow-xl dark:hover:shadow-[0_16px_48px_-12px_rgba(0,0,0,0.7)]",
        "hover:-translate-y-1",
        colSpan === 2 && "md:col-span-2",
        colSpan === 3 && "md:col-span-3",
        rowSpan === 2 && "md:row-span-2",
        className
      )}
    >
      {/* Animated top glow line */}
      <div className="absolute top-0 left-8 right-8 h-[1px] bg-gradient-to-r from-transparent via-primary/40 to-transparent opacity-60 group-hover/bento:opacity-100 transition-opacity duration-500" />

      {/* Inner highlight */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.03] via-transparent to-transparent pointer-events-none" />

      {/* Content */}
      <div className="relative z-10 h-full flex flex-col p-5">
        {header && <div className="mb-4">{header}</div>}
        {children}
        <div className="mt-auto">
          {icon && (
            <div className="mb-2 text-primary/70 group-hover/bento:text-primary transition-colors duration-300">
              {icon}
            </div>
          )}
          {title && (
            <div className="font-semibold text-sm text-foreground/90 group-hover/bento:text-foreground transition-colors duration-300">
              {title}
            </div>
          )}
          {description && (
            <div className="text-xs text-muted-foreground mt-1 leading-relaxed">
              {description}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

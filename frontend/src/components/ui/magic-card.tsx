import React from "react";
import { cn } from "@/lib/utils";

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
  ...props
}: MagicCardProps) {
  return (
    <div
      className={cn(
        "group relative flex size-full overflow-hidden rounded-3xl bg-neutral-100/50 dark:bg-neutral-900/40 border border-neutral-200/50 dark:border-white/10 backdrop-blur-xl",
        containerClassName
      )}
      {...props}
    >
      <div className={cn("relative z-10 size-full w-full", className)}>
        {children}
      </div>
    </div>
  );
}

import React from 'react';
import { motion } from 'framer-motion';

export function LiquidBackground() {
  return (
    <div className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none bg-background">
      {/* Base Abstract Image (premium liquid abstract) */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-30 mix-blend-screen"
        style={{ backgroundImage: "url('https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop')" }}
      />
      
      {/* Floating abstract gradients */}
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          x: [0, 100, 0],
          y: [0, -50, 0],
        }}
        transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
        className="absolute top-[10%] left-[20%] w-[500px] h-[500px] bg-primary/20 rounded-full blur-[120px]"
      />
      
      <motion.div
        animate={{
          scale: [1, 1.3, 1],
          x: [0, -80, 0],
          y: [0, 100, 0],
        }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        className="absolute bottom-[10%] right-[10%] w-[600px] h-[600px] bg-quantum/20 rounded-full blur-[140px]"
      />
      
      <motion.div
        animate={{
          scale: [1, 1.5, 1],
          x: [0, 50, 0],
          y: [0, 80, 0],
        }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut", delay: 5 }}
        className="absolute top-[40%] left-[50%] w-[400px] h-[400px] bg-accent/15 rounded-full blur-[100px]"
      />

      {/* Subtle Noise for texture realism */}
      <div 
        className="absolute inset-0 opacity-[0.02] mix-blend-overlay pointer-events-none" 
        style={{ backgroundImage: "url('data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjAwIDIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZmlsdGVyIGlkPSJub2lzZUZpbHRlciI+CiAgICA8ZmVUdXJidWxlbmNlIHR5cGU9ImZyYWN0YWxOb2lzZSIgYmFzZUZyZXF1ZW5jeT0iMC42NSIgbnVtT2N0YXZlcz0iMyIgc3RpdGNoVGlsZXM9InN0aXRjaCIvPgogIDwvZmlsdGVyPgogIDxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbHRlcj0idXJsKCNub2lzZUZpbHRlcikiLz4KPC9zdmc+')" }}
      />
    </div>
  );
}
